use std::collections::VecDeque;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use super::{
    ChatRequest, ChatResponse, EventStream, LlmClient, Message, StreamEvent, ToolDef, Usage,
};

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        // char_indices to avoid splitting a UTF-8 codepoint at the boundary.
        let cut = s
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= max)
            .last()
            .unwrap_or(0);
        format!(
            "{}\n... <truncated, {} more bytes>",
            &s[..cut],
            s.len() - cut
        )
    }
}

pub struct OpenAiCompatClient {
    url: String,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("pi/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("build reqwest client");
        let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
        Self { url, api_key, http }
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    #[serde(skip_serializing_if = "<[ToolDef]>::is_empty")]
    tools: &'a [ToolDef],
    max_tokens: u32,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Deserialize)]
struct WireResponse {
    choices: Vec<WireChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct WireChoice {
    message: Message,
    #[serde(default)]
    finish_reason: Option<String>,
}

// SSE streaming response types

#[derive(Deserialize)]
struct StreamResponse {
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct StreamChoice {
    delta: StreamDelta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCall>>,
}

#[derive(Deserialize)]
struct StreamToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamToolCallFunction>,
}

#[derive(Deserialize)]
struct StreamToolCallFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// Parse a complete SSE data payload (possibly assembled from multiple `data:` lines)
/// and push resulting `StreamEvent`s into `out`.
fn parse_sse_data(data: &str, out: &mut Vec<Result<StreamEvent>>) -> Result<()> {
    if data == "[DONE]" {
        out.push(Ok(StreamEvent::Done {
            finish_reason: "stop".to_owned(),
            usage: None,
        }));
        return Ok(());
    }
    let resp: StreamResponse = serde_json::from_str(data)
        .map_err(|e| anyhow!("SSE parse error: {e} data: {}", truncate(data, 500)))?;
    let choice = match resp.choices.into_iter().next() {
        Some(c) => c,
        None => return Ok(()),
    };
    if let Some(reason) = choice.finish_reason {
        out.push(Ok(StreamEvent::Done {
            finish_reason: reason,
            usage: resp.usage,
        }));
        return Ok(());
    }
    if let Some(content) = choice.delta.content
        && !content.is_empty()
    {
        out.push(Ok(StreamEvent::ContentDelta(content)));
    }
    if let Some(tool_calls) = choice.delta.tool_calls {
        for tc in tool_calls {
            out.push(Ok(StreamEvent::ToolCallDelta {
                index: tc.index,
                id: tc.id,
                function_name: tc.function.as_ref().and_then(|f| f.name.clone()),
                arguments_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
            }));
        }
    }
    Ok(())
}

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        let body = WireRequest {
            model: &req.model,
            messages: &req.messages,
            tools: &req.tools,
            max_tokens: req.max_tokens,
            stream: false,
        };

        let resp = self
            .http
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!(
                "API {} {}: {}",
                status.as_u16(),
                self.url,
                truncate(&text, 2000)
            ));
        }

        let parsed: WireResponse = serde_json::from_str(&text)
            .map_err(|e| anyhow!("decode response: {e}\nbody: {}", truncate(&text, 2000)))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("API returned no choices"))?;

        let finish_reason = choice
            .finish_reason
            .ok_or_else(|| anyhow!("API response missing finish_reason"))?;

        Ok(ChatResponse {
            message: choice.message,
            finish_reason,
            usage: parsed.usage,
        })
    }

    async fn complete_stream(&self, req: ChatRequest) -> Result<EventStream> {
        let body = WireRequest {
            model: &req.model,
            messages: &req.messages,
            tools: &req.tools,
            max_tokens: req.max_tokens,
            stream: true,
        };

        let resp = self
            .http
            .post(&self.url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            return Err(anyhow!(
                "API {} {}: {}",
                status.as_u16(),
                self.url,
                truncate(&text, 2000)
            ));
        }

        let stream = resp.bytes_stream();
        // Buffer partial lines across TCP chunks — SSE events can span multiple chunks.
        // Accumulate multi-line data fields (per SSE spec) before parsing.
        let event_stream = stream
            .scan(
                (VecDeque::<u8>::new(), Vec::<String>::new()),
                |(buf, data_lines), chunk| {
                    let chunk = match chunk {
                        Ok(b) => b,
                        Err(e) => {
                            return std::future::ready(Some(vec![Err(anyhow!(
                                "stream read error: {e}"
                            ))]));
                        }
                    };
                    buf.extend(chunk.iter().copied());

                    let mut out = Vec::new();
                    // Extract all complete lines (delimited by \n).
                    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
                        let line = String::from_utf8_lossy(&line_bytes);
                        let line = line.trim_end_matches('\r').trim_end_matches('\n').trim();

                        if line.is_empty() {
                            // Empty line = SSE event boundary: flush accumulated data.
                            if !data_lines.is_empty() {
                                let combined = data_lines.join("\n");
                                data_lines.clear();
                                if let Err(e) = parse_sse_data(&combined, &mut out) {
                                    out.push(Err(e));
                                }
                            }
                            continue;
                        }
                        if line.starts_with(':') {
                            continue; // SSE comment
                        }
                        // Handle "data: " and "data:" (no trailing space).
                        if let Some(rest) = line.strip_prefix("data:") {
                            let value = rest.strip_prefix(' ').unwrap_or(rest);
                            data_lines.push(value.to_owned());
                        }
                    }
                    std::future::ready(Some(out))
                },
            )
            .flat_map(stream::iter);

        Ok(Box::pin(event_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_string() {
        assert_eq!(truncate("hello", 100), "hello");
    }

    #[test]
    fn truncate_exact_limit() {
        let s = "abcde";
        assert_eq!(truncate(s, 5), "abcde");
    }

    #[test]
    fn truncate_long_string() {
        let s = "a".repeat(200);
        let result = truncate(&s, 100);
        assert!(result.contains("truncated"));
        assert!(result.contains("more bytes"));
    }

    #[test]
    fn truncate_utf8_boundary() {
        // 'ñ' is 2 bytes. Truncation should not split it.
        let s = "a".repeat(99) + "ñ";
        let result = truncate(&s, 100);
        assert!(result.starts_with(&"a".repeat(99)));
    }

    #[test]
    fn parse_sse_done() {
        let mut out = Vec::new();
        parse_sse_data("[DONE]", &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::Done { finish_reason, .. }) => assert_eq!(finish_reason, "stop"),
            _ => panic!("expected Done event"),
        }
    }

    #[test]
    fn parse_sse_content_delta() {
        let data = r#"{"choices":[{"delta":{"content":"hello"},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::ContentDelta(s)) => assert_eq!(s, "hello"),
            _ => panic!("expected ContentDelta"),
        }
    }

    #[test]
    fn parse_sse_empty_content_ignored() {
        let data = r#"{"choices":[{"delta":{"content":""},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        // Empty content should be ignored.
        assert!(out.is_empty());
    }

    #[test]
    fn parse_sse_no_content_no_toolcalls() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn parse_sse_finish_reason() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::Done { finish_reason, .. }) => assert_eq!(finish_reason, "stop"),
            _ => panic!("expected Done"),
        }
    }

    #[test]
    fn parse_sse_finish_reason_with_usage() {
        let data = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        match &out[0] {
            Ok(StreamEvent::Done { usage, .. }) => {
                assert!(usage.is_some());
                assert_eq!(usage.as_ref().unwrap().total_tokens, 15);
            }
            _ => panic!("expected Done with usage"),
        }
    }

    #[test]
    fn parse_sse_tool_call_delta() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"bash","arguments":"{\"c"}}]},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::ToolCallDelta {
                index,
                id,
                function_name,
                arguments_delta,
            }) => {
                assert_eq!(*index, 0);
                assert_eq!(id.as_deref(), Some("call_1"));
                assert_eq!(function_name.as_deref(), Some("bash"));
                assert_eq!(arguments_delta.as_deref(), Some("{\"c"));
            }
            _ => panic!("expected ToolCallDelta"),
        }
    }

    #[test]
    fn parse_sse_no_choices() {
        let data = r#"{"choices":[]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn parse_sse_invalid_json_errors() {
        let mut out = Vec::new();
        let result = parse_sse_data("not json", &mut out);
        assert!(result.is_err());
    }

    #[test]
    fn parse_sse_multiple_tool_calls() {
        let data = r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"a"}},{"index":1,"function":{"name":"b"}}]},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn parse_sse_content_and_tool_calls() {
        let data = r#"{"choices":[{"delta":{"content":"thinking","tool_calls":[{"index":0,"function":{"name":"bash"}}]},"finish_reason":null}]}"#;
        let mut out = Vec::new();
        parse_sse_data(data, &mut out).unwrap();
        // Should produce both a ContentDelta and a ToolCallDelta.
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn new_client_builds_url() {
        let client =
            OpenAiCompatClient::new("https://api.example.com/v1/".to_owned(), "key".to_owned());
        assert_eq!(client.url, "https://api.example.com/v1/chat/completions");
    }

    #[test]
    fn new_client_strips_trailing_slash() {
        let client =
            OpenAiCompatClient::new("https://api.example.com/v1".to_owned(), "key".to_owned());
        assert_eq!(client.url, "https://api.example.com/v1/chat/completions");
    }
}

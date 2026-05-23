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

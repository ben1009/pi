use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::StreamExt;

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
            .timeout(std::time::Duration::from_secs(600))
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
        let event_stream = stream.filter_map(move |chunk| {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => return Some(Err(anyhow!("stream read error: {e}"))),
            };
            let text = String::from_utf8_lossy(&bytes);
            // Parse SSE lines: each line is either "data: {...}" or "data: [DONE]"
            for line in text.split('\n') {
                let line = line.trim();
                if line.is_empty() || line.starts_with(':') {
                    continue; // skip empty lines and comments
                }
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        return Some(Ok(StreamEvent::Done {
                            finish_reason: "stop".to_owned(),
                            usage: None,
                        }));
                    }
                    match serde_json::from_str::<StreamResponse>(data) {
                        Ok(resp) => {
                            let choice = match resp.choices.into_iter().next() {
                                Some(c) => c,
                                None => continue,
                            };
                            if let Some(reason) = choice.finish_reason {
                                return Some(Ok(StreamEvent::Done {
                                    finish_reason: reason,
                                    usage: resp.usage,
                                }));
                            }
                            if let Some(content) = choice.delta.content
                                && !content.is_empty()
                            {
                                return Some(Ok(StreamEvent::ContentDelta(content)));
                            }
                            if let Some(tool_calls) = choice.delta.tool_calls
                                && let Some(tc) = tool_calls.into_iter().next()
                            {
                                return Some(Ok(StreamEvent::ToolCallDelta {
                                    index: tc.index,
                                    id: tc.id,
                                    function_name: tc
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.name.clone()),
                                    arguments_delta: tc
                                        .function
                                        .as_ref()
                                        .and_then(|f| f.arguments.clone()),
                                }));
                            }
                        }
                        Err(e) => {
                            return Some(Err(anyhow!("SSE parse error: {e} data: {data}")));
                        }
                    }
                }
            }
            None
        });

        Ok(Box::pin(event_stream))
    }
}

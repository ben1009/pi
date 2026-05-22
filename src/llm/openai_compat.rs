use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{ChatRequest, ChatResponse, LlmClient, Message, ToolDef, Usage};

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

#[async_trait]
impl LlmClient for OpenAiCompatClient {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        let body = WireRequest {
            model: &req.model,
            messages: &req.messages,
            tools: &req.tools,
            max_tokens: req.max_tokens,
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
}

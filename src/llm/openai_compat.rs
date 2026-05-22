use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{ChatRequest, ChatResponse, LlmClient, Message, ToolDef, Usage};

pub struct OpenAiCompatClient {
    url: String,
    api_key: String,
    http: reqwest::Client,
}

impl OpenAiCompatClient {
    pub fn new(base_url: String, api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("pi/", env!("CARGO_PKG_VERSION")))
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
            return Err(anyhow!("API {} {}: {}", status.as_u16(), self.url, text));
        }

        let parsed: WireResponse = serde_json::from_str(&text)
            .map_err(|e| anyhow!("decode response: {e}\nbody: {text}"))?;

        let choice = parsed
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("API returned no choices"))?;

        Ok(ChatResponse {
            message: choice.message,
            finish_reason: choice.finish_reason.unwrap_or_else(|| "stop".to_owned()),
            usage: parsed.usage,
        })
    }
}

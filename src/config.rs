use anyhow::Result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("missing API key: set ${env} for provider '{provider}'")]
    MissingKey { env: &'static str, provider: &'static str },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    Openai,
    Gemini,
    Deepseek,
    Kimi,
}

impl Provider {
    pub fn parse(s: &str) -> Result<Self, ConfigError> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Self::Anthropic,
            "openai" | "oai" => Self::Openai,
            "gemini" | "google" => Self::Gemini,
            "deepseek" | "ds" => Self::Deepseek,
            "kimi" | "moonshot" => Self::Kimi,
            other => return Err(ConfigError::UnknownProvider(other.to_owned())),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::Openai => "openai",
            Self::Gemini => "gemini",
            Self::Deepseek => "deepseek",
            Self::Kimi => "kimi",
        }
    }

    pub fn base_url(self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com/v1",
            Self::Openai => "https://api.openai.com/v1",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
            Self::Deepseek => "https://api.deepseek.com/v1",
            Self::Kimi => "https://api.moonshot.ai/v1",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-6",
            Self::Openai => "gpt-5",
            Self::Gemini => "gemini-2.5-pro",
            Self::Deepseek => "deepseek-chat",
            Self::Kimi => "kimi-k2-0905-preview",
        }
    }

    pub fn api_key_env(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::Openai => "OPENAI_API_KEY",
            Self::Gemini => "GEMINI_API_KEY",
            Self::Deepseek => "DEEPSEEK_API_KEY",
            Self::Kimi => "MOONSHOT_API_KEY",
        }
    }
}

pub struct ResolvedConfig {
    pub provider: Provider,
    pub model: String,
    pub api_key: String,
    pub base_url: String,
    pub max_tokens: u32,
    pub yolo: bool,
    pub max_turns: u32,
    pub max_tool_output: usize,
}

pub struct ResolveInput<'a> {
    pub provider: Option<&'a str>,
    pub model: Option<&'a str>,
    pub max_tokens: Option<u32>,
    pub yolo: bool,
    pub max_turns: u32,
    pub max_tool_output: usize,
}

pub fn resolve(input: ResolveInput<'_>) -> Result<ResolvedConfig, ConfigError> {
    let provider = match input.provider {
        Some(s) => Provider::parse(s)?,
        None => Provider::Anthropic,
    };

    let model = input
        .model
        .map(str::to_owned)
        .unwrap_or_else(|| provider.default_model().to_owned());

    let max_tokens = input.max_tokens.unwrap_or(8192);

    let api_key = std::env::var(provider.api_key_env())
        .ok()
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
        .ok_or(ConfigError::MissingKey {
            env: provider.api_key_env(),
            provider: provider.name(),
        })?;

    Ok(ResolvedConfig {
        provider,
        model,
        api_key,
        base_url: provider.base_url().to_owned(),
        max_tokens,
        yolo: input.yolo,
        max_turns: input.max_turns,
        max_tool_output: input.max_tool_output,
    })
}

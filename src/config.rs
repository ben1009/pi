use anyhow::{Result, anyhow};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    Openai,
    Gemini,
    Deepseek,
    Kimi,
}

impl Provider {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s.to_ascii_lowercase().as_str() {
            "anthropic" | "claude" => Self::Anthropic,
            "openai" | "oai" => Self::Openai,
            "gemini" | "google" => Self::Gemini,
            "deepseek" | "ds" => Self::Deepseek,
            "kimi" | "moonshot" => Self::Kimi,
            other => return Err(anyhow!("unknown provider: {other}")),
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
}

pub fn resolve(
    provider: Option<&str>,
    model: Option<&str>,
    max_tokens: Option<u32>,
) -> Result<ResolvedConfig> {
    let provider = match provider {
        Some(s) => Provider::parse(s)?,
        None => match std::env::var("PI_PROVIDER") {
            Ok(s) => Provider::parse(&s)?,
            Err(_) => Provider::Anthropic,
        },
    };

    let model = model
        .map(str::to_owned)
        .or_else(|| std::env::var("PI_MODEL").ok())
        .unwrap_or_else(|| provider.default_model().to_owned());

    let max_tokens = max_tokens
        .or_else(|| std::env::var("PI_MAX_TOKENS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(8192);

    let api_key = std::env::var(provider.api_key_env()).map_err(|_| {
        anyhow!(
            "missing API key: set ${} for provider '{}'",
            provider.api_key_env(),
            provider.name()
        )
    })?;

    Ok(ResolvedConfig {
        provider,
        model,
        api_key,
        base_url: provider.base_url().to_owned(),
        max_tokens,
    })
}

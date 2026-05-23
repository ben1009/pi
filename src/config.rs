use anyhow::Result;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("missing API key: set ${env} for provider '{provider}'")]
    MissingKey {
        env: &'static str,
        provider: &'static str,
    },
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

#[derive(Debug)]
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

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn provider_parse_known() {
        assert_eq!(Provider::parse("anthropic").unwrap(), Provider::Anthropic);
        assert_eq!(Provider::parse("claude").unwrap(), Provider::Anthropic);
        assert_eq!(Provider::parse("openai").unwrap(), Provider::Openai);
        assert_eq!(Provider::parse("oai").unwrap(), Provider::Openai);
        assert_eq!(Provider::parse("gemini").unwrap(), Provider::Gemini);
        assert_eq!(Provider::parse("google").unwrap(), Provider::Gemini);
        assert_eq!(Provider::parse("deepseek").unwrap(), Provider::Deepseek);
        assert_eq!(Provider::parse("ds").unwrap(), Provider::Deepseek);
        assert_eq!(Provider::parse("kimi").unwrap(), Provider::Kimi);
        assert_eq!(Provider::parse("moonshot").unwrap(), Provider::Kimi);
    }

    #[test]
    fn provider_parse_case_insensitive() {
        assert_eq!(Provider::parse("Anthropic").unwrap(), Provider::Anthropic);
        assert_eq!(Provider::parse("OPENAI").unwrap(), Provider::Openai);
        assert_eq!(Provider::parse("DeepSeek").unwrap(), Provider::Deepseek);
    }

    #[test]
    fn provider_parse_unknown() {
        assert!(Provider::parse("unknown").is_err());
        assert!(Provider::parse("").is_err());
    }

    #[test]
    fn provider_properties() {
        assert_eq!(Provider::Anthropic.name(), "anthropic");
        assert_eq!(Provider::Openai.api_key_env(), "OPENAI_API_KEY");
        assert_eq!(
            Provider::Gemini.base_url(),
            "https://generativelanguage.googleapis.com/v1beta/openai"
        );
        assert_eq!(Provider::Kimi.default_model(), "kimi-k2-0905-preview");
    }

    #[test]
    fn resolve_defaults() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::set_var("ANTHROPIC_API_KEY", "test-key") };
        let cfg = resolve(ResolveInput {
            provider: None,
            model: None,
            max_tokens: None,
            yolo: false,
            max_turns: 50,
            max_tool_output: 4096,
        })
        .unwrap();
        assert_eq!(cfg.provider, Provider::Anthropic);
        assert_eq!(cfg.model, "claude-sonnet-4-6");
        assert_eq!(cfg.max_tokens, 8192);
        assert_eq!(cfg.api_key, "test-key");
        unsafe { std::env::remove_var("ANTHROPIC_API_KEY") };
    }

    #[test]
    fn resolve_missing_key_errors() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { std::env::remove_var("OPENAI_API_KEY") };
        let err = resolve(ResolveInput {
            provider: Some("openai"),
            model: None,
            max_tokens: None,
            yolo: false,
            max_turns: 50,
            max_tool_output: 4096,
        })
        .unwrap_err();
        assert!(err.to_string().contains("OPENAI_API_KEY"));
    }
}

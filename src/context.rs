use crate::config::Provider;

/// Return the context window size (in tokens) for the given provider + model.
pub fn context_window(provider: Provider, model: &str) -> u32 {
    // Exact model matches first.
    let exact = match model {
        // Anthropic
        "claude-opus-4-6" | "claude-opus-4-5-20250620" => 200_000,
        "claude-sonnet-4-6" | "claude-sonnet-4-5-20250620" => 200_000,
        "claude-haiku-4-5-20251001" => 200_000,
        // OpenAI
        "gpt-5" => 256_000,
        "gpt-4o" | "gpt-4o-2024-08-06" | "gpt-4o-2024-11-20" => 128_000,
        "gpt-4o-mini" | "gpt-4o-mini-2024-07-18" => 128_000,
        "gpt-4-turbo" | "gpt-4-turbo-2024-04-09" => 128_000,
        "gpt-4-0125-preview" | "gpt-4-1106-preview" => 128_000,
        "gpt-4-32k" => 32_768,
        "gpt-4" | "gpt-4-0613" => 8_192,
        "gpt-3.5-turbo" | "gpt-3.5-turbo-0125" => 16_385,
        "o1" | "o1-2024-12-17" => 200_000,
        "o1-mini" | "o1-mini-2024-09-12" => 128_000,
        "o3" | "o3-2025-04-16" => 200_000,
        "o3-mini" | "o3-mini-2025-01-31" => 200_000,
        "o4-mini" | "o4-mini-2025-04-16" => 200_000,
        // Gemini
        "gemini-2.5-pro" => 1_048_576,
        "gemini-2.5-flash" => 1_048_576,
        "gemini-2.0-flash" => 1_048_576,
        "gemini-1.5-pro" => 2_097_152,
        "gemini-1.5-flash" => 1_048_576,
        // DeepSeek
        "deepseek-chat" | "deepseek-coder" => 128_000,
        "deepseek-reasoner" => 128_000,
        // Kimi
        "kimi-k2-0905-preview" | "kimi-k2-turbo-preview" => 131_072,
        "moonshot-v1-128k" => 131_072,
        "moonshot-v1-32k" => 32_768,
        "moonshot-v1-8k" => 8_192,
        _ => 0, // Fall through to prefix match.
    };
    if exact > 0 {
        return exact;
    }

    // Provider-based fallback (most reliable when user overrides model name).
    match provider {
        Provider::Anthropic => 200_000,
        Provider::Openai => 128_000,
        Provider::Gemini => 1_048_576,
        Provider::Deepseek => 128_000,
        Provider::Kimi => 131_072,
    }
}

const WARN_THRESHOLD: f64 = 0.80;

/// Tracks context window usage across turns.
pub struct ContextTracker {
    context_window: u32,
    last_prompt_tokens: u32,
}

impl ContextTracker {
    pub fn new(context_window: u32) -> Self {
        Self {
            context_window,
            last_prompt_tokens: 0,
        }
    }

    /// Update with the latest prompt_tokens from the API response.
    /// Returns a warning message if usage exceeds the threshold.
    pub fn update(&mut self, prompt_tokens: u32) -> Option<String> {
        self.last_prompt_tokens = prompt_tokens;
        if self.context_window == 0 {
            return None;
        }
        let ratio = prompt_tokens as f64 / self.context_window as f64;
        if ratio >= WARN_THRESHOLD {
            let pct = (ratio * 100.0) as u32;
            Some(format!(
                "pi: context window {pct}% full ({}/{}) — consider /compact",
                prompt_tokens, self.context_window
            ))
        } else {
            None
        }
    }

    pub fn last_prompt_tokens(&self) -> u32 {
        self.last_prompt_tokens
    }
}

mod config;
mod confirm;
mod llm;
mod tools;

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::config::{ConfigError, ResolveInput, ResolvedConfig};
use crate::llm::openai_compat::OpenAiCompatClient;
use crate::llm::{ChatRequest, ChatResponse, LlmClient, Message, Role, Usage};
use crate::tools::{Registry, ToolCtx};

#[derive(Parser, Debug)]
#[command(name = "pi", about = "a multi-LLM coding agent", version)]
struct Cli {
    /// Provider: anthropic, openai, gemini, deepseek, kimi
    #[arg(short = 'P', long, env = "PI_PROVIDER")]
    provider: Option<String>,

    /// Model id (defaults per provider)
    #[arg(short = 'm', long, env = "PI_MODEL")]
    model: Option<String>,

    /// Max output tokens
    #[arg(long, env = "PI_MAX_TOKENS")]
    max_tokens: Option<u32>,

    /// Skip y/n confirmations on bash and out-of-CWD writes/edits.
    #[arg(short = 'y', long)]
    yolo: bool,

    /// Cap on tool-use iterations per user turn.
    #[arg(long, env = "PI_MAX_TURNS", default_value_t = 50)]
    max_turns: u32,

    /// Per-tool-result character cap (excess truncated).
    #[arg(long, env = "PI_MAX_TOOL_OUTPUT", default_value_t = 100_000)]
    max_tool_output: usize,

    /// One-shot prompt: send, print final assistant text, exit.
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Print rendered system prompt and exit 0.
    #[arg(long)]
    print_system_prompt: bool,
}

const EXIT_API_OR_TURNS: i32 = 1;
const EXIT_MISSING_KEY: i32 = 2;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    if cli.print_system_prompt {
        println!("{}", system_prompt());
        return;
    }

    let cfg = match config::resolve(ResolveInput {
        provider: cli.provider.as_deref(),
        model: cli.model.as_deref(),
        max_tokens: cli.max_tokens,
        yolo: cli.yolo,
        max_turns: cli.max_turns,
        max_tool_output: cli.max_tool_output,
    }) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pi: {e}");
            let code = match e {
                ConfigError::MissingKey { .. } => EXIT_MISSING_KEY,
                ConfigError::UnknownProvider(_) => EXIT_API_OR_TURNS,
            };
            std::process::exit(code);
        }
    };

    let code = run(cfg, cli.prompt).await;
    std::process::exit(code);
}

fn system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<unknown>".to_owned());
    let os = std::env::consts::OS;
    let date = today_utc();
    format!(
        "You are pi, a CLI coding agent. You help the user edit and run code in their working directory.\n\n\
         Working directory: {cwd}\n\
         Operating system: {os}\n\
         Date: {date}\n\n\
         Prefer using the provided tools (bash, read, write, edit) over guessing. \
         When a tool returns an error, read the error and try a different approach. \
         Be concise."
    )
}

fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

struct ReplState {
    last_usage: Option<Usage>,
}

async fn run(cfg: ResolvedConfig, one_shot: Option<String>) -> i32 {
    let client = OpenAiCompatClient::new(cfg.base_url.clone(), cfg.api_key.clone());
    let registry = Registry::with_defaults();
    let mut messages = vec![Message::system(system_prompt())];

    eprintln!(
        "pi: provider={} model={} max_tokens={}",
        cfg.provider.name(),
        cfg.model,
        cfg.max_tokens
    );

    if let Some(prompt) = one_shot {
        messages.push(Message::user(prompt));
        match drive(&client, &cfg, &registry, &mut messages).await {
            Ok(Some(resp)) => {
                let text = resp.message.content.as_deref().unwrap_or("");
                if !text.is_empty() {
                    println!("{text}");
                }
                if resp.finish_reason != "stop" {
                    eprintln!("pi: finish_reason={} (incomplete)", resp.finish_reason);
                    return EXIT_API_OR_TURNS;
                }
                if text.is_empty() {
                    eprintln!("pi: model produced no text");
                    return EXIT_API_OR_TURNS;
                }
                0
            }
            Ok(None) => {
                eprintln!("pi: max turns reached");
                EXIT_API_OR_TURNS
            }
            Err(e) => {
                eprintln!("pi: {e}");
                EXIT_API_OR_TURNS
            }
        }
    } else {
        repl(&client, &cfg, &registry, messages).await
    }
}

async fn repl(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    mut messages: Vec<Message>,
) -> i32 {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let mut state = ReplState { last_usage: None };

    loop {
        print!("> ");
        std::io::stdout().flush().ok();

        let line = match reader.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => return 0,
            Err(e) => {
                eprintln!("pi: stdin: {e}");
                return EXIT_API_OR_TURNS;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/exit" | "/quit" => return 0,
            "/clear" => {
                messages.truncate(1); // keep system prompt
                state.last_usage = None;
                eprintln!("pi: cleared.");
                continue;
            }
            "/tokens" => {
                match &state.last_usage {
                    Some(u) => println!(
                        "prompt={} completion={} total={}",
                        u.prompt_tokens, u.completion_tokens, u.total_tokens
                    ),
                    None => println!("(no usage yet)"),
                }
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(line));
        match drive(client, cfg, registry, &mut messages).await {
            Ok(Some(resp)) => {
                if let Some(u) = &resp.usage {
                    state.last_usage = Some(u.clone());
                }
                let text = resp.message.content.as_deref().unwrap_or("");
                if !text.is_empty() {
                    println!("{text}");
                } else {
                    eprintln!("pi: model produced no text");
                }
            }
            Ok(None) => {
                eprintln!("pi: max turns reached");
            }
            Err(e) => {
                eprintln!("pi: {e}");
                // Drop the failed user turn so retry doesn't pile up. Stop at
                // index 1 so the system prompt at index 0 always survives.
                while messages.len() > 1 {
                    let role = messages.last().unwrap().role.clone();
                    messages.pop();
                    if matches!(role, Role::User) {
                        break;
                    }
                }
            }
        }
    }
}

/// Run the tool-use loop until the model returns an assistant turn with no tool
/// calls, or `max_turns` is exhausted. The terminal assistant message is left
/// on `messages` and also returned for display/usage.
///
/// Returns:
///   Ok(Some(resp)) — model produced a final assistant message.
///   Ok(None)       — max_turns reached.
///   Err(e)         — API or fatal error.
async fn drive(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    messages: &mut Vec<Message>,
) -> Result<Option<ChatResponse>> {
    let tool_ctx = ToolCtx {
        yolo: cfg.yolo,
        max_output: cfg.max_tool_output,
    };

    for _ in 0..cfg.max_turns {
        let resp = send_full(client, cfg, registry, messages).await?;
        let calls = resp
            .message
            .tool_calls
            .clone()
            .unwrap_or_default();

        if calls.is_empty() {
            // Don't pollute history with an empty assistant turn — the next
            // request would 400 on most OpenAI-compat servers.
            let has_content = resp.message.content.as_deref().is_some_and(|s| !s.is_empty());
            if has_content {
                messages.push(resp.message.clone());
            }
            return Ok(Some(resp));
        }

        messages.push(resp.message.clone());

        for call in calls {
            let content = match registry.get(&call.function.name) {
                None => format!("Error: unknown tool '{}'", call.function.name),
                Some(tool) => {
                    let args: serde_json::Value =
                        match serde_json::from_str(&call.function.arguments) {
                            Ok(v) => v,
                            Err(e) => {
                                let msg = format!(
                                    "Error: invalid JSON arguments for '{}': {e}",
                                    call.function.name
                                );
                                messages.push(tool_message(&call.id, msg));
                                continue;
                            }
                        };
                    match tool.run(tool_ctx, args).await {
                        Ok(s) => s,
                        Err(e) => format!("Error: {} failed: {e}", call.function.name),
                    }
                }
            };
            messages.push(tool_message(&call.id, content));
        }
    }

    Ok(None)
}

fn tool_message(id: &str, content: String) -> Message {
    Message {
        role: Role::Tool,
        content: Some(content),
        tool_calls: None,
        tool_call_id: Some(id.to_owned()),
    }
}

async fn send_full(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    messages: &[Message],
) -> Result<ChatResponse> {
    let req = ChatRequest {
        model: cfg.model.clone(),
        messages: messages.to_vec(),
        tools: registry.definitions(),
        max_tokens: cfg.max_tokens,
    };
    client.complete(req).await
}

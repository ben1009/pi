use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use pi::config::{self, ConfigError, ResolveInput, ResolvedConfig};
use pi::llm::openai_compat::OpenAiCompatClient;
use pi::llm::{ChatRequest, ChatResponse, LlmClient, Message, Role, Usage};
use pi::system_prompt;
use pi::tools::{Registry, ToolCtx};

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
        match repl(&client, &cfg, &registry, messages).await {
            Ok(code) => code,
            Err(e) => {
                eprintln!("pi: {e}");
                EXIT_API_OR_TURNS
            }
        }
    }
}

fn history_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("pi").join("history"))
}

async fn repl(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    mut messages: Vec<Message>,
) -> Result<i32> {
    let history_path = history_path();
    if let Some(p) = &history_path {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    let mut rl = DefaultEditor::new()?;
    if let Some(p) = &history_path {
        let _ = rl.load_history(p);
    }
    let mut state = ReplState { last_usage: None };

    loop {
        let mut editor = Some(rl);
        let read = tokio::task::spawn_blocking(move || {
            let mut ed = editor.take().unwrap();
            let res = ed.readline("> ");
            (ed, res)
        })
        .await?;
        rl = read.0;

        let line = match read.1 {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => {
                if let Some(p) = &history_path {
                    let _ = rl.save_history(p);
                }
                return Ok(0);
            }
            Err(e) => {
                eprintln!("pi: stdin: {e}");
                return Ok(EXIT_API_OR_TURNS);
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line.as_str());
        if let Some(p) = &history_path {
            let _ = rl.append_history(p);
        }
        match trimmed {
            "/exit" | "/quit" => {
                if let Some(p) = &history_path {
                    let _ = rl.save_history(p);
                }
                return Ok(0);
            }
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

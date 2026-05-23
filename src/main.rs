use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pi_rs::{
    config::{self, ConfigError, ResolveInput, ResolvedConfig},
    llm::{
        ChatRequest, ChatResponse, LlmClient, Message, Role, Usage,
        openai_compat::OpenAiCompatClient,
    },
    session, system_prompt,
    tools::{Registry, ToolCtx},
};
use rustyline::{DefaultEditor, config::Configurer, error::ReadlineError};

#[derive(Parser, Debug)]
#[command(name = "pi-rs", about = "a multi-LLM coding agent", version)]
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

    /// Resume a previous session by ID.
    #[arg(long)]
    resume: Option<String>,

    /// List saved sessions and exit.
    #[arg(long)]
    sessions: bool,

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

    if cli.sessions {
        match session::list() {
            Ok(sessions) => {
                if sessions.is_empty() {
                    println!("(no saved sessions)");
                } else {
                    for s in &sessions {
                        let preview = s.first_prompt.chars().take(60).collect::<String>();
                        println!("{}  {}  {}", s.id, s.created_at, preview);
                    }
                }
            }
            Err(e) => eprintln!("pi-rs: {e}"),
        }
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
            eprintln!("pi-rs: {e}");
            let code = match e {
                ConfigError::MissingKey { .. } => EXIT_MISSING_KEY,
                ConfigError::UnknownProvider(_) => EXIT_API_OR_TURNS,
            };
            std::process::exit(code);
        }
    };

    let code = run(cfg, cli.prompt, cli.resume).await;
    std::process::exit(code);
}

struct ReplState {
    last_usage: Option<Usage>,
    session_id: Option<String>,
    session_created_at: Option<String>,
}

async fn run(cfg: ResolvedConfig, one_shot: Option<String>, resume_id: Option<String>) -> i32 {
    let client = OpenAiCompatClient::new(cfg.base_url.clone(), cfg.api_key.clone());
    let registry = Registry::with_defaults();
    let (mut messages, resume_created_at) = if let Some(id) = &resume_id {
        match session::load(id) {
            Ok(s) => {
                eprintln!("pi: resumed session {id} ({} messages)", s.messages.len());
                (s.messages, Some(s.created_at))
            }
            Err(e) => {
                eprintln!("pi-rs: {e}");
                return EXIT_API_OR_TURNS;
            }
        }
    } else {
        (vec![Message::system(system_prompt())], None)
    };

    eprintln!(
        "pi: provider={} model={} max_tokens={}",
        cfg.provider.name(),
        cfg.model,
        cfg.max_tokens
    );

    if let Some(prompt) = one_shot {
        messages.push(Message::user(prompt));
        match drive(&client, &cfg, &registry, &mut messages, false).await {
            Ok(Some(resp)) => {
                let text = resp.message.content.as_deref().unwrap_or("");
                if !text.is_empty() {
                    println!("{text}");
                }
                if resp.finish_reason != "stop" {
                    eprintln!("pi-rs: finish_reason={} (incomplete)", resp.finish_reason);
                    return EXIT_API_OR_TURNS;
                }
                if text.is_empty() {
                    eprintln!("pi-rs: model produced no text");
                    return EXIT_API_OR_TURNS;
                }
                0
            }
            Ok(None) => {
                eprintln!("pi-rs: max turns reached");
                EXIT_API_OR_TURNS
            }
            Err(e) => {
                eprintln!("pi-rs: {e}");
                EXIT_API_OR_TURNS
            }
        }
    } else {
        match repl(
            &client,
            &cfg,
            &registry,
            messages,
            resume_id,
            resume_created_at,
        )
        .await
        {
            Ok(code) => code,
            Err(e) => {
                eprintln!("pi-rs: {e}");
                EXIT_API_OR_TURNS
            }
        }
    }
}

fn history_path() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("pi-rs").join("history"))
}

/// ISO 8601 timestamp (UTC, second precision).
fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let sod = (secs % 86_400) as u32;
    let h = sod / 3600;
    let m = (sod % 3600) / 60;
    let s = sod % 60;
    let (y, mo, d) = days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
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

async fn repl(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    mut messages: Vec<Message>,
    initial_session_id: Option<String>,
    initial_created_at: Option<String>,
) -> Result<i32> {
    let history_path = history_path();
    if let Some(p) = &history_path
        && let Some(parent) = p.parent()
        && let Err(e) = std::fs::create_dir_all(parent)
    {
        eprintln!(
            "pi: warning: could not create history dir {}: {e}",
            parent.display()
        );
    }

    let mut rl = DefaultEditor::new()?;
    rl.set_max_history_size(1000)?;
    if let Some(p) = &history_path {
        let _ = rl.load_history(p);
    }
    let mut state = ReplState {
        last_usage: None,
        session_id: initial_session_id,
        session_created_at: initial_created_at,
    };
    let mut retry_input: Option<String> = None;

    /// Save the current session, initializing id/created_at on first call.
    fn save_session(state: &mut ReplState, messages: &[Message], label: &str) {
        let first = messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .and_then(|m| m.content.clone())
            .unwrap_or_default();
        let id = state.session_id.get_or_insert_with(session::new_id).clone();
        let created_at = state
            .session_created_at
            .get_or_insert_with(chrono_now)
            .clone();
        let sess = session::Session {
            id,
            created_at,
            first_prompt: first,
            messages: messages.to_vec(),
        };
        match session::save(&sess) {
            Ok(path) => {
                if label == "save" {
                    eprintln!("pi: saved session {} → {}", sess.id, path.display());
                }
            }
            Err(e) => eprintln!("pi-rs: {label} failed: {e}"),
        }
    }

    loop {
        let initial = retry_input.clone().unwrap_or_default();
        let read = tokio::task::spawn_blocking(move || {
            let res = rl.readline_with_initial("> ", (&initial, ""));
            (rl, res)
        })
        .await?;
        rl = read.0;

        let line = match read.1 {
            Ok(l) => {
                retry_input = None;
                l
            }
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => {
                // Compact the on-disk history once on graceful exit so
                // append_history's growth stays bounded by max_history_size.
                if let Some(p) = &history_path {
                    let _ = rl.save_history(p);
                }
                return Ok(0);
            }
            Err(e) => {
                eprintln!("pi-rs: stdin: {e}");
                return Ok(EXIT_API_OR_TURNS);
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
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
                state.session_id = None;
                state.session_created_at = None;
                eprintln!("pi-rs: cleared.");
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
            "/save" => {
                save_session(&mut state, &messages, "save");
                continue;
            }
            "/sessions" => {
                match session::list() {
                    Ok(sessions) => {
                        if sessions.is_empty() {
                            println!("(no saved sessions)");
                        } else {
                            for s in &sessions {
                                let preview = s.first_prompt.chars().take(60).collect::<String>();
                                println!("{}  {}  {}", s.id, s.created_at, preview);
                            }
                        }
                    }
                    Err(e) => eprintln!("pi-rs: {e}"),
                }
                continue;
            }
            _ => {}
        }
        // Persist actual prompts only — slash commands stay out of history.
        let _ = rl.add_history_entry(line.as_str());
        if let Some(p) = &history_path {
            let _ = rl.append_history(p);
        }

        messages.push(Message::user(line));
        match drive(client, cfg, registry, &mut messages, true).await {
            Ok(Some(resp)) => {
                if let Some(u) = &resp.usage {
                    state.last_usage = Some(u.clone());
                }
                let text = resp.message.content.as_deref().unwrap_or("");
                if !text.is_empty() {
                    println!("{text}");
                } else {
                    eprintln!("pi-rs: model produced no text");
                }
                // Auto-save session after successful turn.
                save_session(&mut state, &messages, "auto-save");
            }
            Ok(None) => {
                eprintln!("pi-rs: max turns reached");
            }
            Err(e) => {
                eprintln!("pi-rs: {e}");
                // Drop the failed user turn so retry doesn't pile up. Stop at
                // index 1 so the system prompt at index 0 always survives.
                // Save the user input so readline_with_initial can prefill it.
                while messages.len() > 1 {
                    let msg = messages.last().unwrap();
                    if matches!(msg.role, Role::User) {
                        retry_input = msg.content.clone();
                        messages.pop();
                        break;
                    }
                    messages.pop();
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
    stream_stderr: bool,
) -> Result<Option<ChatResponse>> {
    let tool_ctx = ToolCtx {
        yolo: cfg.yolo,
        max_output: cfg.max_tool_output,
        stream_stderr,
    };

    for _ in 0..cfg.max_turns {
        let resp = send_full(client, cfg, registry, messages).await?;
        let calls = resp.message.tool_calls.clone().unwrap_or_default();

        if calls.is_empty() {
            // Don't pollute history with an empty assistant turn — the next
            // request would 400 on most OpenAI-compat servers.
            let has_content = resp
                .message
                .content
                .as_deref()
                .is_some_and(|s| !s.is_empty());
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

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use clap::Parser;
use pi_rs::{
    config::{self, ConfigError, Provider, ResolveInput, ResolvedConfig},
    context::{ContextTracker, context_window},
    days_to_ymd,
    llm::{
        ChatRequest, ChatResponse, LlmClient, Message, Role, StreamEvent, Usage,
        anthropic::AnthropicNativeClient, openai_compat::OpenAiCompatClient,
    },
    mcp, session, system_prompt,
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

    /// MCP server config (JSON string or path to JSON file). Repeatable.
    /// Example: --mcp-server
    /// '{"name":"fs","command":"npx","args":["-y","@modelcontextprotocol/server-filesystem","/tmp"
    /// ]}'
    #[arg(long = "mcp-server", env = "PI_MCP_SERVERS")]
    mcp_server: Vec<String>,
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

    let code = run(cfg, cli.prompt, cli.resume, cli.mcp_server).await;
    std::process::exit(code);
}

struct ReplState {
    last_usage: Option<Usage>,
    session_id: Option<String>,
    session_created_at: Option<String>,
    ctx: ContextTracker,
}

async fn run(
    cfg: ResolvedConfig,
    one_shot: Option<String>,
    resume_id: Option<String>,
    mcp_server_args: Vec<String>,
) -> i32 {
    let client: Box<dyn LlmClient> = match cfg.provider {
        Provider::AnthropicNative => Box::new(AnthropicNativeClient::new(cfg.api_key.clone())),
        _ => Box::new(OpenAiCompatClient::new(
            cfg.base_url.clone(),
            cfg.api_key.clone(),
        )),
    };
    let mut registry = Registry::with_defaults();

    // Connect to MCP servers if configured.
    if !mcp_server_args.is_empty() {
        match mcp::parse_mcp_configs(&mcp_server_args) {
            Ok(configs) => {
                if let Err(e) = mcp::connect_servers(&configs, &mut registry).await {
                    eprintln!("pi: warning: MCP server connection error: {e}");
                }
            }
            Err(e) => {
                eprintln!("pi: warning: failed to parse MCP server config: {e}");
            }
        }
    }
    let (mut messages, resume_created_at) = if let Some(id) = &resume_id {
        match session::load(id) {
            Ok(mut s) => {
                eprintln!("pi: resumed session {id} ({} messages)", s.messages.len());
                // Replace stale system prompt (old date/CWD) with a fresh one.
                if !s.messages.is_empty() && matches!(s.messages[0].role, Role::System) {
                    s.messages[0] = Message::system(system_prompt());
                }
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
        match drive(&*client, &cfg, &registry, &mut messages, false).await {
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
            &*client,
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

async fn repl(
    client: &dyn LlmClient,
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
        ctx: ContextTracker::new(context_window(cfg.provider, &cfg.model)),
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
            // TODO: debounce auto-save or use Arc<[Message]> to avoid cloning on every turn.
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
                state.ctx = ContextTracker::new(context_window(cfg.provider, &cfg.model));
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
            "/context" => {
                let used = state.ctx.last_prompt_tokens();
                let window = context_window(cfg.provider, &cfg.model);
                if used == 0 {
                    println!("(no usage yet)");
                } else {
                    let pct = (used as u64 * 100 / window as u64) as u32;
                    println!("{used}/{window} tokens ({pct}%)");
                }
                continue;
            }
            "/compact" => {
                match compact(client, cfg, registry, &mut messages).await {
                    Ok(n) => {
                        eprintln!("pi: compacted {n} messages into summary");
                        state.ctx = ContextTracker::new(context_window(cfg.provider, &cfg.model));
                        state.last_usage = None;
                        save_session(&mut state, &messages, "auto-save");
                    }
                    Err(e) => eprintln!("pi-rs: compact failed: {e}"),
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
                    if let Some(warn) = state.ctx.update(u.prompt_tokens) {
                        eprintln!("{warn}");
                    }
                }
                // In streaming mode, content was already printed to stderr.
                // Only print here for non-streaming (one-shot) mode.
                let text = resp.message.content.as_deref().unwrap_or("");
                if text.is_empty() && resp.message.tool_calls.is_none() {
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
    client: &dyn LlmClient,
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
        let resp = if stream_stderr {
            send_streaming(client, cfg, registry, messages).await?
        } else {
            send_full(client, cfg, registry, messages).await?
        };
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
    client: &dyn LlmClient,
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

/// Stream a chat completion, printing content deltas to stdout as they arrive.
/// Accumulates tool call deltas into complete ToolCall structs.
async fn send_streaming(
    client: &dyn LlmClient,
    cfg: &ResolvedConfig,
    registry: &Registry,
    messages: &[Message],
) -> Result<ChatResponse> {
    use std::collections::BTreeMap;

    use futures_util::StreamExt;

    let req = ChatRequest {
        model: cfg.model.clone(),
        messages: messages.to_vec(),
        tools: registry.definitions(),
        max_tokens: cfg.max_tokens,
    };

    let mut stream = client.complete_stream(req).await?;
    let mut content = String::new();
    let mut tool_calls: BTreeMap<usize, (String, String, String)> = BTreeMap::new(); // index -> (id, name, args)
    let mut finish_reason = "stop".to_owned();
    let mut usage = None;

    while let Some(event) = stream.next().await {
        match event? {
            StreamEvent::ContentDelta(text) => {
                content.push_str(&text);
                eprint!("{text}");
            }
            StreamEvent::ToolCallDelta {
                index,
                id,
                function_name,
                arguments_delta,
            } => {
                let entry = tool_calls.entry(index).or_default();
                if let Some(id) = id {
                    entry.0 = id;
                }
                if let Some(name) = function_name {
                    entry.1 = name;
                }
                if let Some(args) = arguments_delta {
                    entry.2.push_str(&args);
                }
            }
            StreamEvent::Done {
                finish_reason: reason,
                usage: u,
            } => {
                finish_reason = reason;
                usage = u;
                break;
            }
            StreamEvent::Error(msg) => {
                return Err(anyhow!("stream error: {msg}"));
            }
        }
    }

    // Flush newline after streaming content
    if !content.is_empty() {
        eprintln!();
    }

    let message = Message {
        role: Role::Assistant,
        content: if content.is_empty() {
            None
        } else {
            Some(content)
        },
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            let calls: Vec<_> = tool_calls
                .into_values()
                .filter(|(id, name, _)| !id.is_empty() && !name.is_empty())
                .map(|(id, name, args)| pi_rs::llm::ToolCall {
                    id,
                    kind: "function".to_owned(),
                    function: pi_rs::llm::ToolCallFunction {
                        name,
                        arguments: args,
                    },
                })
                .collect();
            if calls.is_empty() { None } else { Some(calls) }
        },
        tool_call_id: None,
    };

    Ok(ChatResponse {
        message,
        finish_reason,
        usage,
    })
}

/// Summarize the conversation history into a single message.
/// Returns the number of messages that were compacted.
async fn compact(
    client: &dyn LlmClient,
    cfg: &ResolvedConfig,
    _registry: &Registry,
    messages: &mut Vec<Message>,
) -> Result<usize> {
    let original_len = messages.len();
    if original_len < 3 {
        anyhow::bail!("nothing to compact (need at least 3 messages including system prompt)");
    }
    let non_system = original_len - 1; // exclude system prompt

    // Build a fresh message list: system prompt + conversation dump + summarization request.
    let history_text: String = messages
        .iter()
        .filter(|m| !matches!(m.role, Role::System))
        .map(|m| {
            let role = match m.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::Tool => "Tool",
                Role::System => "System",
            };
            let mut parts = Vec::new();
            if let Some(content) = &m.content
                && !content.is_empty()
            {
                parts.push(content.clone());
            }
            if let Some(calls) = &m.tool_calls {
                for tc in calls {
                    parts.push(format!(
                        "[tool_call: {}({})]",
                        tc.function.name, tc.function.arguments
                    ));
                }
            }
            let text = parts.join("\n");
            format!("[{role}]: {text}")
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    // Truncate to avoid exceeding the summarization model's context window.
    const MAX_HISTORY_CHARS: usize = 100_000;
    let history_text = if history_text.len() > MAX_HISTORY_CHARS {
        let truncated = &history_text[..history_text
            .char_indices()
            .map(|(i, _)| i)
            .take_while(|&i| i <= MAX_HISTORY_CHARS)
            .last()
            .unwrap_or(0)];
        format!(
            "{truncated}\n... <truncated, {} more chars>",
            history_text.len() - truncated.len()
        )
    } else {
        history_text
    };

    let compact_messages = vec![
        Message::system(
            "You are a conversation summarizer. Produce a concise summary of the key facts, \
             decisions, and pending tasks from the conversation below. Preserve file paths, \
             function names, error messages, and any actionable context. Output only the summary, \
             no preamble.",
        ),
        Message::user(format!("Summarize this conversation:\n\n{history_text}")),
    ];

    // Use non-streaming request without tools — summarization doesn't need them.
    let req = ChatRequest {
        model: cfg.model.clone(),
        messages: compact_messages,
        tools: vec![],
        max_tokens: 4096,
    };
    let resp = client.complete(req).await?;

    let summary = resp
        .message
        .content
        .as_deref()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| anyhow!("model produced no summary"))?;

    // Replace messages: keep system prompt + add summary as system context.
    let system = messages
        .iter()
        .find(|m| matches!(m.role, Role::System))
        .cloned()
        .unwrap_or_else(|| Message::system(system_prompt()));
    messages.clear();
    messages.push(system);
    messages.push(Message::system(format!(
        "[Conversation summary — previous {non_system} messages compacted]\n\n{summary}"
    )));

    Ok(non_system)
}

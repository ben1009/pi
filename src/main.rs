mod config;
mod llm;

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::config::ResolvedConfig;
use crate::llm::openai_compat::OpenAiCompatClient;
use crate::llm::{ChatRequest, ChatResponse, LlmClient, Message};

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

    let cfg = match config::resolve(
        cli.provider.as_deref(),
        cli.model.as_deref(),
        cli.max_tokens,
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("pi: {e}");
            // Only missing-API-key errors should map to exit 2; other config
            // errors (unknown provider, etc.) are usage problems → exit 1.
            let code = if e.to_string().starts_with("missing API key") {
                EXIT_MISSING_KEY
            } else {
                EXIT_API_OR_TURNS
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

// Lightweight UTC date (YYYY-MM-DD) without pulling in chrono.
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
    // Algorithm by Howard Hinnant: civil_from_days. Days since 1970-01-01.
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

async fn run(cfg: ResolvedConfig, one_shot: Option<String>) -> i32 {
    let client = OpenAiCompatClient::new(cfg.base_url.clone(), cfg.api_key.clone());
    let mut messages = vec![Message::system(system_prompt())];

    eprintln!(
        "pi: provider={} model={} max_tokens={}",
        cfg.provider.name(),
        cfg.model,
        cfg.max_tokens
    );

    if let Some(prompt) = one_shot {
        messages.push(Message::user(prompt));
        match send_full(&client, &cfg, &messages).await {
            Ok(resp) => {
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
            Err(e) => {
                eprintln!("pi: {e}");
                EXIT_API_OR_TURNS
            }
        }
    } else {
        repl(&client, &cfg, messages).await
    }
}

async fn repl(client: &OpenAiCompatClient, cfg: &ResolvedConfig, mut messages: Vec<Message>) -> i32 {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();

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
                eprintln!("pi: cleared.");
                continue;
            }
            _ => {}
        }

        messages.push(Message::user(line));
        match send(client, cfg, &messages).await {
            Ok(reply) => {
                if let Some(text) = reply.content.as_deref() {
                    println!("{text}");
                }
                messages.push(reply);
            }
            Err(e) => {
                eprintln!("pi: {e}");
                messages.pop(); // drop failed turn so retry doesn't duplicate
            }
        }
    }
}

async fn send(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    messages: &[Message],
) -> Result<Message> {
    Ok(send_full(client, cfg, messages).await?.message)
}

async fn send_full(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    messages: &[Message],
) -> Result<ChatResponse> {
    let req = ChatRequest {
        model: cfg.model.clone(),
        messages: messages.to_vec(),
        tools: Vec::new(),
        max_tokens: cfg.max_tokens,
    };
    client.complete(req).await
}

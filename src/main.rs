mod config;
mod llm;

use std::io::Write;

use anyhow::Result;
use clap::Parser;
use tokio::io::{AsyncBufReadExt, BufReader};

use crate::config::ResolvedConfig;
use crate::llm::openai_compat::OpenAiCompatClient;
use crate::llm::{ChatRequest, LlmClient, Message};

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
            std::process::exit(EXIT_MISSING_KEY);
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
    format!(
        "You are pi, a CLI coding agent. You help the user edit and run code in their working directory.\n\n\
         Working directory: {cwd}\n\
         Operating system: {os}\n\n\
         Prefer using the provided tools (bash, read, write, edit) over guessing. \
         When a tool returns an error, read the error and try a different approach. \
         Be concise."
    )
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
        match send(&client, &cfg, &messages).await {
            Ok(reply) => {
                if let Some(text) = reply.content.as_deref() {
                    println!("{text}");
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
                // RFC §6.2: keep user message in history; user can edit and retry.
            }
        }
    }
}

async fn send(
    client: &OpenAiCompatClient,
    cfg: &ResolvedConfig,
    messages: &[Message],
) -> Result<Message> {
    let req = ChatRequest {
        model: cfg.model.clone(),
        messages: messages.to_vec(),
        tools: Vec::new(),
        max_tokens: cfg.max_tokens,
    };
    let resp = client.complete(req).await?;
    Ok(resp.message)
}

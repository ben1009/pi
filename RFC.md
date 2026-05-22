# RFC: `pi` — a multi-LLM coding agent in Rust

Status: draft
Author: liu
Date: 2026-05-22

## 1. Goals

- Single-binary CLI that runs an interactive loop with an LLM and can edit code on disk.
- Small, readable surface. Target ~1.5k LOC for v0.
- Mirror the Claude Code tool-use loop closely enough that prompts and behaviors transfer.
- Support five providers out of the box: **OpenAI, Anthropic, Google Gemini, DeepSeek, Kimi**.

## 2. Non-goals (v0)

- TUI, streaming UI, syntax highlighting.
- MCP, sub-agents, hooks, slash commands beyond `/clear`, skills.
- Sandboxing/permissioning beyond a confirm-y/n on bash + writes outside CWD.
- Conversation persistence, compaction, memory.

## 3. Wire format: OpenAI-compatible Chat Completions

All five providers expose an OpenAI-compatible `/v1/chat/completions` endpoint with tool-use support. We speak that protocol exclusively in v0.

| Provider  | Base URL                                            | Auth header        | Default model               |
|-----------|-----------------------------------------------------|--------------------|-----------------------------|
| OpenAI    | `https://api.openai.com/v1`                         | `Authorization: Bearer` | `gpt-5`                |
| Anthropic | `https://api.anthropic.com/v1`                      | `Authorization: Bearer` + `x-api-key` | `claude-sonnet-4-6` |
| Gemini    | `https://generativelanguage.googleapis.com/v1beta/openai` | `Authorization: Bearer` | `gemini-2.5-pro` |
| DeepSeek  | `https://api.deepseek.com/v1`                       | `Authorization: Bearer` | `deepseek-chat`         |
| Kimi      | `https://api.moonshot.cn/v1`                        | `Authorization: Bearer` | `kimi-k2-0905-preview`  |

Tradeoff: we lose vendor-specific features (Anthropic prompt-caching headers, Gemini grounding, OpenAI strict tools, extended thinking blocks). Acceptable for v0 — a `LlmClient` trait keeps the door open for native adapters later.

## 4. Architecture

```
src/
  main.rs              # CLI entry, arg parsing
  agent.rs             # message loop, tool dispatch, REPL
  llm/
    mod.rs             # LlmClient trait, ChatRequest/ChatResponse, ToolCall
    openai_compat.rs   # the only impl in v0
  tools/
    mod.rs             # Tool trait, registry, JSON schema helpers
    bash.rs
    read.rs
    write.rs
    edit.rs
  config.rs            # provider registry, precedence
  errors.rs
```

`LlmClient` trait:

```rust
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse>;
}
```

`ChatRequest` / `ChatResponse` use the OpenAI shape (`role`, `content`, `tool_calls`, `tool_call_id`). It is the most expressive common denominator and round-trips cleanly across all five providers.

## 5. Configuration

Precedence: CLI flag → env var → `~/.config/pi/config.toml` → built-in defaults.

```toml
# ~/.config/pi/config.toml
default_provider = "anthropic"

[providers.openai]
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5"

[providers.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
model = "claude-sonnet-4-6"

[providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"
model = "gemini-2.5-pro"

[providers.deepseek]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"
model = "deepseek-chat"

[providers.kimi]
base_url = "https://api.moonshot.cn/v1"
api_key_env = "MOONSHOT_API_KEY"
model = "kimi-k2-0905-preview"
```

Env: `PI_PROVIDER`, `PI_MODEL`. CLI: `pi -P openai -m gpt-5`.

Built-in defaults match the table above so a user can do `OPENAI_API_KEY=sk-... pi -P openai` with no config file.

## 6. Agent loop

```
loop {
    line = read_user_input()
    push { role: user, content: line }
    loop {
        resp = client.complete(messages, tools)
        push resp.message
        if resp.message.tool_calls.is_empty() { print(resp.message.content); break }
        for call in resp.message.tool_calls {
            result = dispatch_tool(call)   // String, may be error
            push { role: tool, tool_call_id: call.id, content: result }
        }
    }
}
```

Stop conditions:
- `finish_reason = stop` and no tool calls → return to outer loop, await next user line.
- Ctrl-C → exit cleanly.
- API non-2xx → print body, drop the unsent user turn, keep loop alive.
- Tool error → returned as tool message with the error text; model retries or gives up.

## 7. Tools (v0)

| name    | input                                            | behavior                                                                |
|---------|--------------------------------------------------|-------------------------------------------------------------------------|
| `bash`  | `{ command: string, timeout_ms?: number }`       | `/bin/sh -c`, captured stdout+stderr, 120s default, 600s max.           |
| `read`  | `{ path: string, offset?: number, limit?: number }` | UTF-8, `cat -n` style line numbers, 2000-line default window.        |
| `write` | `{ path: string, content: string }`              | Overwrite/create. Parent dir must exist. Confirm if path outside CWD.   |
| `edit`  | `{ path, old_string, new_string, replace_all? }` | Exact-string replace; error if `old_string` not unique and not `replace_all`. |

Tool trait:

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> serde_json::Value;
    async fn run(&self, input: serde_json::Value) -> Result<String>;
}
```

Registry: `HashMap<&'static str, Box<dyn Tool>>` built once at startup.

Tool result cap: 25,000 chars per result; excess replaced with `\n... <truncated, N more chars>`.

Confirmation: `bash`, plus `write`/`edit` to paths outside CWD, prompt y/n unless `--yolo`.

## 8. CLI

```
pi                                  # interactive REPL in CWD
pi -p "fix the failing test"        # one-shot prompt, prints final assistant text, exits 0/1
pi -P anthropic -m claude-opus-4-7  # provider/model override
pi --yolo                           # skip confirmations
pi --max-turns 30                   # cap tool-loop iterations per user turn
```

In REPL: `/clear` resets messages, `/exit` quits, `Ctrl-D` quits.

## 9. Errors

- API non-2xx → `eprintln!` body, keep loop alive.
- Tool dispatch error (bad JSON, unknown tool) → tool message with `Error: ...`.
- Panics → `color-eyre` backtrace, exit 1.
- No API key for selected provider → exit 2 with a clear "set $X" message.

## 10. Dependencies

Pinned, minimal:

- `tokio` (rt-multi-thread, macros, process, fs)
- `reqwest` (json, rustls-tls)
- `serde`, `serde_json`
- `clap` (derive)
- `anyhow`
- `async-trait`
- `rustyline` (readline + history)
- `toml`
- `dirs` (config path resolution)

## 11. Compatibility caveats (documented, not papered over)

- **Reasoning models** (o-series, gemini-2.5-thinking, claude extended thinking) emit reasoning content differently. v0 ignores reasoning fields; visible answer still works.
- **Strict tool schemas.** OpenAI strict mode requires `additionalProperties: false`; some compat servers reject it. Default to non-strict.
- **Tool-call IDs.** We round-trip whatever the server sends.
- **System messages.** All five providers accept the OpenAI `system` role on their compat endpoints. If a future provider chokes, fold it into the first user turn.
- **Anthropic compat endpoint** accepts both `Authorization: Bearer` and `x-api-key`; send both — harmless for the others.

## 12. Decisions (locked)

1. Default provider: **Anthropic, `claude-sonnet-4-6`**.
2. Streaming: **off in v0**. Visible latency on long replies; revisit in v1.
3. Bash sandboxing: **confirm-prompt only**. Not safe for untrusted prompts. Documented.
4. Tool result cap: **25,000 chars**.
5. Quirks: **fail loudly** if a provider rejects something; do not silently rewrite requests.

## 13. Milestones

- **M1** — scaffold + `LlmClient` trait + `openai_compat` impl + non-tool chat working against all five providers.
- **M2** — tool trait + four tools + tool-use loop.
- **M3** — REPL polish (`rustyline`, history file, `/clear`, `/exit`), `--yolo`, `--max-turns`, README with one-line install + per-provider snippets.

## 14. Out-of-scope, parked for v1+

- Streaming responses.
- Native Anthropic adapter (prompt caching, extended thinking).
- Native Gemini adapter (grounding, file API).
- Conversation persistence and `/resume`.
- Sub-agents and parallel tool calls beyond what the model emits in one response.
- MCP client.
- A real permission model.

# RFC: `pi-rs` — a multi-LLM coding agent in Rust

Status: **living document** (updated 2026-05-23 to reflect current implementation)
Author: liu
Date: 2026-05-22

## 1. Goals

- Single-binary CLI that runs an interactive loop with an LLM and can edit code on disk.
- Small, readable surface. Target ~1.5k LOC for v0.
- Mirror the Claude Code tool-use loop closely enough that prompts and behaviors transfer.
- Support five providers out of the box: **OpenAI, Anthropic, Google Gemini, DeepSeek, Kimi**.

## 2. Non-goals (v0)

- TUI, streaming UI, syntax highlighting.
- Sub-agents, hooks, skills.
- Slash commands beyond `/clear`, `/exit`, `/tokens`, `/save`, `/compact`.
- Sandboxing/permissioning beyond a confirm-y/n on bash + writes outside CWD.
- Binary file handling (base64 read/write).
- Native provider adapters (Anthropic Messages API, Gemini native).

> **Note:** Several items originally listed here (MCP, conversation persistence, context accounting, SSE streaming) have since been implemented and are documented in §14.

## 3. Wire format: OpenAI-compatible Chat Completions

All five providers expose an OpenAI-compatible `/v1/chat/completions` endpoint with tool-use support. We speak that protocol in the default code path.

| Provider  | Base URL (no trailing slash)                                  | Auth                       |
|-----------|---------------------------------------------------------------|----------------------------|
| OpenAI    | `https://api.openai.com/v1`                                   | `Authorization: Bearer …`  |
| Anthropic | `https://api.anthropic.com/v1`                                | `Authorization: Bearer …`  |
| Gemini    | `https://generativelanguage.googleapis.com/v1beta/openai`     | `Authorization: Bearer …`  |
| DeepSeek  | `https://api.deepseek.com/v1`                                 | `Authorization: Bearer …`  |
| Kimi      | `https://api.moonshot.ai/v1`                                  | `Authorization: Bearer …`  |

Notes:
- Base URLs **must not** have a trailing slash; the client appends `/chat/completions`.
- Anthropic's compat endpoint accepts `Authorization: Bearer` only; the `x-api-key` header is for the native Messages API and is **not** sent.
- Kimi default is `api.moonshot.ai` (international). The CN endpoint `api.moonshot.cn` is configurable.

Default model IDs live in a single `const` table in `config.rs`; the README links to provider model-list docs. Model names rot fast.

Tradeoff: we lose vendor-specific features (Anthropic prompt-caching headers, Gemini grounding, OpenAI strict tools, extended thinking blocks). A `LlmClient` trait keeps the door open for native adapters later.

## 4. Architecture

Current source tree (as of 2026-05-23):

```
src/
  main.rs              # CLI entry, arg parsing, agent loop
  lib.rs               # Library exports
  config.rs            # provider registry, default model table, precedence
  confirm.rs           # y/n confirmation prompts (--yolo bypass)
  context.rs           # per-model context-window tracking and compaction
  session.rs           # conversation persistence, --resume, --sessions, /save
  llm/
    mod.rs             # LlmClient trait, ChatRequest/ChatResponse, ToolCall
    openai_compat.rs   # OpenAI-compatible client (default)
  tools/
    mod.rs             # Tool trait, registry, JSON schema helpers
    bash.rs            # bash execution with streaming output
    read_file.rs       # read UTF-8 files with line numbers (tool name: `read`)
    write_file.rs      # write files (tool name: `write`)
    edit.rs            # exact string replacement
  mcp.rs               # MCP client over stdio JSON-RPC 2.0
```

The original RFC planned `agent.rs` and `errors.rs`; the agent loop lives in `main.rs` and errors are handled via `anyhow` + `color-eyre` throughout.

`LlmClient` trait:

```rust
#[async_trait]
pub(crate) trait LlmClient: Send + Sync {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse>;
    async fn complete_stream(&self, req: ChatRequest) -> Result<EventStream>;
}
```

`ChatRequest` / `ChatResponse` use the OpenAI shape (`role`, `content`, `tool_calls`, `tool_call_id`) and are crate-private.

## 5. Configuration

Precedence: CLI flag → env var → built-in defaults.

```toml
# Configuration via environment variables:
# PI_PROVIDER, PI_MODEL, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.

[providers.anthropic]
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"

[providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
api_key_env = "GEMINI_API_KEY"

[providers.deepseek]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"

[providers.kimi]
base_url = "https://api.moonshot.ai/v1"
api_key_env = "MOONSHOT_API_KEY"
```

Env: `PI_PROVIDER`, `PI_MODEL`, `PI_MAX_TOKENS`. CLI: `pi-rs -P openai -m gpt-5 --max-tokens 8192`.

Built-in defaults match the table in §3 so a user can do `OPENAI_API_KEY=sk-… pi-rs -P openai` with no config file.

## 6. Agent loop

```
loop {
    line = read_user_input()
    push { role: user, content: line }
    loop {
        resp = client.complete(messages, tools, max_tokens)
        push resp.message
        if resp.message.tool_calls.is_empty() { print(resp.message.content); break }
        for call in resp.message.tool_calls {        // sequential, in emission order
            result = dispatch_tool(call)             // String, may carry an error
            push { role: tool, tool_call_id: call.id, content: result }
        }
        if turns >= max_turns { warn; break }
    }
}
```

- **Sequential tool execution.** Even when the model emits multiple `tool_calls` in one turn, they run in order. Predictable, matches Claude Code behavior, avoids races on the filesystem.
- **`max_tokens`.** Required by Anthropic compat, optional elsewhere. Default `8192`, override via `--max-tokens` / `PI_MAX_TOKENS`.
- **Reasoning models.** Reasoning content is ignored for display but **still consumes output tokens** and counts against `max_tokens`. Users on o-series / extended-thinking models may need a higher cap.
- **Context warnings.** Per-model context tracking is implemented in `context.rs`; the agent warns when approaching limits and supports `/compact` to summarize history.

### 6.1 System prompt (v0)

Sent on every request, not stored in user-visible history; preserved across `/clear`:

```
You are pi-rs, a CLI coding agent. You help the user edit and run code in their working directory.

Working directory: {cwd}
Operating system: {os}
Date: {date}

Prefer using the provided tools (bash, read, write, edit) over guessing. When a tool returns an error, read the error and try a different approach. Be concise.
```

`--print-system-prompt` dumps the rendered prompt and exits 0; useful for tests and debugging.

### 6.2 Stop and cancellation

- `finish_reason = stop` and no tool calls → return to outer loop.
- Ctrl-C while waiting for input (`ReadlineError::Interrupted`): discard input, return to prompt. (Note: Ctrl-C during in-flight API calls or tool runs is not yet implemented — see §14.)
- Ctrl-D / `/exit` → exit 0.
- API non-2xx → print body to stderr, **remove the failed user turn from history** and **restore the user's last input into the readline buffer** so they can edit and retry without retyping. Loop stays alive.
- Unknown tool name from the model → tool message `Error: unknown tool '<name>'`; the model recovers.

## 7. Tools (v0)

| name        | input                                            | behavior                                                                |
|-------------|--------------------------------------------------|-------------------------------------------------------------------------|
| `bash`      | `{ command: string, timeout_ms?: number }`       | `bash -c` (non-interactive, non-login), **stdout+stderr merged** in source order, 120s default, 600s max. Streaming output supported. |
| `read`      | `{ path: string, offset?: number, limit?: number }` | UTF-8 only; `cat -n` style line numbers; 2000-line default window.   |
| `write`     | `{ path: string, content: string }`              | Overwrite/create; **auto-creates parent directories** (`mkdir -p` semantics); **content written as-is**, no newline coercion. Confirm if path outside CWD. |
| `edit`      | `{ path, old_string, new_string, replace_all? }` | **Exact byte match** (no whitespace normalization). Error if `old_string` not unique and not `replace_all`. On miss, return up to 3 nearest line-number candidates. |

Path resolution: relative paths resolve against the **process CWD at startup**; `bash` inherits the same CWD. The agent does not `cd` between calls.

CWD boundary checks (for the "outside CWD → confirm" rule on `write`/`edit`) operate on the **canonicalized path** — i.e. symlinks are resolved before the prefix comparison so a symlink pointing outside CWD cannot bypass the prompt.

Encoding & errors:
- `read` on a non-UTF-8 file → tool message `Error: <path> is not valid UTF-8` (no base64 fallback in v0).
- `write` on a path whose parent dir doesn't exist → parent dirs are created automatically.
- `edit` miss → tool error including nearest matching lines so the model can self-correct without a re-`read`.

Tool trait:

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> serde_json::Value;
    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String>;
}
```

Tools return **plain UTF-8 strings**, which the agent puts into the OpenAI `tool` message `content` field as-is. No JSON wrapping.

Result cap: **100,000 chars** per tool result by default, override with `--max-tool-output`. Excess replaced with `\n... <truncated, N more chars>`. Large enough for typical source files and build logs without ballooning context cost.

Confirmation: `bash`, plus `read`/`write`/`edit` to paths outside CWD, prompt y/n unless `--yolo`.

## 8. CLI

```
pi-rs                               # interactive REPL in CWD
pi-rs -p "fix the failing test"     # one-shot prompt
pi-rs -P anthropic -m claude-opus-4-7  # provider/model override
pi-rs --max-tokens 16384
pi-rs --max-turns 30
pi-rs --max-tool-output 200000      # bump per-tool-result cap
pi-rs --yolo                        # skip confirmations
pi-rs --print-system-prompt         # dump rendered prompt and exit 0
pi-rs --resume <SESSION_ID>         # resume a specific session
pi-rs --sessions                    # list saved sessions
```

REPL slash commands: `/clear` (reset history; system prompt preserved), `/exit`, `/tokens` (print last response's `usage` block), `/save` (persist current session), `/compact` (summarize history to free context).

Exit codes for `-p` one-shot mode:
- `0` — model finished (`finish_reason = stop`) and produced text.
- `1` — API error, tool fatal error, or `--max-turns` exhausted.
- `2` — missing API key for the selected provider.

## 9. Errors

- API non-2xx → `eprintln!` body, keep loop alive.
- Tool dispatch (bad JSON, unknown tool, tool runtime failure) → tool message with `Error: …`.
- Panics → `color-eyre`-style backtrace, exit 1.
- Missing API key for selected provider → exit 2 with a clear "set $X" message.

## 10. Dependencies

Pinned, minimal:

- `tokio` (rt-multi-thread, macros, process, fs, time)
- `reqwest` (json, rustls-tls, stream — for SSE)
- `serde`, `serde_json`
- `clap` (derive)
- `anyhow`
- `async-trait`
- `rustyline` (readline + history)
- `toml`
- `dirs` (config path resolution)
- `color-eyre` (panic / error reporting)

Test-only:
- `wiremock` (mocked `/v1/chat/completions` server for integration tests; no real-API calls in CI)

## 11. Compatibility caveats

- **Reasoning models** (o-series, gemini-2.5-thinking, claude extended thinking) emit reasoning content differently and consume extra output tokens. The agent ignores reasoning fields for display.
- **Strict tool schemas.** OpenAI strict mode requires `additionalProperties: false`; some compat servers reject it. Default to non-strict.
- **Tool-call IDs.** We round-trip whatever the server sends.
- **System messages.** All five providers accept the OpenAI `system` role on their compat endpoints.

## 12. Decisions (locked)

1. Default provider: **Anthropic** (model from `config.rs` const table).
2. Streaming: **SSE streaming implemented** (was off in v0, enabled post-v0).
3. Bash sandboxing: **confirm-prompt only**. Not safe for untrusted prompts. Documented.
4. Tool result cap: **100,000 chars** default; configurable via `--max-tool-output`.
5. Quirks: **fail loudly** if a provider rejects something; do not silently rewrite requests.
6. Tool execution: **sequential**, in the order the model emitted them.
7. `max_tokens` default: **8192**.
8. Tool result encoding: **plain UTF-8 strings**, no JSON wrapping.

## 13. Milestones

- **M1** — scaffold + `LlmClient` trait + `openai_compat` impl + non-tool chat working against all five providers. ✅
- **M2** — tool trait + four tools + tool-use loop + `--yolo` + `--max-turns`. ✅
- **M3** — REPL polish (`rustyline`, history file, `/clear`, `/exit`, `/tokens`), `--print-system-prompt`, `wiremock`-backed integration tests, README with one-line install + per-provider snippets. ✅
- **M4** (post-v0) — SSE streaming, session persistence, context accounting + `/compact`, MCP client integration. ✅

## 14. Out-of-scope, parked for v1+

- Native Anthropic adapter (prompt caching via `cache_control: ephemeral`, extended thinking).
- Native Gemini adapter (grounding, file API).
- Sub-agents and parallel tool execution.
- A real permission model (sandboxing, filesystem restrictions).
- Binary file handling (base64 read/write).
- Hooks / skills system.

> **Implemented since v0 (moved out of non-goals):**
> - ~~SSE streaming~~ — Implemented in M4.
> - ~~Conversation persistence and `/resume`~~ — Implemented in `session.rs`.
> - ~~MCP client~~ — Implemented in `mcp.rs`.
> - ~~Per-model context-window accounting and compaction~~ — Implemented in `context.rs`.

## 15. Follow-up issues to file before M1

- [x] Pick LICENSE (MIT or Apache-2.0) and add it to the repo.
- [x] Track the model-IDs `const` table in `config.rs` so it can be updated independently of the RFC.

## 16. Open items (next)

1. **Native Anthropic adapter** — The OpenAI-compat endpoint lacks `cache_control: ephemeral` prompt caching, which significantly reduces cost for multi-turn conversations with long system prompts. A native Messages API adapter would enable this.
2. **Sub-agents** — RFC 002 drafted. Potential use cases: parallel research, multi-file refactoring with separate context windows.

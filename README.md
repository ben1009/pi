# pi

[![Check](https://github.com/ben1009/pi-rs/actions/workflows/check.yml/badge.svg)](https://github.com/ben1009/pi-rs/actions/workflows/check.yml)
[![Test](https://github.com/ben1009/pi-rs/actions/workflows/test.yml/badge.svg)](https://github.com/ben1009/pi-rs/actions/workflows/test.yml)

A multi-LLM coding agent in Rust. Single binary, OpenAI-compatible Chat Completions wire format, four built-in tools (bash, read, write, edit).

## Install

```
cargo install --git https://github.com/ben1009/pi-rs
```

## Quickstart

```
ANTHROPIC_API_KEY=sk-... pi -p "fix the failing test"
```

With no `-p`, `pi` opens an interactive REPL in the current directory. Provider defaults to Anthropic.

## Providers

Pick a provider with `-P` (or `PI_PROVIDER`) and supply its API key via env var:

| Provider  | `-P` value  | Env var              | Default base URL                                              |
|-----------|-------------|----------------------|---------------------------------------------------------------|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY`  | `https://api.anthropic.com/v1`                                |
| OpenAI    | `openai`    | `OPENAI_API_KEY`     | `https://api.openai.com/v1`                                   |
| Gemini    | `gemini`    | `GEMINI_API_KEY`     | `https://generativelanguage.googleapis.com/v1beta/openai`     |
| DeepSeek  | `deepseek`  | `DEEPSEEK_API_KEY`   | `https://api.deepseek.com/v1`                                 |
| Kimi      | `kimi`      | `MOONSHOT_API_KEY`   | `https://api.moonshot.ai/v1`                                  |

Examples:

```
OPENAI_API_KEY=sk-...     pi -P openai     -p "summarise main.rs"
GEMINI_API_KEY=...        pi -P gemini     -m gemini-2.5-pro
DEEPSEEK_API_KEY=sk-...   pi -P deepseek
MOONSHOT_API_KEY=sk-...   pi -P kimi
```

## CLI flags

| Flag                    | Env                  | Default   | Meaning                                                        |
|-------------------------|----------------------|-----------|----------------------------------------------------------------|
| `-p, --prompt <STR>`    |                      |           | One-shot: send prompt, print final text, exit.                 |
| `-P, --provider <NAME>` | `PI_PROVIDER`        | anthropic | Pick provider from the table above.                            |
| `-m, --model <ID>`      | `PI_MODEL`           | per-prov. | Override the model id.                                         |
| `--max-tokens <N>`      | `PI_MAX_TOKENS`      | 8192      | Output token cap per request.                                  |
| `--max-turns <N>`       | `PI_MAX_TURNS`       | 50        | Cap on tool-use iterations per user turn.                      |
| `--max-tool-output <N>` | `PI_MAX_TOOL_OUTPUT` | 100000    | Per-tool-result character cap; excess truncated.               |
| `-y, --yolo`            |                      | off       | Skip y/N prompts on bash and out-of-CWD writes/edits.          |
| `--print-system-prompt` |                      |           | Render the system prompt and exit 0.                           |

## Tools

- `bash` — runs `bash -c <command>`, stdout+stderr merged, 120s default timeout (max 600s). Confirms y/N unless `--yolo`.
- `read` — UTF-8 read with `cat -n` style line numbers; supports `offset` / `limit`.
- `write` — overwrite or create a file; auto-creates parent directories. Confirms if the path is outside CWD.
- `edit` — exact-byte string replacement; errors if `old_string` is not unique unless `replace_all` is set. Confirms if the path is outside CWD.

CWD checks canonicalize the target path before comparing, so symlinks pointing outside the working directory cannot bypass the prompt.

## REPL slash commands

| Command            | Effect                                              |
|--------------------|-----------------------------------------------------|
| `/clear`           | Reset conversation history; system prompt preserved. |
| `/exit`, `/quit`   | Exit the REPL with code 0.                          |
| `/tokens`          | Print the last response's `usage` block.            |

History is persisted to `$XDG_DATA_HOME/pi/history` (typically `~/.local/share/pi/history`).

## Exit codes

| Code | Meaning                                                          |
|------|------------------------------------------------------------------|
| 0    | Success (model finished and produced text, or REPL exited cleanly). |
| 1    | API error, tool fatal error, `--max-turns` exhausted, or unknown provider. |
| 2    | Missing API key for the selected provider.                       |

## Development

Prerequisites: Rust nightly (pinned via `rust-toolchain`), `cargo-make`.

```
./dev              # list available tasks
./dev check        # run all checks (fmt, clippy, machete, test, typos)
./dev test         # run tests only
./dev check-fmt    # format check only
./dev check-clippy # clippy only
```

The `dev` script auto-installs `cargo-binstall` and `cargo-make` on first run.

CI runs on every push/PR: fmt, clippy, typos, tests (ubuntu + macOS), coverage (codecov), sanitizers (ASan/leak), dependency review, and a nightly rolling build.

## Design

See [RFC.md](./RFC.md) for the full spec, tradeoffs, and v0 non-goals.

## License

Apache-2.0 — see [LICENSE](./LICENSE).

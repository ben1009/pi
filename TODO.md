# TODO

Pre-v1, in priority order.

## Now

- **Live smoke against the 5 providers.** Run a short tool-using prompt
  (e.g. "list files in this directory and read RFC.md") against each of
  OpenAI / Anthropic / Gemini / DeepSeek / Kimi with real keys. Confirm
  non-2xx error path, finish_reason variants, tool-call shape. Fix
  whatever falls over. The bots couldn't catch this; we haven't actually
  run `pi` end-to-end.
- **LICENSE.** RFC §15 promised one before code landed; we shipped four
  PRs without it. Pick MIT or Apache-2.0, drop a `LICENSE` file at repo
  root, mention in README and `Cargo.toml`.
- **CI.** `.github/workflows/ci.yml` running `cargo fmt --check`,
  `cargo clippy -- -D warnings`, `cargo test` on push and PR. No live-API
  jobs (wiremock covers the network surface). Add a status badge to
  README.

## Next

- **Subagents.** Parked while wrapping M2. Needs its own RFC pass before
  code: `task` tool shape (stateless one-shot vs model override vs
  parallel fan-out), `--max-subagent-turns`, cost/blast-radius story.
- **Streaming responses.** RFC §14, parked. Server-Sent Events on
  `/v1/chat/completions`; rustyline output without breaking the REPL
  prompt; tool calls arrive in deltas, need accumulation.
- **Conversation `/resume`.** Persist message history per session
  (`$XDG_DATA_HOME/pi/sessions/<id>.json`); `pi --resume <id>` reloads.
  Also enables a `/save` slash command.
- **Native Anthropic adapter.** Prompt caching headers
  (`cache_control: ephemeral` on system prompt + tool list) cut cost
  significantly across multi-turn sessions. Implement as a second
  `LlmClient` impl behind `pi -P anthropic-native`, leave OpenAI-compat
  Anthropic as the default fallback.

## Later

- **MCP client.** External tool servers via the Model Context Protocol.
  Big surface; plan separately.
- **Real permission model.** Confirm-prompt + `--yolo` is the v0 honest
  story but isn't a sandbox. Bubblewrap on Linux / Seatbelt on macOS,
  or shell out to firejail when present. Untrusted-prompt safety story
  for code agents in general is unsolved; this is exploratory.
- **Per-model context accounting + compaction.** Today the API surfaces
  context errors and we let `--max-turns` bound runaway loops. A
  per-model context table + a `/compact` slash command that summarizes
  early turns would help long sessions.
- **Binary file handling.** `read` errors on non-UTF-8; consider
  base64-mode for images / pdfs once the model surface gains
  multimodal-tool-result support.

## Quality follow-ups parked from review rounds

- **Benchmark `messages.to_vec()` clone-per-request.** Three reviewers
  flagged it across PRs #2 and #3; we declined as v0 micro-optimization.
  Profile a real long session to confirm or refute.
- **Streaming bash output.** Today we capture-then-return; long-running
  commands look frozen. Stream stdout to the user as it arrives, then
  send the captured tail as the tool result. Keep the post-cap pipe
  drain and reader-task abort guarantees from the existing tool — they
  are load-bearing for crash safety.
- **rustyline buffer restore on API error.** Round-2 PR #2 reviewer
  suggestion: on retry, prefill the readline buffer with the last
  failed input via `readline_with_initial`. Currently the user retypes.
- **Edit nearest-line hint quality.** Today's heuristic anchors on the
  first non-blank line; it misses if the model paraphrases that anchor.
  Could use a real diff (e.g. `similar`) to score candidates.

## Out of scope, possibly forever

- TUI rewrite. The plain stdin/stdout REPL is a feature, not a placeholder.
- Multi-provider abstraction beyond `LlmClient`. Five providers via one
  wire format is enough.

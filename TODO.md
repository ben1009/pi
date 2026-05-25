# TODO

Pre-v1, in priority order.

## Now

- **Live smoke against the 5 providers.** Run a short tool-using prompt
  (e.g. "list files in this directory and read RFC.md") against each of
  OpenAI / Anthropic / Gemini / DeepSeek / Kimi with real keys. Confirm
  non-2xx error path, finish_reason variants, tool-call shape. Fix
  whatever falls over. The bots couldn't catch this; we haven't actually
  run `pi-rs` end-to-end.
- ~~**LICENSE.**~~ Done. Apache-2.0.
- ~~**CI.**~~ Done. Full pipeline: fmt, clippy, typos, tests (ubuntu + macOS), coverage (codecov), sanitizers, dependency review, nightly rolling.

## Next

- **Subagents.** RFC drafted (`docs/rfc/002-subagents.md`, PR #26).
  Ready for implementation: `task` tool shape, `--max-subagent-turns`,
  model arbitrage, tool isolation.

## Done

- ~~**Native Anthropic adapter.**~~ Done. PR #27. `AnthropicNativeClient`
  with prompt caching (`cache_control: ephemeral` on system + tools),
  SSE streaming, dynamic dispatch via `Box<dyn LlmClient>`.
- ~~**Streaming responses.**~~ Done. SSE streaming with content deltas and
  tool-call accumulation via `complete_stream` / `send_streaming`.
- ~~**Conversation `/resume`.**~~ Done. `pi-rs --resume <id>` reloads saved
  sessions; `pi-rs --sessions` lists them; `/save` slash command persists
  current session.
- ~~**MCP client.**~~ Done. External tool servers via stdio transport,
  with tool name sanitization, duplicate server detection, and collision
  guards.
- ~~**Per-model context accounting + compaction.**~~ Done. Per-provider
  context window sizes in `context.rs`; `/compact` slash command summarizes
  early turns.
- ~~**Streaming bash output.**~~ Done. Bash tool streams stdout to stderr
  in real-time when `stream_stderr` is enabled.

## Later

- **Real permission model.** Confirm-prompt + `--yolo` is the v0 honest
  story but isn't a sandbox. Bubblewrap on Linux / Seatbelt on macOS,
  or shell out to firejail when present. Untrusted-prompt safety story
  for code agents in general is unsolved; this is exploratory.
- **Binary file handling.** `read` errors on non-UTF-8; consider
  base64-mode for images / pdfs once the model surface gains
  multimodal-tool-result support.

## Quality follow-ups parked from review rounds

- **Benchmark `messages.to_vec()` clone-per-request.** Three reviewers
  flagged it across PRs #2 and #3; we declined as v0 micro-optimization.
  Profile a real long session to confirm or refute.
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

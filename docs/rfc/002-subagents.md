# RFC: Sub-agents for `pi-rs`

Status: **draft**
Author: liu
Date: 2026-05-24

## 1. Motivation

The current agent loop is single-threaded: one LLM, one context window, sequential tool calls. This limits several real workflows:

- **Parallel research.** "Find all usages of `foo` across the codebase and check if each caller handles errors" — today requires many sequential turns or a single massive prompt.
- **Multi-file refactoring.** A long context window fills up fast when touching 10+ files. Splitting into subtasks with independent context windows is cheaper and more reliable.
- **Model arbitrage.** Some tasks (grep, summarize) don't need Opus; routing them to Haiku/Sonnet cuts cost 5-10x.
- **Tool isolation.** A subagent running `cargo test` shouldn't have write access to production configs.

Sub-agents address all four by letting the main agent spawn lightweight, scoped workers.

## 2. `task` tool shape

```json
{
  "task": "Find all places that call `parse_config` and check if errors are handled",
  "model": "haiku",
  "tools": ["bash", "read"],
  "max_turns": 15,
  "max_tokens": 8192,
  "timeout_ms": 120000
}
```

| Field        | Required | Default              | Description |
|--------------|----------|----------------------|-------------|
| `task`       | yes      | —                    | Natural-language task description. Becomes the sub-agent's first user message. |
| `model`      | no       | parent's model       | Model alias (`haiku`, `sonnet`, `opus`) or full model ID. |
| `tools`      | no       | `["read"]`           | Tool names the sub-agent may use. Subset of parent's tools. Validated against registry. |
| `max_turns`  | no       | 10                   | Max agent-loop iterations (user→assistant→tool cycles). |
| `max_tokens` | no       | 8192                 | `max_tokens` passed to the LLM API per request. |
| `timeout_ms` | no       | 120000               | Wall-clock timeout for the entire sub-agent run. Prevents stuck `bash` from hanging the parent. |

Default tools are `["read"]` only. `bash` can write files and make network calls — it is **not read-only** and must be explicitly opted in. `edit` and `write` are also excluded by default.

## 3. Execution model

Stateless one-shot:

```
Parent: "Find all callers of parse_config"
  └─ Sub-agent gets fresh context: system prompt + task message
     └─ Runs up to max_turns (capped by timeout_ms)
        └─ Returns final assistant message as text to parent
```

- Sub-agent has **no** parent conversation history.
- Sub-agent's context is: its own system prompt (with tool schemas) + the `task` string + tool results it generates.
- Parent receives a single text result.
- Tool-call IDs are scoped to the sub-agent's own message vector — no collision with parent.

### Known limitation: write conflicts

If a sub-agent's tool whitelist includes `edit` or `write`, multiple sub-agents writing to the same file can corrupt it. v1 runs sequentially so this is safe, but parallel fan-out (v1.1) will need worktree isolation or file-level locking.

## 4. Cancellation and timeouts

```rust
pub async fn run(&self, cancel: CancellationToken) -> Result<String> {
    let result = tokio::time::timeout(
        Duration::from_millis(self.timeout_ms),
        self.run_inner(cancel),
    ).await;

    match result {
        Ok(r) => r,
        Err(_) => Ok("[error: sub-agent timed out]".into()),
    }
}

async fn run_inner(&self, cancel: CancellationToken) -> Result<String> {
    for turn in 0..self.max_turns {
        cancel.is_cancelled().then(|| anyhow::bail!("cancelled"));
        // ... agent loop
    }
}
```

- `CancellationToken` from the parent propagates to the sub-agent. On Ctrl-C, the parent drops the token; the sub-agent checks it each turn and aborts cleanly.
- `timeout_ms` wraps the entire `run()` — a stuck `bash` subprocess doesn't hang the parent forever.
- Tool execution (`bash`) should forward the cancellation token to kill child processes.

## 5. System prompt

```
You are a sub-agent of pi-rs. Your task is described below.
Complete it and return a concise text summary of your findings.

Working directory: {cwd}
Operating system: {os}
Date: {date}

Max turns: {max_turns}

Available tools (with schemas):
{tool_schemas_json}

Task: {task}
```

`{tool_schemas_json}` includes the full JSON schema for each allowed tool (from `Tool::schema()`), not just names. Without schemas, the LLM hallucinates parameter shapes and wastes turns on malformed calls.

The parent's system prompt is **not** included — the sub-agent doesn't need to know about the parent's capabilities or constraints.

## 6. Result handling

### What the parent receives

The sub-agent's last **substantive** assistant message is returned as the `task` tool's result. A "substantive" message is one with non-empty text and no pending tool calls.

On `max_turns` exhaustion: return the last substantive message + `[warning: sub-agent hit max_turns limit]`. If no substantive message exists (e.g., the sub-agent was looping on tool calls), return `[warning: sub-agent hit max_turns with no summary produced]`.

On `timeout_ms` exhaustion: return `[error: sub-agent timed out]`.

### Error handling

| Scenario                          | Behavior |
|-----------------------------------|----------|
| Sub-agent finishes normally       | Return last substantive assistant message. |
| Sub-agent hits `max_turns`        | Return last substantive message + warning. |
| Sub-agent hits `timeout_ms`       | Return timeout error. |
| Sub-agent API error               | Return `Error: sub-agent failed: <reason>`. |
| Sub-agent tool transport error    | Abort early, return `Error: sub-agent tool failure: <reason>`. |
| Sub-agent tool user-visible error | Sub-agent sees the error and may recover (the LLM retries). |
| Parent cancels (Ctrl-C)           | Cancel via `CancellationToken`, return `Error: cancelled`. |
| Unknown tool name in whitelist    | Return `Error: unknown tool '<name>'` before spawning. |

### Progress observability

While the sub-agent runs, emit a progress line to stderr on each turn:

```
[sub-agent] turn 3/10: running bash
[sub-agent] turn 4/10: running read
[sub-agent] done (7 turns, ~12K tokens)
```

This is a lightweight `on_turn` callback, not a full message channel. Prevents the user from thinking the process is hung during long runs.

## 7. Implementation sketch

```
src/
  subagent.rs          # SubAgent struct, spawn logic, result handling
  tools/
    task.rs            # TaskTool implementing the Tool trait
```

### `SubAgent` struct

```rust
pub(crate) struct SubAgent {
    task: String,
    model: String,
    tools: Vec<String>,
    max_turns: usize,
    max_tokens: usize,
    timeout_ms: u64,
    client: Arc<dyn LlmClient>,
    tool_registry: Arc<ToolRegistry>,
}

impl SubAgent {
    pub async fn run(&self, cancel: CancellationToken) -> Result<String> {
        let result = tokio::time::timeout(
            Duration::from_millis(self.timeout_ms),
            self.run_inner(cancel),
        ).await;
        match result {
            Ok(r) => r,
            Err(_) => Ok("[error: sub-agent timed out]".into()),
        }
    }

    async fn run_inner(&self, cancel: CancellationToken) -> Result<String> {
        let mut messages = vec![self.system_prompt(), user_message(&self.task)];
        let tools = self.filtered_tools();

        for turn in 0..self.max_turns {
            if cancel.is_cancelled() {
                return Ok("[error: cancelled]".into());
            }
            eprintln!("[sub-agent] turn {}/{}: awaiting response", turn + 1, self.max_turns);

            let req = ChatRequest {
                model: self.model.clone(),
                messages: messages.clone(),
                tools: tools.clone(),
                max_tokens: self.max_tokens,
            };
            let resp = self.client.complete(req).await?;
            messages.push(resp.message.clone());

            if resp.message.tool_calls.is_empty() {
                return Ok(resp.message.content.unwrap_or_default());
            }

            for call in &resp.message.tool_calls {
                let tool = match self.tool_registry.get(&call.name) {
                    Some(t) => t,
                    None => {
                        let msg = format!("Error: unknown tool '{}'", call.name);
                        messages.push(tool_result(call.id.clone(), msg));
                        continue;
                    }
                };
                let result = match tool.run(call.arguments.clone()).await {
                    Ok(r) => r,
                    Err(e) => {
                        // Transport-level error — abort, don't loop
                        return Ok(format!("[error: sub-agent tool failure: {}]", e));
                    }
                };
                eprintln!("[sub-agent] turn {}/{}: running {}", turn + 1, self.max_turns, call.name);
                messages.push(tool_result(call.id.clone(), result));
            }
        }

        // Return last substantive message (non-empty text, no tool calls)
        Ok(format!(
            "{}\n[warning: sub-agent hit max_turns limit]",
            last_substantive_content(&messages)
        ))
    }
}
```

### `TaskTool`

```rust
pub(crate) struct TaskTool {
    client: Arc<dyn LlmClient>,
    registry: Arc<ToolRegistry>,
    default_model: String,
    cancel: CancellationToken,
}

#[async_trait]
impl Tool for TaskTool {
    fn name(&self) -> &'static str { "task" }

    fn schema(&self) -> Value { /* tool schema */ }

    async fn run(&self, input: Value) -> Result<String> {
        // Validate tool names against registry before spawning
        let tools: Vec<String> = input["tools"].as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str()).map(|s| s.to_string()).collect())
            .unwrap_or_else(|| vec!["read".into()]);

        for name in &tools {
            if !self.registry.contains(name) {
                return Ok(format!("Error: unknown tool '{}'", name));
            }
        }

        let task = match input["task"].as_str() {
            Some(t) => t.to_string(),
            None => return Ok("Error: missing required field 'task'".into()),
        };

        let sub = SubAgent {
            task,
            model: input["model"].as_str()
                .unwrap_or(&self.default_model).to_string(),
            tools,
            max_turns: input["max_turns"].as_u64().unwrap_or(10) as usize,
            max_tokens: input["max_tokens"].as_u64().unwrap_or(8192) as usize,
            timeout_ms: input["timeout_ms"].as_u64().unwrap_or(120_000),
            client: self.client.clone(),
            registry: self.registry.clone(),
        };
        sub.run(self.cancel.clone()).await
    }
}
```

### Registration

`TaskTool` needs `registry.clone()`, but it is inserted into that same registry. This requires two-phase init:

```rust
// Phase 1: create registry with core tools
let registry = Arc::new(ToolRegistry::new());
registry.register(bash_tool);
registry.register(read_tool);
registry.register(write_tool);
registry.register(edit_tool);

// Phase 2: create and register task tool (needs registry reference)
let task_tool = Arc::new(TaskTool::new(client.clone(), registry.clone(), default_model, cancel.clone()));
registry.register(task_tool);
```

## 8. CLI additions

```
pi-rs --max-subagent-tokens 500000    # session-level cap (v1.1)
pi-rs --max-parallel-subagents 3      # future: parallel fan-out (v1.1)
```

## 9. Milestones

- **M1** — `task` tool, stateless execution, sequential, default `["read"]`, `max_turns` + `model` + `max_tokens` + `timeout_ms` controls, `CancellationToken` propagation, progress stderr output. ~300 LOC.
- **M1.1** — Parallel fan-out (`--max-parallel-subagents`), `--max-subagent-tokens` safety net, worktree isolation for write-capable sub-agents.

## 10. Future considerations

These are deferred, not rejected. Each can be added as a non-breaking extension.

- **Context inheritance** (`inherit_context: true`) — pass a slice of parent history to the sub-agent. Useful for "continue this analysis" patterns. Blocked on: tool_call_id reference cleanup, token accounting.
- **Custom system prompt suffix** (`system_prompt_suffix`) — extend the sub-agent's prompt with domain-specific instructions.
- **Working directory override** (`cwd`) — run sub-agent in a different directory.
- **Parallel fan-out** — run multiple `task` calls concurrently via `tokio::join!`. Needs `--max-parallel-subagents` and worktree isolation for write-capable sub-agents.
- **Nested sub-agents** — sub-agent spawning its own sub-agents. Adds complexity; keep as non-goal until clear use case.
- **Sub-agent ↔ parent streaming** — full message-passing channel. Overkill for v1; stderr progress lines cover the UX need.

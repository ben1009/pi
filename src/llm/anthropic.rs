use std::collections::VecDeque;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use futures_util::{StreamExt, stream};
use serde::{Deserialize, Serialize};

use super::{
    ChatRequest, ChatResponse, EventStream, LlmClient, Message, Role, StreamEvent, ToolCall,
    ToolCallFunction, Usage,
};

const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const MAX_STREAM_BUF: usize = 10 * 1024 * 1024; // 10 MB
const MAX_CONTENT_BLOCKS: usize = 128;
const MAX_TOOL_ARGS: usize = 1024 * 1024; // 1 MB

pub struct AnthropicNativeClient {
    url: String,
    api_key: String,
    http: reqwest::Client,
}

impl AnthropicNativeClient {
    pub fn new(api_key: String) -> Self {
        Self::with_base_url(DEFAULT_BASE_URL, api_key)
    }

    pub fn with_base_url(base_url: &str, api_key: String) -> Self {
        let http = reqwest::Client::builder()
            .user_agent(concat!("pi/", env!("CARGO_PKG_VERSION")))
            .connect_timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("build reqwest client");
        let url = format!("{}/messages", base_url.trim_end_matches('/'));
        Self { url, api_key, http }
    }
}

// ── Anthropic wire types ──

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<SystemBlock>>,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
    stream: bool,
}

#[derive(Serialize)]
struct SystemBlock {
    #[serde(rename = "type")]
    kind: &'static str,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

#[derive(Serialize)]
struct CacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
}

#[derive(Serialize)]
struct WireMessage {
    role: &'static str,
    content: Content,
}

#[derive(Serialize)]
#[serde(untagged)]
enum Content {
    Str(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
enum ContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Serialize)]
struct WireTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<CacheControl>,
}

// ── Response types ──

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ResponseContentBlock>,
    stop_reason: Option<String>,
    usage: ResponseUsage,
}

#[derive(Deserialize)]
struct ResponseContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    input: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ResponseUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

// ── Streaming types ──

#[derive(Deserialize)]
struct StreamEventWire {
    #[serde(rename = "type")]
    kind: String,
    // message_start
    #[serde(default)]
    message: Option<StreamMessage>,
    // content_block_start
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    content_block: Option<StreamContentBlock>,
    // content_block_delta / message_delta (raw JSON because shapes differ)
    #[serde(default)]
    delta: Option<serde_json::Value>,
    // message_delta
    #[serde(default)]
    usage: Option<StreamUsage>,
}

#[derive(Deserialize)]
struct StreamMessage {
    #[serde(default)]
    usage: Option<ResponseUsage>,
}

#[derive(Deserialize)]
struct StreamContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

#[derive(Deserialize)]
struct StreamUsage {
    #[serde(default)]
    output_tokens: Option<u32>,
}

// ── Conversion helpers ──

fn extract_system(messages: &[Message]) -> Option<Vec<SystemBlock>> {
    let mut blocks = Vec::new();
    for msg in messages {
        if !matches!(msg.role, Role::System) {
            break;
        }
        if let Some(text) = &msg.content {
            blocks.push(SystemBlock {
                kind: "text",
                text: text.clone(),
                cache_control: None,
            });
        }
    }
    if blocks.is_empty() {
        return None;
    }
    // Put cache_control on the last system block for maximum coverage.
    if let Some(last) = blocks.last_mut() {
        last.cache_control = Some(CacheControl { kind: "ephemeral" });
    }
    Some(blocks)
}

fn convert_messages(messages: &[Message]) -> Vec<WireMessage> {
    let mut out = Vec::new();
    for msg in messages {
        match msg.role {
            Role::System => continue, // handled by extract_system
            Role::User => {
                let text = msg.content.clone().unwrap_or_default();
                out.push(WireMessage {
                    role: "user",
                    content: Content::Str(text),
                });
            }
            Role::Assistant => {
                let mut blocks = Vec::new();
                if let Some(text) = &msg.content
                    && !text.is_empty()
                {
                    blocks.push(ContentBlock::Text { text: text.clone() });
                }
                if let Some(calls) = &msg.tool_calls {
                    for tc in calls {
                        let input: serde_json::Value =
                            serde_json::from_str(&tc.function.arguments).unwrap_or_default();
                        blocks.push(ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            input,
                        });
                    }
                }
                if blocks.is_empty() {
                    continue;
                }
                out.push(WireMessage {
                    role: "assistant",
                    content: Content::Blocks(blocks),
                });
            }
            Role::Tool => {
                let tool_use_id = msg
                    .tool_call_id
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "unknown".to_owned());
                let text = msg.content.clone().unwrap_or_default();
                // Merge consecutive tool results for the same turn into one user message.
                if let Some(last) = out.last_mut()
                    && last.role == "user"
                    && matches!(&last.content, Content::Blocks(blocks) if blocks.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. })))
                    && let Content::Blocks(blocks) = &mut last.content
                {
                    blocks.push(ContentBlock::ToolResult {
                        tool_use_id,
                        content: text,
                        is_error: None,
                    });
                    continue;
                }
                out.push(WireMessage {
                    role: "user",
                    content: Content::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id,
                        content: text,
                        is_error: None,
                    }]),
                });
            }
        }
    }
    out
}

fn convert_tools(tools: &[super::ToolDef]) -> Vec<WireTool> {
    let mut out: Vec<WireTool> = tools
        .iter()
        .map(|t| WireTool {
            name: t.function.name.to_owned(),
            description: t.function.description.to_owned(),
            input_schema: t.function.parameters.clone(),
            cache_control: None,
        })
        .collect();
    // Put cache_control on the last tool for maximum cache coverage.
    if let Some(last) = out.last_mut() {
        last.cache_control = Some(CacheControl { kind: "ephemeral" });
    }
    out
}

fn stop_reason_to_openai(reason: &str) -> String {
    match reason {
        "end_turn" | "stop_sequence" => "stop".to_owned(),
        "tool_use" => "tool_use".to_owned(),
        "max_tokens" => "length".to_owned(),
        other => other.to_owned(),
    }
}

fn parse_response(resp: AnthropicResponse) -> Result<ChatResponse> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block.kind.as_str() {
            "text" => {
                if let Some(t) = &block.text {
                    text_parts.push(t.clone());
                }
            }
            "tool_use" => {
                let id = block.id.clone().unwrap_or_default();
                let name = block.name.clone().unwrap_or_default();
                let args = block
                    .input
                    .as_ref()
                    .map(|v| serde_json::to_string(v).unwrap_or_default())
                    .unwrap_or_default();
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".to_owned(),
                    function: ToolCallFunction {
                        name,
                        arguments: args,
                    },
                });
            }
            _ => {}
        }
    }

    let content = {
        let joined = text_parts.join("");
        if joined.is_empty() {
            None
        } else {
            Some(joined)
        }
    };

    let prompt_tokens = resp.usage.input_tokens;
    let completion_tokens = resp.usage.output_tokens;
    let cache_read = resp.usage.cache_read_input_tokens.unwrap_or(0);
    let cache_write = resp.usage.cache_creation_input_tokens.unwrap_or(0);
    // Include cached tokens in total so /tokens shows the full picture.
    let total = prompt_tokens + completion_tokens + cache_read + cache_write;

    Ok(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        },
        finish_reason: stop_reason_to_openai(resp.stop_reason.as_deref().unwrap_or("end_turn")),
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: total,
        }),
    })
}

// ── LlmClient impl ──

#[async_trait]
impl LlmClient for AnthropicNativeClient {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse> {
        let body = AnthropicRequest {
            model: &req.model,
            max_tokens: req.max_tokens,
            system: extract_system(&req.messages),
            messages: convert_messages(&req.messages),
            tools: convert_tools(&req.tools),
            stream: false,
        };

        let resp = self
            .http
            .post(&self.url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(anyhow!(
                "Anthropic API {}: {}",
                status.as_u16(),
                super::truncate(&text, 2000)
            ));
        }

        let parsed: AnthropicResponse = serde_json::from_str(&text).map_err(|e| {
            anyhow!(
                "decode Anthropic response: {e}\nbody: {}",
                super::truncate(&text, 2000)
            )
        })?;

        parse_response(parsed)
    }

    async fn complete_stream(&self, req: ChatRequest) -> Result<EventStream> {
        let body = AnthropicRequest {
            model: &req.model,
            max_tokens: req.max_tokens,
            system: extract_system(&req.messages),
            messages: convert_messages(&req.messages),
            tools: convert_tools(&req.tools),
            stream: true,
        };

        let resp = self
            .http
            .post(&self.url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            return Err(anyhow!(
                "Anthropic API {}: {}",
                status.as_u16(),
                super::truncate(&text, 2000)
            ));
        }

        let byte_stream = resp.bytes_stream();
        let event_stream = byte_stream
            .scan(StreamState::default(), |state, chunk| {
                let chunk = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        return std::future::ready(Some(vec![Err(anyhow!(
                            "stream read error: {e}"
                        ))]));
                    }
                };
                state.buf.extend(chunk.iter().copied());
                if state.buf.len() > MAX_STREAM_BUF {
                    state.buf.clear();
                    return std::future::ready(Some(vec![Err(anyhow!(
                        "stream buffer exceeded {MAX_STREAM_BUF} bytes"
                    ))]));
                }

                let mut out = Vec::new();
                while let Some(pos) = state.buf.iter().position(|&b| b == b'\n') {
                    let line_bytes: Vec<u8> = state.buf.drain(..=pos).collect();
                    let line = String::from_utf8_lossy(&line_bytes);
                    let line = line.trim_end_matches('\r').trim_end_matches('\n').trim();

                    if line.is_empty() {
                        if !state.data_lines.is_empty() {
                            let combined = state.data_lines.join("\n");
                            state.data_lines.clear();
                            if let Err(e) = parse_anthropic_sse(&combined, state, &mut out) {
                                out.push(Err(e));
                            }
                        }
                        continue;
                    }
                    if line.starts_with(':') {
                        continue; // SSE comment
                    }
                    if let Some(rest) = line.strip_prefix("data:") {
                        let value = rest.strip_prefix(' ').unwrap_or(rest);
                        state.data_lines.push(value.to_owned());
                    }
                }
                std::future::ready(Some(out))
            })
            .flat_map(stream::iter);

        Ok(Box::pin(event_stream))
    }
}

#[derive(Default)]
struct StreamState {
    buf: VecDeque<u8>,
    data_lines: Vec<String>,
    // Track content blocks by index for tool call assembly.
    content_blocks: Vec<ContentBlockState>,
    usage: Option<Usage>,
    stop_reason: String,
    cache_read: u32,
    cache_write: u32,
}

#[derive(Default)]
struct ContentBlockState {
    kind: String,
    id: String,
    name: String,
    args: String,
}

fn parse_anthropic_sse(
    data: &str,
    state: &mut StreamState,
    out: &mut Vec<Result<StreamEvent>>,
) -> Result<()> {
    let event: StreamEventWire = serde_json::from_str(data).map_err(|e| {
        anyhow!(
            "Anthropic SSE parse error: {e} data: {}",
            super::truncate(data, 500)
        )
    })?;

    match event.kind.as_str() {
        "message_start" => {
            if let Some(msg) = event.message
                && let Some(u) = msg.usage
            {
                let prompt_tokens = u.input_tokens;
                let completion_tokens = u.output_tokens;
                let cache_read = u.cache_read_input_tokens.unwrap_or(0);
                let cache_write = u.cache_creation_input_tokens.unwrap_or(0);
                state.cache_read = cache_read;
                state.cache_write = cache_write;
                state.usage = Some(Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens + cache_read + cache_write,
                });
            }
        }
        "content_block_start" => {
            if let (Some(index), Some(cb)) = (event.index, event.content_block) {
                if index >= MAX_CONTENT_BLOCKS {
                    return Err(anyhow!("content_block index {index} exceeds limit"));
                }
                while state.content_blocks.len() <= index {
                    state.content_blocks.push(ContentBlockState::default());
                }
                state.content_blocks[index].kind = cb.kind.clone();
                if cb.kind == "tool_use" {
                    state.content_blocks[index].id = cb.id.unwrap_or_default();
                    state.content_blocks[index].name = cb.name.unwrap_or_default();
                }
            }
        }
        "content_block_delta" => {
            if let (Some(index), Some(delta)) = (event.index, event.delta) {
                let delta_type = delta.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match delta_type {
                    "text_delta" => {
                        if let Some(text) = delta.get("text").and_then(|v| v.as_str()) {
                            out.push(Ok(StreamEvent::ContentDelta(text.to_owned())));
                        }
                    }
                    "input_json_delta" => {
                        let partial_json = delta
                            .get("partial_json")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_owned();
                        // Emit id/name only on the first delta for this block.
                        let is_first = index < state.content_blocks.len()
                            && state.content_blocks[index].args.is_empty();
                        let (id, name) = if is_first && index < state.content_blocks.len() {
                            let cb = &state.content_blocks[index];
                            (Some(cb.id.clone()), Some(cb.name.clone()))
                        } else {
                            (None, None)
                        };
                        if index < state.content_blocks.len() {
                            if state.content_blocks[index].args.len() + partial_json.len()
                                > MAX_TOOL_ARGS
                            {
                                return Err(anyhow!("tool args exceeded {MAX_TOOL_ARGS} bytes"));
                            }
                            state.content_blocks[index].args.push_str(&partial_json);
                        }
                        out.push(Ok(StreamEvent::ToolCallDelta {
                            index,
                            id,
                            function_name: name,
                            arguments_delta: Some(partial_json),
                        }));
                    }
                    "thinking_delta" => {
                        // Ignore thinking content for display.
                    }
                    _ => {}
                }
            }
        }
        "content_block_stop" => {
            // No action needed.
        }
        "message_delta" => {
            // Read stop_reason from delta if present.
            if let Some(delta) = &event.delta
                && let Some(reason) = delta.get("stop_reason").and_then(|v| v.as_str())
            {
                state.stop_reason = reason.to_owned();
            }
            // Update usage with final output_tokens, preserving cache tokens in total.
            if let Some(u) = event.usage
                && let Some(ref mut existing) = state.usage
            {
                existing.completion_tokens = u.output_tokens.unwrap_or(existing.completion_tokens);
                existing.total_tokens = existing.prompt_tokens
                    + existing.completion_tokens
                    + state.cache_read
                    + state.cache_write;
            }
            // Done event is emitted on message_stop.
        }
        "message_stop" => {
            let reason = if state.stop_reason.is_empty() {
                "stop".to_owned()
            } else {
                stop_reason_to_openai(&state.stop_reason)
            };
            out.push(Ok(StreamEvent::Done {
                finish_reason: reason,
                usage: state.usage.take(),
            }));
            state.stop_reason.clear();
        }
        _ => {}
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_system_from_messages() {
        let msgs = vec![Message::system("you are helpful"), Message::user("hi")];
        let sys = extract_system(&msgs).unwrap();
        assert_eq!(sys.len(), 1);
        assert_eq!(sys[0].text, "you are helpful");
        assert!(sys[0].cache_control.is_some());
    }

    #[test]
    fn extract_system_none_without_system() {
        let msgs = vec![Message::user("hi")];
        assert!(extract_system(&msgs).is_none());
    }

    #[test]
    fn convert_user_message() {
        let msgs = vec![Message::user("hello")];
        let wire = convert_messages(&msgs);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
    }

    #[test]
    fn convert_assistant_with_tool_calls() {
        let msgs = vec![Message {
            role: Role::Assistant,
            content: Some("thinking".into()),
            tool_calls: Some(vec![ToolCall {
                id: "tc1".into(),
                kind: "function".into(),
                function: ToolCallFunction {
                    name: "bash".into(),
                    arguments: r#"{"command":"ls"}"#.into(),
                },
            }]),
            tool_call_id: None,
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "assistant");
        match &wire[0].content {
            Content::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                assert!(matches!(&blocks[0], ContentBlock::Text { text } if text == "thinking"));
                assert!(
                    matches!(&blocks[1], ContentBlock::ToolUse { id, name, .. } if id == "tc1" && name == "bash")
                );
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn convert_tool_message() {
        let msgs = vec![Message {
            role: Role::Tool,
            content: Some("output".into()),
            tool_calls: None,
            tool_call_id: Some("tc1".into()),
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
        match &wire[0].content {
            Content::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        ..
                    } => {
                        assert_eq!(tool_use_id, "tc1");
                        assert_eq!(content, "output");
                    }
                    _ => panic!("expected tool_result"),
                }
            }
            _ => panic!("expected blocks"),
        }
    }

    #[test]
    fn convert_tools_adds_cache_control_on_last() {
        let tools = vec![
            super::super::ToolDef {
                kind: "function",
                function: super::super::ToolDefFunction {
                    name: "a",
                    description: "tool a",
                    parameters: serde_json::json!({"type": "object"}),
                },
            },
            super::super::ToolDef {
                kind: "function",
                function: super::super::ToolDefFunction {
                    name: "b",
                    description: "tool b",
                    parameters: serde_json::json!({"type": "object"}),
                },
            },
        ];
        let wire = convert_tools(&tools);
        assert_eq!(wire.len(), 2);
        assert!(wire[0].cache_control.is_none());
        assert!(wire[1].cache_control.is_some());
        assert_eq!(wire[0].name, "a");
        assert_eq!(wire[1].name, "b");
        assert_eq!(wire[0].input_schema, serde_json::json!({"type": "object"}));
    }

    #[test]
    fn stop_reason_mapping() {
        assert_eq!(stop_reason_to_openai("end_turn"), "stop");
        assert_eq!(stop_reason_to_openai("stop_sequence"), "stop");
        assert_eq!(stop_reason_to_openai("tool_use"), "tool_use");
        assert_eq!(stop_reason_to_openai("max_tokens"), "length");
        assert_eq!(stop_reason_to_openai("other"), "other");
    }

    #[test]
    fn parse_response_text_only() {
        let resp = AnthropicResponse {
            content: vec![ResponseContentBlock {
                kind: "text".into(),
                text: Some("hello".into()),
                id: None,
                name: None,
                input: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let cr = parse_response(resp).unwrap();
        assert_eq!(cr.message.content.as_deref(), Some("hello"));
        assert_eq!(cr.finish_reason, "stop");
        assert!(cr.message.tool_calls.is_none());
        let u = cr.usage.unwrap();
        assert_eq!(u.prompt_tokens, 10);
        assert_eq!(u.completion_tokens, 5);
        assert_eq!(u.total_tokens, 15);
    }

    #[test]
    fn parse_response_tool_use() {
        let resp = AnthropicResponse {
            content: vec![
                ResponseContentBlock {
                    kind: "text".into(),
                    text: Some("let me check".into()),
                    id: None,
                    name: None,
                    input: None,
                },
                ResponseContentBlock {
                    kind: "tool_use".into(),
                    text: None,
                    id: Some("toolu_123".into()),
                    name: Some("bash".into()),
                    input: Some(serde_json::json!({"command": "ls"})),
                },
            ],
            stop_reason: Some("tool_use".into()),
            usage: ResponseUsage {
                input_tokens: 20,
                output_tokens: 10,
                cache_creation_input_tokens: None,
                cache_read_input_tokens: None,
            },
        };
        let cr = parse_response(resp).unwrap();
        assert_eq!(cr.message.content.as_deref(), Some("let me check"));
        assert_eq!(cr.finish_reason, "tool_use");
        let calls = cr.message.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "toolu_123");
        assert_eq!(calls[0].function.name, "bash");
        assert!(calls[0].function.arguments.contains("ls"));
    }

    #[test]
    fn parse_response_with_cache_usage() {
        let resp = AnthropicResponse {
            content: vec![ResponseContentBlock {
                kind: "text".into(),
                text: Some("ok".into()),
                id: None,
                name: None,
                input: None,
            }],
            stop_reason: Some("end_turn".into()),
            usage: ResponseUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_creation_input_tokens: Some(200),
                cache_read_input_tokens: Some(300),
            },
        };
        let cr = parse_response(resp).unwrap();
        let u = cr.usage.unwrap();
        // total = 100 + 50 + 300 + 200 = 650
        assert_eq!(u.total_tokens, 650);
    }

    #[test]
    fn parse_sse_text_delta() {
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#;
        let mut state = StreamState::default();
        let mut out = Vec::new();
        parse_anthropic_sse(data, &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::ContentDelta(t)) => assert_eq!(t, "hello"),
            _ => panic!("expected ContentDelta"),
        }
    }

    #[test]
    fn parse_sse_message_stop() {
        let data = r#"{"type":"message_stop"}"#;
        let mut state = StreamState::default();
        let mut out = Vec::new();
        parse_anthropic_sse(data, &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        assert!(matches!(&out[0], Ok(StreamEvent::Done { .. })));
    }

    #[test]
    fn parse_sse_message_start_with_usage() {
        let data = r#"{"type":"message_start","message":{"usage":{"input_tokens":100,"output_tokens":1,"cache_read_input_tokens":50}}}"#;
        let mut state = StreamState::default();
        let mut out = Vec::new();
        parse_anthropic_sse(data, &mut state, &mut out).unwrap();
        assert!(state.usage.is_some());
        let u = state.usage.unwrap();
        assert_eq!(u.prompt_tokens, 100);
        assert_eq!(u.total_tokens, 151); // 100 + 1 + 50
    }

    #[test]
    fn parse_sse_tool_use_delta() {
        let mut state = StreamState::default();
        // First: content_block_start
        let start = r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash"}}"#;
        let mut out = Vec::new();
        parse_anthropic_sse(start, &mut state, &mut out).unwrap();
        assert!(out.is_empty());

        // Then: input_json_delta
        let delta = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"comm"}}"#;
        parse_anthropic_sse(delta, &mut state, &mut out).unwrap();
        assert_eq!(out.len(), 1);
        match &out[0] {
            Ok(StreamEvent::ToolCallDelta {
                index,
                id,
                function_name,
                arguments_delta,
            }) => {
                assert_eq!(*index, 1);
                assert!(arguments_delta.as_deref().unwrap().contains("comm"));
                // First delta should have id and name.
                assert_eq!(id.as_deref(), Some("toolu_1"));
                assert_eq!(function_name.as_deref(), Some("bash"));
            }
            _ => panic!("expected ToolCallDelta"),
        }
    }

    #[test]
    fn parse_sse_invalid_json() {
        let mut state = StreamState::default();
        let mut out = Vec::new();
        let result = parse_anthropic_sse("not json", &mut state, &mut out);
        assert!(result.is_err());
    }
}

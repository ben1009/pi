use std::{collections::HashMap, time::Duration};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, Command},
    sync::Mutex,
};

use crate::tools::{Tool, ToolCtx, truncate};

// Global store for strings leaked by McpTool::new. Keeps them reachable so
// leak sanitizers (e.g. -Z sanitizer=leak) do not report them.
static MCP_TOOL_STRINGS: std::sync::LazyLock<std::sync::Mutex<Vec<&'static str>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Leak a string for the process lifetime while keeping it reachable from a
/// global root. This satisfies LeakSanitizer without changing the Tool trait.
fn leak_string(s: String) -> &'static str {
    let leaked: &'static str = Box::leak(s.into_boxed_str());
    MCP_TOOL_STRINGS.lock().unwrap().push(leaked);
    leaked
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 wire types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    id: u64,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

#[derive(Serialize)]
struct JsonRpcNotification {
    jsonrpc: &'static str,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 response. `id` uses `Value` to accept both numeric and string IDs.
#[derive(Deserialize, Debug)]
struct JsonRpcResponse {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    result: Option<serde_json::Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcError {
    code: i64,
    message: String,
}

// ---------------------------------------------------------------------------
// MCP protocol types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct InitializeParams {
    #[serde(rename = "protocolVersion")]
    protocol_version: &'static str,
    capabilities: ClientCapabilities,
    #[serde(rename = "clientInfo")]
    client_info: ClientInfo,
}

#[derive(Serialize)]
struct ClientCapabilities {}

#[derive(Serialize)]
struct ClientInfo {
    name: &'static str,
    version: &'static str,
}

#[derive(Deserialize, Debug)]
pub struct InitializeResult {
    #[serde(rename = "protocolVersion")]
    pub protocol_version: String,
    pub capabilities: ServerCapabilities,
    #[serde(rename = "serverInfo")]
    pub server_info: ServerInfo,
}

#[derive(Deserialize, Debug)]
pub struct ServerCapabilities {
    pub tools: Option<ToolsCapability>,
}

#[derive(Deserialize, Debug)]
pub struct ToolsCapability {
    #[serde(rename = "listChanged")]
    pub list_changed: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

#[derive(Deserialize, Debug)]
pub struct McpToolDef {
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "inputSchema")]
    pub input_schema: serde_json::Value,
}

#[derive(Deserialize, Debug)]
struct ListToolsResult {
    tools: Vec<McpToolDef>,
    #[serde(rename = "nextCursor")]
    next_cursor: Option<String>,
}

#[derive(Deserialize, Debug)]
struct CallToolResult {
    content: Vec<CallToolContent>,
    #[serde(rename = "isError")]
    is_error: Option<bool>,
}

#[derive(Deserialize, Debug)]
struct CallToolContent {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

const PROTOCOL_VERSION: &str = "2025-03-26";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// McpServer — manages a single MCP server over stdio transport
// ---------------------------------------------------------------------------

/// A connected MCP server communicating over stdio.
///
/// All methods are `&mut self` because the stdio transport is inherently
/// sequential (request-response). Wrap in `Arc<Mutex<…>>` for shared access.
pub struct McpServer {
    child: Child,
    stdin: tokio::process::ChildStdin,
    stdout: BufReader<tokio::process::ChildStdout>,
    next_id: u64,
    pub server_info: ServerInfo,
}

impl McpServer {
    /// Launch an MCP server subprocess and perform the initialize handshake.
    pub async fn connect(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        let mut cmd = Command::new(command);
        cmd.args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true);
        for (k, v) in env {
            cmd.env(k, v);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("failed to spawn MCP server '{command}': {e}"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let stdout = BufReader::new(stdout);

        let mut server = Self {
            child,
            stdin,
            stdout,
            next_id: 1,
            server_info: ServerInfo {
                name: String::new(),
                version: String::new(),
            },
        };

        // Initialize handshake with timeout.
        let init_result: InitializeResult = tokio::time::timeout(
            REQUEST_TIMEOUT,
            server.request(
                "initialize",
                Some(serde_json::to_value(InitializeParams {
                    protocol_version: PROTOCOL_VERSION,
                    capabilities: ClientCapabilities {},
                    client_info: ClientInfo {
                        name: "pi-rs",
                        version: env!("CARGO_PKG_VERSION"),
                    },
                })?),
            ),
        )
        .await
        .map_err(|_| anyhow!("MCP server '{command}' timed out during initialization"))??;

        if init_result.protocol_version != PROTOCOL_VERSION {
            eprintln!(
                "pi: warning: MCP server '{command}' protocol version '{}', expected '{PROTOCOL_VERSION}'",
                init_result.protocol_version
            );
        }

        server.server_info = init_result.server_info;

        // Send initialized notification.
        server.notify("notifications/initialized", None).await?;

        Ok(server)
    }

    /// List all tools exposed by this server, following pagination cursors.
    pub async fn list_tools(&mut self) -> Result<Vec<McpToolDef>> {
        let mut all_tools = Vec::new();
        let mut cursor: Option<String> = None;

        loop {
            let params = cursor.as_ref().map(|c| serde_json::json!({ "cursor": c }));
            let page: ListToolsResult =
                tokio::time::timeout(REQUEST_TIMEOUT, self.request("tools/list", params))
                    .await
                    .map_err(|_| anyhow!("MCP server timed out listing tools"))??;
            all_tools.extend(page.tools);
            match page.next_cursor {
                Some(next) => cursor = Some(next),
                None => break,
            }
        }

        Ok(all_tools)
    }

    /// Call a tool on this server.
    pub async fn call_tool(&mut self, name: &str, arguments: serde_json::Value) -> Result<String> {
        let result: CallToolResult = tokio::time::timeout(
            REQUEST_TIMEOUT,
            self.request(
                "tools/call",
                Some(serde_json::json!({
                    "name": name,
                    "arguments": arguments,
                })),
            ),
        )
        .await
        .map_err(|_| anyhow!("MCP server timed out calling tool '{name}'"))??;

        let mut parts = Vec::new();
        for content in &result.content {
            match content.kind.as_str() {
                "text" => {
                    if let Some(text) = &content.text {
                        parts.push(text.clone());
                    }
                }
                other => {
                    // Non-text content (image, resource, etc.) — include a note
                    // so the LLM knows something was returned but couldn't be
                    // rendered as text.
                    parts.push(format!("[{other} content omitted]"));
                }
            }
        }

        if result.is_error == Some(true) {
            Ok(format!("Error: {}", parts.join("\n")))
        } else {
            Ok(parts.join("\n"))
        }
    }

    /// Send a JSON-RPC request and wait for the matching response.
    ///
    /// Server-initiated requests (messages with an `id` that doesn't match ours)
    /// are logged and skipped — we don't handle server-to-client method calls
    /// in this initial implementation.
    async fn request<T: serde::de::DeserializeOwned>(
        &mut self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<T> {
        let id = self.next_id;
        self.next_id += 1;

        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method: method.to_owned(),
            params,
        };
        let msg = serde_json::to_string(&req)?;
        self.send_line(&msg).await?;

        // Read response lines until we get one matching our id.
        loop {
            let line = self.read_line().await?;
            let trimmed = line.trim();
            if trimmed.is_empty() || (!trimmed.starts_with('{') && !trimmed.starts_with('[')) {
                continue;
            }

            // Handle JSON-RPC batch messages (array of requests/responses).
            if trimmed.starts_with('[') {
                let batch: Vec<serde_json::Value> = match serde_json::from_str(trimmed) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("pi: warning: MCP batch parse error: {e}");
                        continue;
                    }
                };
                for item in batch {
                    if let Some(resp) = self.process_response_value(item, id).await? {
                        return Ok(resp);
                    }
                }
                continue;
            }

            // Single message — parse and process.
            let value: serde_json::Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("pi: warning: MCP JSON-RPC parse error: {e} line: {trimmed}");
                    continue;
                }
            };
            if let Some(resp) = self.process_response_value(value, id).await? {
                return Ok(resp);
            }
        }
    }

    /// Send a JSON-RPC error response for method not found.
    async fn send_method_not_found(&mut self, req_id: &serde_json::Value) -> Result<()> {
        let error_resp = serde_json::json!({
            "jsonrpc": "2.0",
            "id": req_id,
            "error": { "code": -32601, "message": "Method not found" }
        });
        let msg = serde_json::to_string(&error_resp)?;
        self.send_line(&msg).await
    }

    /// Process a single JSON-RPC response value from a batch or single message.
    /// Returns Ok(Some(result)) if this is our response, Ok(None) if processed but not ours.
    async fn process_response_value<T: serde::de::DeserializeOwned>(
        &mut self,
        value: serde_json::Value,
        expected_id: u64,
    ) -> Result<Option<T>> {
        let resp: JsonRpcResponse =
            serde_json::from_value(value).map_err(|e| anyhow!("MCP JSON-RPC parse error: {e}"))?;

        // Skip notifications (no id).
        if resp.id.is_none() {
            return Ok(None);
        }

        // Server-initiated request — send error response to prevent deadlock.
        if resp.id.as_ref() != Some(&serde_json::Value::Number(expected_id.into())) {
            if let Some(req_id) = &resp.id {
                self.send_method_not_found(req_id).await?;
            }
            return Ok(None);
        }

        if let Some(err) = resp.error {
            return Err(anyhow!("MCP error {}: {}", err.code, err.message));
        }

        let result = resp.result.ok_or_else(|| anyhow!("MCP: no result"))?;
        Ok(Some(serde_json::from_value(result).map_err(|e| {
            anyhow!("MCP: failed to parse result: {e}")
        })?))
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(&mut self, method: &str, params: Option<serde_json::Value>) -> Result<()> {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0",
            method: method.to_owned(),
            params,
        };
        let msg = serde_json::to_string(&notif)?;
        self.send_line(&msg).await
    }

    async fn send_line(&mut self, line: &str) -> Result<()> {
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    async fn read_line(&mut self) -> Result<String> {
        let mut line = String::new();
        self.stdout.read_line(&mut line).await?;
        if line.is_empty() {
            return Err(anyhow!("MCP server closed connection"));
        }
        Ok(line.trim_end().to_owned())
    }

    /// Shut down the server process, waiting for it to exit.
    pub async fn shutdown(&mut self) -> Result<()> {
        self.child.kill().await.ok();
        self.child.wait().await.ok();
        Ok(())
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        // Best-effort kill on drop. `start_kill()` is synchronous.
        let _ = self.child.start_kill();
    }
}

// ---------------------------------------------------------------------------
// McpTool — adapts an MCP tool into the pi-rs Tool trait
// ---------------------------------------------------------------------------

/// Wraps a remote MCP tool as a local `Tool` implementation.
pub struct McpTool {
    prefixed_name: &'static str,
    desc: &'static str,
    schema: serde_json::Value,
    remote_name: String,
    server: std::sync::Arc<Mutex<McpServer>>,
}

/// Sanitize a tool name for OpenAI-compatible function constraints.
/// Replaces characters outside `[a-zA-Z0-9_-]` with `_` and truncates to 64 chars.
fn sanitize_tool_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect()
}

impl McpTool {
    pub fn new(
        server_name: &str,
        mcp_tool: &McpToolDef,
        server: std::sync::Arc<Mutex<McpServer>>,
    ) -> Self {
        let prefixed = sanitize_tool_name(&format!("{}__{}", server_name, mcp_tool.name));
        let desc = mcp_tool
            .description
            .clone()
            .unwrap_or_else(|| "MCP tool".to_owned());
        Self {
            // Leak once at construction — name and description are fixed for
            // the process lifetime, so this is a bounded one-time cost.
            // leak_string keeps them reachable from a global root so leak
            // sanitizers do not flag them.
            prefixed_name: leak_string(prefixed),
            desc: leak_string(desc),
            schema: mcp_tool.input_schema.clone(),
            remote_name: mcp_tool.name.clone(),
            server,
        }
    }
}

#[async_trait]
impl Tool for McpTool {
    fn name(&self) -> &'static str {
        self.prefixed_name
    }

    fn description(&self) -> &'static str {
        self.desc
    }

    fn schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String> {
        let mut server = self.server.lock().await;
        let result = server.call_tool(&self.remote_name, input).await?;
        Ok(truncate(result, ctx.max_output))
    }
}

// ---------------------------------------------------------------------------
// Public API — connect to MCP servers and register their tools
// ---------------------------------------------------------------------------

/// Configuration for a single MCP server.
///
/// ```json
/// {
///   "name": "fs",
///   "command": "npx",
///   "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
///   "env": {}
/// }
/// ```
#[derive(Deserialize)]
pub struct McpServerConfig {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
}

impl McpServerConfig {
    pub fn from_json(value: &serde_json::Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|e| anyhow!("invalid MCP server config: {e}"))
    }
}

/// Connect to configured MCP servers and register their tools into the registry.
/// Returns the names of successfully connected servers.
pub async fn connect_servers(
    configs: &[McpServerConfig],
    registry: &mut crate::tools::Registry,
) -> Result<Vec<String>> {
    use std::sync::Arc;

    let mut server_names = Vec::new();
    let mut seen_names = std::collections::HashSet::new();
    for cfg in configs {
        if !seen_names.insert(&cfg.name) {
            eprintln!(
                "pi: warning: duplicate MCP server name '{}', skipping",
                cfg.name
            );
            continue;
        }
        let mut server = match McpServer::connect(&cfg.command, &cfg.args, &cfg.env).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!(
                    "pi: warning: failed to connect MCP server '{}': {e}",
                    cfg.name
                );
                continue;
            }
        };

        let tools = match server.list_tools().await {
            Ok(t) => t,
            Err(e) => {
                eprintln!(
                    "pi: warning: failed to list tools from MCP server '{}': {e}",
                    cfg.name
                );
                server.shutdown().await.ok();
                continue;
            }
        };

        let server = Arc::new(Mutex::new(server));
        let count = tools.len();

        for mcp_tool in &tools {
            let tool = McpTool::new(&cfg.name, mcp_tool, server.clone());
            if registry.get(tool.name()).is_some() {
                eprintln!(
                    "pi: warning: tool name collision '{}' (from server '{}'), skipping",
                    tool.name(),
                    cfg.name
                );
                continue;
            }
            registry.register(Box::new(tool));
        }

        eprintln!("pi: connected to MCP server '{}' ({count} tools)", cfg.name);
        server_names.push(cfg.name.clone());
    }
    Ok(server_names)
}

/// Parse MCP server configs from `--mcp-server` CLI values.
/// Each entry is a JSON object string, or a path to a JSON file containing one.
pub fn parse_mcp_configs(values: &[String]) -> Result<Vec<McpServerConfig>> {
    let mut configs = Vec::new();
    for val in values {
        // Try JSON parse first; fall back to treating as a file path.
        let json_val: serde_json::Value = match serde_json::from_str(val) {
            Ok(v) => v,
            Err(_) => {
                let content = std::fs::read_to_string(val)
                    .map_err(|e| anyhow!("failed to read MCP config file '{val}': {e}"))?;
                serde_json::from_str(&content)?
            }
        };
        configs.push(McpServerConfig::from_json(&json_val)?);
    }
    Ok(configs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_preserves_valid_chars() {
        assert_eq!(sanitize_tool_name("abc_123-XYZ"), "abc_123-XYZ");
    }

    #[test]
    fn sanitize_replaces_special_chars() {
        assert_eq!(sanitize_tool_name("tool.name/here"), "tool_name_here");
        assert_eq!(sanitize_tool_name("a:b:c"), "a_b_c");
        assert_eq!(sanitize_tool_name("foo bar"), "foo_bar");
    }

    #[test]
    fn sanitize_truncates_to_64_chars() {
        let long_name = "a".repeat(100);
        let result = sanitize_tool_name(&long_name);
        assert_eq!(result.len(), 64);
        assert_eq!(result.chars().count(), 64);
    }

    #[test]
    fn sanitize_multibyte_utf8_boundary() {
        // 'ñ' is 2 bytes in UTF-8 — truncation must not split it
        let name = "a".repeat(63) + "ñ";
        let result = sanitize_tool_name(&name);
        assert_eq!(result.chars().count(), 64);
        // The 'ñ' should be replaced with '_' since it's not ASCII alphanumeric
        assert!(result.ends_with('_'));
    }

    #[test]
    fn sanitize_empty_string() {
        assert_eq!(sanitize_tool_name(""), "");
    }

    #[test]
    fn sanitize_all_special_chars() {
        assert_eq!(sanitize_tool_name("./:!@#$%"), "________");
    }

    #[test]
    fn mcp_server_config_from_json() {
        let json = serde_json::json!({
            "name": "fs",
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
            "env": {"KEY": "value"}
        });
        let cfg = McpServerConfig::from_json(&json).unwrap();
        assert_eq!(cfg.name, "fs");
        assert_eq!(cfg.command, "npx");
        assert_eq!(cfg.args.len(), 3);
        assert_eq!(cfg.env.get("KEY").unwrap(), "value");
    }

    #[test]
    fn mcp_server_config_minimal() {
        let json = serde_json::json!({
            "name": "test",
            "command": "echo"
        });
        let cfg = McpServerConfig::from_json(&json).unwrap();
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.command, "echo");
        assert!(cfg.args.is_empty());
        assert!(cfg.env.is_empty());
    }

    #[test]
    fn mcp_server_config_invalid_json() {
        let json = serde_json::json!({"name": "test"}); // missing "command"
        let result = McpServerConfig::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn parse_mcp_configs_from_json_string() {
        let values = vec![r#"{"name":"test","command":"echo"}"#.to_owned()];
        let configs = parse_mcp_configs(&values).unwrap();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].name, "test");
    }

    #[test]
    fn parse_mcp_configs_invalid_json_string() {
        let values = vec!["not json".to_owned()];
        // Should fail since it tries to parse as JSON, then as file path (which doesn't exist).
        let result = parse_mcp_configs(&values);
        assert!(result.is_err());
    }

    #[test]
    fn parse_mcp_configs_empty() {
        let configs = parse_mcp_configs(&[]).unwrap();
        assert!(configs.is_empty());
    }

    #[test]
    fn mcp_tool_content_text() {
        let content = CallToolContent {
            kind: "text".to_owned(),
            text: Some("result text".to_owned()),
        };
        assert_eq!(content.kind, "text");
        assert_eq!(content.text.as_deref(), Some("result text"));
    }

    #[test]
    fn mcp_tool_content_non_text() {
        let content = CallToolContent {
            kind: "image".to_owned(),
            text: None,
        };
        assert_eq!(content.kind, "image");
        assert!(content.text.is_none());
    }

    #[test]
    fn json_rpc_request_serialization() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 1,
            method: "initialize".to_owned(),
            params: Some(serde_json::json!({"key": "value"})),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"id\":1"));
        assert!(json.contains("\"method\":\"initialize\""));
    }

    #[test]
    fn json_rpc_notification_serialization() {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0",
            method: "notifications/initialized".to_owned(),
            params: None,
        };
        let json = serde_json::to_string(&notif).unwrap();
        assert!(json.contains("notifications/initialized"));
        assert!(!json.contains("params"));
    }

    #[test]
    fn json_rpc_response_with_error() {
        let json_str =
            r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, -32601);
        assert!(resp.result.is_none());
    }

    #[test]
    fn json_rpc_response_with_result() {
        let json_str = r#"{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"test","version":"1.0"}}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.error.is_none());
        assert!(resp.result.is_some());
    }

    #[test]
    fn json_rpc_response_notification() {
        // Notifications have no id.
        let json_str = r#"{"jsonrpc":"2.0","method":"some/notification","params":{}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.id.is_none());
    }

    #[test]
    fn initialize_result_deserialization() {
        let json_str = r#"{"protocolVersion":"2025-03-26","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"test-server","version":"0.1.0"}}"#;
        let result: InitializeResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.protocol_version, "2025-03-26");
        assert_eq!(result.server_info.name, "test-server");
        assert_eq!(result.server_info.version, "0.1.0");
        assert!(result.capabilities.tools.is_some());
        assert!(result.capabilities.tools.unwrap().list_changed.unwrap());
    }

    #[test]
    fn mcp_tool_def_deserialization() {
        let json_str = r#"{"name":"read_file","description":"Read a file","inputSchema":{"type":"object","properties":{"path":{"type":"string"}}}}"#;
        let def: McpToolDef = serde_json::from_str(json_str).unwrap();
        assert_eq!(def.name, "read_file");
        assert_eq!(def.description.unwrap(), "Read a file");
    }

    #[test]
    fn list_tools_result_deserialization() {
        let json_str = r#"{"tools":[{"name":"tool1","inputSchema":{}},{"name":"tool2","inputSchema":{}}],"nextCursor":"abc"}"#;
        let result: ListToolsResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.tools.len(), 2);
        assert_eq!(result.next_cursor.unwrap(), "abc");
    }

    #[test]
    fn call_tool_result_deserialization() {
        let json_str = r#"{"content":[{"type":"text","text":"hello"}],"isError":false}"#;
        let result: CallToolResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.content.len(), 1);
        assert_eq!(result.content[0].text.as_deref(), Some("hello"));
        assert_eq!(result.is_error, Some(false));
    }

    #[test]
    fn call_tool_result_error() {
        let json_str = r#"{"content":[{"type":"text","text":"not found"}],"isError":true}"#;
        let result: CallToolResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn call_tool_result_no_is_error() {
        let json_str = r#"{"content":[{"type":"text","text":"ok"}]}"#;
        let result: CallToolResult = serde_json::from_str(json_str).unwrap();
        assert!(result.is_error.is_none());
    }

    #[test]
    fn server_capabilities_deserialization() {
        let json_str = r#"{"tools":{"listChanged":true}}"#;
        let caps: ServerCapabilities = serde_json::from_str(json_str).unwrap();
        assert!(caps.tools.is_some());
        assert!(caps.tools.unwrap().list_changed.unwrap());
    }

    #[test]
    fn server_capabilities_no_tools() {
        let json_str = r#"{}"#;
        let caps: ServerCapabilities = serde_json::from_str(json_str).unwrap();
        assert!(caps.tools.is_none());
    }

    #[test]
    fn server_info_deserialization() {
        let json_str = r#"{"name":"test-server","version":"1.0.0"}"#;
        let info: ServerInfo = serde_json::from_str(json_str).unwrap();
        assert_eq!(info.name, "test-server");
        assert_eq!(info.version, "1.0.0");
    }

    #[test]
    fn tools_capability_deserialization() {
        let json_str = r#"{"listChanged":false}"#;
        let cap: ToolsCapability = serde_json::from_str(json_str).unwrap();
        assert_eq!(cap.list_changed, Some(false));
    }

    #[test]
    fn tools_capability_no_list_changed() {
        let json_str = r#"{}"#;
        let cap: ToolsCapability = serde_json::from_str(json_str).unwrap();
        assert!(cap.list_changed.is_none());
    }

    #[test]
    fn call_tool_content_multiple_types() {
        let json_str =
            r#"{"content":[{"type":"text","text":"hello"},{"type":"image"}],"isError":false}"#;
        let result: CallToolResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.content.len(), 2);
        assert_eq!(result.content[0].kind, "text");
        assert_eq!(result.content[0].text.as_deref(), Some("hello"));
        assert_eq!(result.content[1].kind, "image");
        assert!(result.content[1].text.is_none());
    }

    #[test]
    fn list_tools_result_no_cursor() {
        let json_str = r#"{"tools":[{"name":"t","inputSchema":{}}]}"#;
        let result: ListToolsResult = serde_json::from_str(json_str).unwrap();
        assert_eq!(result.tools.len(), 1);
        assert!(result.next_cursor.is_none());
    }

    #[test]
    fn mcp_tool_def_no_description() {
        let json_str = r#"{"name":"tool","inputSchema":{"type":"object"}}"#;
        let def: McpToolDef = serde_json::from_str(json_str).unwrap();
        assert_eq!(def.name, "tool");
        assert!(def.description.is_none());
    }

    #[test]
    fn json_rpc_request_no_params() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id: 5,
            method: "tools/list".to_owned(),
            params: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("params"));
    }

    #[test]
    fn json_rpc_notification_with_params() {
        let notif = JsonRpcNotification {
            jsonrpc: "2.0",
            method: "test/method".to_owned(),
            params: Some(serde_json::json!({"key": "value"})),
        };
        let json = serde_json::to_string(&notif).unwrap();
        assert!(json.contains("params"));
        assert!(json.contains("key"));
    }

    #[test]
    fn json_rpc_response_string_id() {
        let json_str = r#"{"jsonrpc":"2.0","id":"abc-123","result":{"ok":true}}"#;
        let resp: JsonRpcResponse = serde_json::from_str(json_str).unwrap();
        assert!(resp.id.is_some());
        assert_eq!(resp.id.unwrap(), serde_json::json!("abc-123"));
    }

    #[test]
    fn initialize_result_no_tools_capability() {
        let json_str = r#"{"protocolVersion":"2025-03-26","capabilities":{},"serverInfo":{"name":"s","version":"v"}}"#;
        let result: InitializeResult = serde_json::from_str(json_str).unwrap();
        assert!(result.capabilities.tools.is_none());
    }

    #[test]
    fn sanitize_preserves_underscores_and_hyphens() {
        assert_eq!(sanitize_tool_name("my_tool-name"), "my_tool-name");
    }

    #[test]
    fn sanitize_mixed_valid_invalid() {
        assert_eq!(sanitize_tool_name("tool.name/v2"), "tool_name_v2");
    }
}

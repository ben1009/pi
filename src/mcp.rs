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
    let sanitized: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.len() > 64 {
        sanitized[..64].to_owned()
    } else {
        sanitized
    }
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
            prefixed_name: Box::leak(prefixed.into_boxed_str()),
            desc: Box::leak(desc.into_boxed_str()),
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

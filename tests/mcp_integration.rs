use std::collections::HashMap;

use pi_rs::{
    mcp::{McpServer, McpServerConfig, connect_servers, parse_mcp_configs},
    tools::Registry,
};

#[tokio::test]
async fn mcp_server_connect_and_handshake() {
    let mut server = McpServer::connect(
        "bash",
        &["tests/mock_mcp_server.sh".to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect should succeed");

    assert_eq!(server.server_info.name, "mock-server");
    assert_eq!(server.server_info.version, "0.1.0");

    server.shutdown().await.ok();
}

#[tokio::test]
async fn mcp_server_list_tools() {
    let mut server = McpServer::connect(
        "bash",
        &["tests/mock_mcp_server.sh".to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    let tools = server.list_tools().await.expect("list_tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "echo");
    assert_eq!(tools[0].description.as_deref(), Some("Echo input back"));

    server.shutdown().await.ok();
}

#[tokio::test]
async fn mcp_server_call_tool() {
    let mut server = McpServer::connect(
        "bash",
        &["tests/mock_mcp_server.sh".to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    let result = server
        .call_tool("echo", serde_json::json!({"text": "hello"}))
        .await
        .expect("call_tool");
    assert_eq!(result, "mock result");

    server.shutdown().await.ok();
}

#[tokio::test]
async fn mcp_server_connect_with_env() {
    let mut env = HashMap::new();
    env.insert("TEST_KEY".to_string(), "test_value".to_string());

    let mut server = McpServer::connect("bash", &["tests/mock_mcp_server.sh".to_string()], &env)
        .await
        .expect("connect with env");

    assert_eq!(server.server_info.name, "mock-server");
    server.shutdown().await.ok();
}

#[tokio::test]
async fn mcp_server_connect_invalid_command() {
    let result = McpServer::connect("nonexistent_command_xyz", &[], &HashMap::new()).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn parse_mcp_configs_from_json_string() {
    let values =
        vec![r#"{"name":"test","command":"echo","args":["hello"],"env":{"K":"V"}}"#.to_string()];
    let configs = parse_mcp_configs(&values).unwrap();
    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0].name, "test");
    assert_eq!(configs[0].command, "echo");
    assert_eq!(configs[0].args, vec!["hello"]);
    assert_eq!(configs[0].env.get("K").unwrap(), "V");
}

#[tokio::test]
async fn parse_mcp_configs_from_file() {
    let dir = std::env::temp_dir().join(format!("mcp-test-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("config.json");
    std::fs::write(&path, r#"{"name":"file-test","command":"echo"}"#).unwrap();

    let configs = parse_mcp_configs(&[path.display().to_string()]).unwrap();
    assert_eq!(configs.len(), 1);
    assert_eq!(configs[0].name, "file-test");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn mcp_server_tool_call_result_formatting() {
    let mut server = McpServer::connect(
        "bash",
        &["tests/mock_mcp_server.sh".to_string()],
        &HashMap::new(),
    )
    .await
    .expect("connect");

    // The mock always returns "mock result"
    let result = server
        .call_tool("echo", serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(result, "mock result");

    server.shutdown().await.ok();
}

#[tokio::test]
async fn connect_servers_registers_tools() {
    let configs = vec![McpServerConfig {
        name: "mock".to_string(),
        command: "bash".to_string(),
        args: vec!["tests/mock_mcp_server.sh".to_string()],
        env: HashMap::new(),
    }];
    let mut registry = Registry::with_defaults();
    let names = connect_servers(&configs, &mut registry).await.unwrap();
    assert_eq!(names, vec!["mock"]);
    // The mock server exposes an "echo" tool, which should be registered as "mock__echo".
    assert!(registry.get("mock__echo").is_some());
}

#[tokio::test]
async fn connect_servers_duplicate_name_skipped() {
    let configs = vec![
        McpServerConfig {
            name: "dup".to_string(),
            command: "bash".to_string(),
            args: vec!["tests/mock_mcp_server.sh".to_string()],
            env: HashMap::new(),
        },
        McpServerConfig {
            name: "dup".to_string(),
            command: "bash".to_string(),
            args: vec!["tests/mock_mcp_server.sh".to_string()],
            env: HashMap::new(),
        },
    ];
    let mut registry = Registry::with_defaults();
    let names = connect_servers(&configs, &mut registry).await.unwrap();
    // Only one should connect; the duplicate is skipped.
    assert_eq!(names.len(), 1);
}

#[tokio::test]
async fn connect_servers_invalid_command_continues() {
    let configs = vec![
        McpServerConfig {
            name: "bad".to_string(),
            command: "nonexistent_cmd_xyz".to_string(),
            args: vec![],
            env: HashMap::new(),
        },
        McpServerConfig {
            name: "good".to_string(),
            command: "bash".to_string(),
            args: vec!["tests/mock_mcp_server.sh".to_string()],
            env: HashMap::new(),
        },
    ];
    let mut registry = Registry::with_defaults();
    let names = connect_servers(&configs, &mut registry).await.unwrap();
    // The bad server should fail silently, the good one should connect.
    assert_eq!(names, vec!["good"]);
    assert!(registry.get("good__echo").is_some());
}

#[tokio::test]
async fn connect_servers_empty_config() {
    let mut registry = Registry::with_defaults();
    let names = connect_servers(&[], &mut registry).await.unwrap();
    assert!(names.is_empty());
}

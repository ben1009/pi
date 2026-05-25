use pi_rs::llm::{ChatRequest, LlmClient, Message, Role, anthropic::AnthropicNativeClient};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{header, method, path},
};

fn req(model: &str, messages: Vec<Message>) -> ChatRequest {
    ChatRequest {
        model: model.to_owned(),
        messages,
        tools: vec![],
        max_tokens: 256,
    }
}

fn req_with_tools(
    model: &str,
    messages: Vec<Message>,
    tools: Vec<pi_rs::llm::ToolDef>,
) -> ChatRequest {
    ChatRequest {
        model: model.to_owned(),
        messages,
        tools,
        max_tokens: 256,
    }
}

fn anthropic_text_response(text: &str) -> Value {
    json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5
        }
    })
}

fn anthropic_tool_use_response(id: &str, name: &str, input: Value) -> Value {
    json!({
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": id, "name": name, "input": input}
        ],
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 20,
            "output_tokens": 10
        }
    })
}

fn parse_body(req: &Request) -> Value {
    serde_json::from_slice(&req.body).expect("decode JSON body")
}

#[tokio::test]
async fn plain_reply() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(anthropic_text_response("hello back")),
        )
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let resp = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect("complete should succeed");
    assert_eq!(resp.message.content.as_deref(), Some("hello back"));
    assert_eq!(resp.finish_reason, "stop");
    let usage = resp.usage.unwrap();
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
}

#[tokio::test]
async fn sends_correct_headers() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "my-secret-key"))
        .and(header("anthropic-version", "2023-06-01"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client = AnthropicNativeClient::with_base_url(
        &format!("{}/v1", server.uri()),
        "my-secret-key".into(),
    );
    client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();
}

#[tokio::test]
async fn system_prompt_extracted_and_has_cache_control() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    client
        .complete(req(
            "m",
            vec![Message::system("you are helpful"), Message::user("hi")],
        ))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let body = parse_body(&requests[0]);

    // System should be an array of content blocks.
    let system = body.get("system").expect("system field");
    let blocks = system.as_array().expect("system is array");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "text");
    assert_eq!(blocks[0]["text"], "you are helpful");
    assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");

    // Messages should not contain the system message.
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[tokio::test]
async fn tool_definitions_with_cache_control() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let tools = vec![
        pi_rs::llm::ToolDef {
            kind: "function",
            function: pi_rs::llm::ToolDefFunction {
                name: "bash",
                description: "run a shell command",
                parameters: json!({"type": "object"}),
            },
        },
        pi_rs::llm::ToolDef {
            kind: "function",
            function: pi_rs::llm::ToolDefFunction {
                name: "read",
                description: "read a file",
                parameters: json!({"type": "object"}),
            },
        },
    ];
    client
        .complete(req_with_tools("m", vec![Message::user("hi")], tools))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let body = parse_body(&requests[0]);
    let api_tools = body["tools"].as_array().unwrap();
    assert_eq!(api_tools.len(), 2);

    // Anthropic format: name, description, input_schema (not function.name/description/parameters).
    assert_eq!(api_tools[0]["name"], "bash");
    assert_eq!(api_tools[0]["description"], "run a shell command");
    assert!(api_tools[0]["input_schema"].is_object());
    // First tool should NOT have cache_control.
    assert!(api_tools[0]["cache_control"].is_null());

    // Last tool SHOULD have cache_control.
    assert_eq!(api_tools[1]["name"], "read");
    assert_eq!(api_tools[1]["cache_control"]["type"], "ephemeral");
}

#[tokio::test]
async fn tool_call_round_trip() {
    let server = MockServer::start().await;

    // First response: tool_use
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(anthropic_tool_use_response(
                "toolu_123",
                "bash",
                json!({"command": "echo hi"}),
            )),
        )
        .up_to_n_times(1)
        .mount(&server)
        .await;

    // Second response: text reply
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("done")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let tools = vec![pi_rs::llm::ToolDef {
        kind: "function",
        function: pi_rs::llm::ToolDefFunction {
            name: "bash",
            description: "run a shell command",
            parameters: json!({"type": "object"}),
        },
    }];

    let mut messages = vec![Message::user("run echo hi")];
    let first = client
        .complete(req_with_tools("m", messages.clone(), tools.clone()))
        .await
        .expect("first call");

    let calls = first.message.tool_calls.clone().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "toolu_123");
    assert_eq!(calls[0].function.name, "bash");

    // Build assistant + tool result messages.
    messages.push(Message {
        role: Role::Assistant,
        content: first.message.content.clone(),
        tool_calls: first.message.tool_calls.clone(),
        tool_call_id: None,
    });
    messages.push(Message {
        role: Role::Tool,
        content: Some("hi\n".into()),
        tool_calls: None,
        tool_call_id: Some(calls[0].id.clone()),
    });

    let second = client
        .complete(req_with_tools("m", messages.clone(), tools))
        .await
        .expect("second call");
    assert_eq!(second.message.content.as_deref(), Some("done"));

    // Verify the second request has correct Anthropic message format.
    let requests = server.received_requests().await.unwrap();
    assert_eq!(requests.len(), 2);
    let body = parse_body(&requests[1]);
    let msgs = body["messages"].as_array().unwrap();

    // Should have: user("run echo hi"), assistant(text + tool_use), user(tool_result)
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[2]["role"], "user");

    // The tool result should be in the user message as a content block.
    let tool_result_blocks = msgs[2]["content"].as_array().unwrap();
    assert_eq!(tool_result_blocks.len(), 1);
    assert_eq!(tool_result_blocks[0]["type"], "tool_result");
    assert_eq!(tool_result_blocks[0]["tool_use_id"], "toolu_123");
    assert_eq!(tool_result_blocks[0]["content"], "hi\n");
}

#[tokio::test]
async fn api_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad request"}
        })))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let err = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect_err("should error on 400");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("400"),
        "error should mention status code: {msg}"
    );
}

#[tokio::test]
async fn api_error_body_truncation() {
    let server = MockServer::start().await;
    let big_body = "x".repeat(5_000);
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string(big_body))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let err = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect_err("should error on 500");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("truncated"),
        "error should mention truncation: {msg}"
    );
    assert!(msg.len() < 5_000, "error should be shorter than raw body");
}

#[tokio::test]
async fn usage_with_cache_tokens() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 200,
                "cache_read_input_tokens": 300
            }
        })))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let resp = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();
    let usage = resp.usage.unwrap();
    // total = 100 + 50 + 300 + 200 = 650
    assert_eq!(usage.total_tokens, 650);
    assert_eq!(usage.prompt_tokens, 100);
    assert_eq!(usage.completion_tokens, 50);
}

// ── Streaming tests ──

fn sse_message_start() -> String {
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-sonnet-4-6\",\"stop_reason\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n".to_owned()
}

fn sse_content_delta(text: &str) -> String {
    format!(
        "data: {{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{{\"type\":\"text_delta\",\"text\":\"{}\"}}}}\n\n",
        text
    )
}

fn sse_content_block_start() -> &'static str {
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
}

fn sse_content_block_stop() -> &'static str {
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
}

fn sse_message_stop() -> &'static str {
    "data: {\"type\":\"message_stop\"}\n\n"
}

fn sse_message_delta() -> &'static str {
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n"
}

#[tokio::test]
async fn streaming_content() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}{}{}{}",
        sse_message_start(),
        sse_content_block_start(),
        sse_content_delta("hello "),
        sse_content_delta("world"),
        sse_content_block_stop(),
        sse_message_delta(),
    ) + sse_message_stop();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut texts = Vec::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            pi_rs::llm::StreamEvent::ContentDelta(t) => texts.push(t),
            pi_rs::llm::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert_eq!(texts.join(""), "hello world");
}

#[tokio::test]
async fn streaming_tool_calls() {
    let server = MockServer::start().await;
    let tool_start = format!(
        "data: {}\n\n",
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"bash"}})
    );
    let tool_delta1 = format!(
        "data: {}\n\n",
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"comm"}})
    );
    let tool_delta2 = format!(
        "data: {}\n\n",
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"and\":\"ls\"}"}})
    );
    let body = format!(
        "{}{}{}{}{}{}{}",
        sse_message_start(),
        sse_content_block_start(),
        sse_content_block_stop(),
        tool_start,
        tool_delta1,
        tool_delta2,
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
    ) + sse_message_delta()
        + sse_message_stop();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> =
        std::collections::BTreeMap::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            pi_rs::llm::StreamEvent::ToolCallDelta {
                index,
                id,
                function_name,
                arguments_delta,
            } => {
                let entry = tool_calls.entry(index).or_default();
                if let Some(id) = id {
                    entry.0 = id;
                }
                if let Some(name) = function_name {
                    entry.1 = name;
                }
                if let Some(args) = arguments_delta {
                    entry.2.push_str(&args);
                }
            }
            pi_rs::llm::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert_eq!(tool_calls.len(), 1);
    let (id, name, args) = tool_calls.get(&1).unwrap();
    assert_eq!(id, "toolu_1");
    assert_eq!(name, "bash");
    assert!(args.contains("ls"));
}

#[tokio::test]
async fn streaming_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let result = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn streaming_with_usage() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}{}",
        sse_message_start(),
        sse_content_block_start(),
        sse_content_delta("ok"),
        sse_content_block_stop(),
    ) + sse_message_delta()
        + sse_message_stop();

    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut done_usage = None;
    while let Some(event) = stream.next().await {
        if let Ok(pi_rs::llm::StreamEvent::Done { usage, .. }) = event {
            done_usage = usage;
            break;
        }
    }
    let usage = done_usage.expect("should have usage");
    assert_eq!(usage.prompt_tokens, 10);
    assert_eq!(usage.completion_tokens, 5);
}

#[tokio::test]
async fn multiple_system_messages() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    client
        .complete(req(
            "m",
            vec![
                Message::system("first"),
                Message::system("second"),
                Message::user("hi"),
            ],
        ))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let body = parse_body(&requests[0]);
    let system = body["system"].as_array().unwrap();
    assert_eq!(system.len(), 2);
    assert_eq!(system[0]["text"], "first");
    assert_eq!(system[1]["text"], "second");
    assert_eq!(system[1]["cache_control"]["type"], "ephemeral");
    // First should NOT have cache_control.
    assert!(system[0]["cache_control"].is_null());
}

#[tokio::test]
async fn max_tokens_in_request() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let body = parse_body(&requests[0]);
    assert_eq!(body["max_tokens"], 256);
}

#[tokio::test]
async fn multiple_tool_results_merged() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .respond_with(ResponseTemplate::new(200).set_body_json(anthropic_text_response("ok")))
        .mount(&server)
        .await;

    let client =
        AnthropicNativeClient::with_base_url(&format!("{}/v1", server.uri()), "test-key".into());
    let messages = vec![
        Message::user("run tools"),
        Message {
            role: pi_rs::llm::Role::Assistant,
            content: Some("calling tools".into()),
            tool_calls: Some(vec![
                pi_rs::llm::ToolCall {
                    id: "tc1".into(),
                    kind: "function".into(),
                    function: pi_rs::llm::ToolCallFunction {
                        name: "bash".into(),
                        arguments: "{}".into(),
                    },
                },
                pi_rs::llm::ToolCall {
                    id: "tc2".into(),
                    kind: "function".into(),
                    function: pi_rs::llm::ToolCallFunction {
                        name: "read".into(),
                        arguments: "{}".into(),
                    },
                },
            ]),
            tool_call_id: None,
        },
        Message {
            role: pi_rs::llm::Role::Tool,
            content: Some("result1".into()),
            tool_calls: None,
            tool_call_id: Some("tc1".into()),
        },
        Message {
            role: pi_rs::llm::Role::Tool,
            content: Some("result2".into()),
            tool_calls: None,
            tool_call_id: Some("tc2".into()),
        },
    ];
    client
        .complete(req_with_tools("m", messages, vec![]))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let body = parse_body(&requests[0]);
    let msgs = body["messages"].as_array().unwrap();
    // Should have: user, assistant, user(merged tool results)
    assert_eq!(msgs.len(), 3);
    let tool_blocks = msgs[2]["content"].as_array().unwrap();
    assert_eq!(tool_blocks.len(), 2);
    assert_eq!(tool_blocks[0]["type"], "tool_result");
    assert_eq!(tool_blocks[0]["tool_use_id"], "tc1");
    assert_eq!(tool_blocks[0]["content"], "result1");
    assert_eq!(tool_blocks[1]["type"], "tool_result");
    assert_eq!(tool_blocks[1]["tool_use_id"], "tc2");
    assert_eq!(tool_blocks[1]["content"], "result2");
}

#!/bin/bash
# Minimal mock MCP server for testing.
# Reads JSON-RPC from stdin, writes responses to stdout.
while IFS= read -r line; do
    # Skip empty lines
    [ -z "$line" ] && continue

    # Parse method from the JSON
    method=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('method',''))" 2>/dev/null)
    id=$(echo "$line" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id','null'))" 2>/dev/null)

    case "$method" in
        initialize)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"protocolVersion\":\"2025-03-26\",\"capabilities\":{\"tools\":{\"listChanged\":true}},\"serverInfo\":{\"name\":\"mock-server\",\"version\":\"0.1.0\"}}}"
            ;;
        notifications/initialized)
            # Notification, no response needed
            ;;
        tools/list)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"tools\":[{\"name\":\"echo\",\"description\":\"Echo input back\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}}}}]}}"
            ;;
        tools/call)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"mock result\"}]}}"
            ;;
        *)
            echo "{\"jsonrpc\":\"2.0\",\"id\":$id,\"error\":{\"code\":-32601,\"message\":\"Method not found\"}}"
            ;;
    esac
done

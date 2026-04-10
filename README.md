# AIProxy

A single-port proxy that emulates both **OpenAI-compatible** (`/v1/*`) and **Ollama-compatible** (`/api/*`, `/tags`) endpoints. It forwards requests to a selectable upstream provider, making it easy to switch between different LLM backends while keeping your client configuration unchanged.

## Features

- **Dual Protocol Support** — Serve both OpenAI and Ollama API formats from a single port
- **Multiple Providers** — Connect to OpenCode, LM Studio, Ollama, or any custom OpenAI-compatible API
- **Model Mapping** — Remap model names on-the-fly (e.g., map `llama3.2` → `claude-3.5-sonnet`)
- **Custom Context Lengths** — Set per-model context window sizes
- **Streaming Support** — Full support for Server-Sent Events (SSE) streaming
- **TLS/HTTPS** — Secure connections with custom certificates
- **CORS Enabled** — Works directly with browser-based clients
- **Verbose Logging** — Debug incoming requests and upstream responses

## Installation

```bash
# Clone the repository
git clone https://github.com/poliva/aiproxy.git
cd aiproxy

# Make executable (optional)
chmod +x aiproxy.py

# Run directly
python3 aiproxy.py

# Or install system-wide
sudo cp aiproxy.py /usr/local/bin/aiproxy
```

## Quick Start

### Basic Usage

Start the proxy with the default OpenCode provider:

```bash
python3 aiproxy.py
```

The proxy listens on `http://127.0.0.1:11434` by default (Ollama's default port).

### Using with Different Providers

```bash
# OpenCode (default)
python3 aiproxy.py --provider opencode

# LM Studio (localhost)
python3 aiproxy.py --provider lmstudio --base-url http://localhost:1234/v1

# Ollama passthrough
python3 aiproxy.py --provider ollama --base-url http://localhost:11434 --port 11435

# Custom OpenAI-compatible API (eg: OpenRouter, OllamaCloud, etc...)
python3 aiproxy.py --provider custom --base-url https://api.yourprovider.com/v1 --api-key YOUR_KEY
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | Upstream provider (`opencode`, `custom`, `lmstudio`, `ollama`) | `opencode` |
| `--host` | Listen address | `127.0.0.1` |
| `--port` | Listen port | `11434` |
| `--api-key` | API key for the upstream provider | - |
| `--base-url` | Override provider base URL | - |
| `--model-mapping` | JSON map of model name remappings | `{}` |
| `--model-contexts` | JSON map of per-model context lengths | `{}` |
| `--timeout` | Upstream request timeout (seconds) | `300` |
| `--passthrough` | Pass through original body without transformation | `false` |
| `--coerce-input-to-messages` | Build chat `messages` from Responses-style `input` when `messages` is missing or empty (e.g. Cursor) | `false` |
| `--sanitize-chat-tools` | Normalize `tools` / `tool_calls` for strict upstreams (e.g. Minimax via OpenCode) | `false` |
| `--tls` | Enable HTTPS | `false` |
| `--tls-cert` | Path to TLS certificate (PEM) | - |
| `--tls-key` | Path to TLS private key (PEM) | - |
| `--verbose` / `-v` | Enable verbose logging | `false` |

## Examples

### Model Mapping

Remap client-side model names to upstream names:

```bash
python3 aiproxy.py \
  --model-mapping '{"llama3.2": "claude-3.5-sonnet", "gpt-4": "claude-3-opus"}'
```

### Custom Context Lengths

Set context window sizes for models that don't advertise them:

```bash
python3 aiproxy.py \
  --model-contexts '{"gpt-4-turbo": 128000, "claude-3-opus": 200000}'
```

### HTTPS Server

```bash
python3 aiproxy.py \
  --tls \
  --tls-cert /path/to/cert.pem \
  --tls-key /path/to/key.pem
```

### Verbose Logging

```bash
python3 aiproxy.py -v
```

This prints detailed logs of incoming requests, headers, bodies, and upstream responses.

### Cursor / strict upstream compatibility

Some clients send OpenAI **Responses**-style bodies (`input` instead of `messages`) or tool definitions that strict providers reject. Enable as needed:

```bash
python3 aiproxy.py \
  --coerce-input-to-messages \
  --sanitize-chat-tools
```

- **`--coerce-input-to-messages`** — When the request has no `messages` (or an empty list), synthesize them from `input`. Without this flag, the proxy forwards the body unchanged for providers that copy the request (e.g. OpenCode, Ollama passthrough), matching older behavior.
- **`--sanitize-chat-tools`** — Rewrites `tools` and message `tool_calls` into a conservative OpenAI chat shape (e.g. coerces `type: custom` to `function`, fills empty `parameters`). Used in **OpenCode** and **Ollama passthrough** chat transforms. Omit if your client already sends vanilla `function` tools.

## Supported Endpoints

### OpenAI-Compatible

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completion |
| `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | Embeddings |

### Ollama-Compatible

| Endpoint | Description |
|----------|-------------|
| `GET /` | Health check / status |
| `GET /tags` | List available models |
| `GET /api/tags` | List available models |
| `GET /api/ps` | List running models |
| `GET /ps` | List running models |
| `POST /api/chat` | Chat completion |
| `POST /api/generate` | Text completion |
| `POST /api/embeddings` | Embeddings |
| `POST /api/show` | Model info |
| `GET /api/version` | API version |
| `/ollama/*` | Catch-all passthrough |

## Architecture

```
                    ┌─────────────────┐
                    │   AIProxy       │
  Client ──────────►│  :11434         │──────────► OpenCode
  (OpenAI/Ollama)   │                 │──────────► LM Studio
                    │                 │──────────► Ollama
                    │                 │──────────► Custom API
                    └─────────────────┘
```

AIProxy acts as a protocol translator, accepting requests in either OpenAI or Ollama format and forwarding them to your chosen upstream provider.

## License

MIT

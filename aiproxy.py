#!/usr/bin/env python3
"""
AIProxy - single-port proxy that emulates both:
- OpenAI-compatible endpoints (/v1/*)
- Ollama-compatible endpoints (/api/*, /tags)

It forwards to a selectable upstream provider:
  opencode, custom, lmstudio, ollama (passthrough)
"""

import argparse
import hashlib
import json
import socket
import ssl
import sys
import threading
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


# Module-level cache for local IP
_local_ip = None
_local_ip_lock = threading.Lock()


def get_local_ip():
    """Get local IP address with caching to avoid repeated network calls."""
    global _local_ip
    with _local_ip_lock:
        if _local_ip is not None:
            return _local_ip
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(1.0)  # Non-blocking with timeout
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            _local_ip = ip
            return ip
        except Exception:
            return "127.0.0.1"


def _read_json_body(body_bytes: bytes | None) -> dict | None:
    if not body_bytes:
        return None
    try:
        return json.loads(body_bytes.decode("utf-8"))
    except Exception:
        return None


def _as_json_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def _send_bytes(handler: BaseHTTPRequestHandler, status: int, content_type: str, body: bytes, add_cors: bool = True):
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Cache-Control", "no-store")
    if add_cors:
        # Allow browser clients to access this proxy
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        handler.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Requested-With")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    if body:
        handler.wfile.write(body)


def _send_json(handler: BaseHTTPRequestHandler, status: int, obj):
    _send_bytes(handler, status=status, content_type="application/json", body=_as_json_bytes(obj))


def _extract_context_length_from_obj(obj: dict):
    if not isinstance(obj, dict):
        return None
    meta = obj.get("metadata", {}) if isinstance(obj.get("metadata", {}), dict) else {}
    for k in (
        "context_length",
        "context",
        "max_context_tokens",
        "max_context_length",
        "max_tokens",
        "max_output_tokens",
    ):
        # Check meta first; only fall through to obj if meta has no value (None or absent)
        meta_val = meta.get(k)
        if meta_val is not None:
            return meta_val
        obj_val = obj.get(k)
        if obj_val is not None:
            return obj_val
    return None


def _get_context_length_for_model(model_id: str, model_obj: dict | None = None):
    try:
        if isinstance(model_id, str) and model_id in (AIProxyHandler._model_contexts or {}):
            return AIProxyHandler._model_contexts.get(model_id)
    except Exception:
        pass
    if isinstance(model_obj, dict):
        val = _extract_context_length_from_obj(model_obj)
        if val is not None:
            return val
    return 192000


def transform_models_to_tags(models_response: dict) -> dict:
    models = []
    for model in models_response.get("data", []):
        model_name = model.get("id", "")
        created_time = model.get("created", 0)
        modified_at = (
            datetime.fromtimestamp(created_time, tz=timezone.utc).isoformat()
            if created_time
            else datetime.now(timezone.utc).isoformat()
        )
        context_length = _get_context_length_for_model(model_name, model)
        models.append(
            {
                "name": model_name,
                "model": model_name,
                "modified_at": modified_at,
                "size": 0,
                "digest": hashlib.sha256(model_name.encode()).hexdigest(),
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": model_name.split(":")[0] if ":" in model_name else model_name,
                    "families": [model_name.split(":")[0] if ":" in model_name else model_name],
                    "parameter_size": "120B",
                    "quantization_level": "Q4_K_M",
                    "context_length": context_length,
                },
            }
        )
    return {"models": models}


def build_show_response_from_model(model_data: dict, model_name: str) -> dict:
    model_id = model_data.get("id", model_name)
    parts = model_id.split(":")
    family = parts[0] if parts else model_id
    context_length = _get_context_length_for_model(model_id, model_data)

    model_info = {
        "general.architecture": family,
        "general.basename": model_id,
        "general.type": "model",
    }
    if context_length is not None:
        model_info["general.context_length"] = context_length
        model_info[f"{family}.context_length"] = context_length

    return {
        "license": "",
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": "120B",
            "quantization_level": "Q4_K_M",
        },
        "model_info": model_info,
        "tensors": [],
        "capabilities": ["completion", "tools", "thinking", "vision", "insert"],
        "modified_at": datetime.now(timezone.utc).isoformat(),
    }


def get_or_fetch_models_cache(provider, headers, timeout, verbose=False):
    with AIProxyHandler._cache_lock:
        if AIProxyHandler._models_cache is not None:
            if verbose:
                print("Using cached models data", file=sys.stderr)
            return AIProxyHandler._models_cache, None
        models_url = f"{provider.get_base_url()}/models"
        try:
            req = Request(url=models_url, data=None, headers=headers, method="GET")
            response = urlopen(req, timeout=timeout)
            models_text = response.read().decode("utf-8")
            models_json = json.loads(models_text)
            AIProxyHandler._models_cache = models_json
            if verbose:
                print("Fetched and cached models data", file=sys.stderr)
            return models_json, None
        except Exception as e:
            if verbose:
                print(f"Failed to fetch models: {e}", file=sys.stderr)
            return None, str(e)


def find_model_in_cache(models_data: dict, model_name: str) -> dict | None:
    """Find a model in the cache by name, using case-insensitive comparison."""
    model_name_lower = model_name.lower()
    for model in models_data.get("data", []):
        model_id = model.get("id", "")
        if isinstance(model_id, str) and model_id.lower() == model_name_lower:
            return model
    return None


class BaseProvider(ABC):
    def __init__(self, api_key=None, base_url=None, model_mapping=None, verbose=False):
        self.api_key = api_key
        self.base_url = base_url
        self.model_mapping = model_mapping or {}
        self.verbose = verbose

    @abstractmethod
    def get_base_url(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_headers(self, body=None, upstream_auth: str | None = None) -> dict:
        raise NotImplementedError

    @abstractmethod
    def transform_chat_request(self, body: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def transform_generate_request(self, body: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def transform_embeddings_request(self, body: dict) -> dict:
        raise NotImplementedError

    @abstractmethod
    def map_model_name(self, model: str) -> str:
        raise NotImplementedError

    def transform_response(self, response: dict, endpoint: str) -> dict:
        return response


class OpenCodeProvider(BaseProvider):
    def get_base_url(self) -> str:
        return self.base_url or "https://opencode.ai/zen/v1"

    def get_headers(self, body=None, upstream_auth: str | None = None) -> dict:
        auth = self.api_key or upstream_auth or "public"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth}",
            "x-opencode-client": "cli",
            "x-opencode-session": str(uuid.uuid4()),
            "User-Agent": "opencode/local/local/cli",
            "HTTP-Referer": "https://opencode.ai/",
            "X-Title": "opencode",
        }
        real_ip = get_local_ip()
        if real_ip:
            headers["x-real-ip"] = real_ip
        return headers

    def map_model_name(self, model: str) -> str:
        return self.model_mapping.get(model, model)

    def transform_chat_request(self, body: dict) -> dict:
        transformed = dict(body)
        transformed["model"] = self.map_model_name(body.get("model", ""))
        return transformed

    def transform_generate_request(self, body: dict) -> dict:
        transformed = dict(body)
        transformed["model"] = self.map_model_name(body.get("model", ""))
        return transformed

    def transform_embeddings_request(self, body: dict) -> dict:
        return {
            "model": self.map_model_name(body.get("model", "nomic-embed-text")),
            "input": body.get("input", body.get("prompt", "")),
        }


class CustomProvider(BaseProvider):
    def get_base_url(self) -> str:
        return self.base_url

    def get_headers(self, body=None, upstream_auth: str | None = None) -> dict:
        auth = self.api_key or upstream_auth
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "aiproxy/local/local/cli",
            "HTTP-Referer": "https://aiproxy.local/",
            "X-Title": "AIProxy",
        }
        if auth:
            headers["Authorization"] = f"Bearer {auth}"
        return headers

    def map_model_name(self, model: str) -> str:
        return self.model_mapping.get(model, model)

    def _extract_options(self, body: dict) -> dict:
        """Extract common options from request body."""
        extracted = {}
        if "options" in body and isinstance(body.get("options"), dict):
            options = body["options"]
            for key in ("temperature", "top_p", "max_tokens", "stop", "stop_seq"):
                if key in options:
                    extracted[key] = options[key]
        return extracted

    def transform_chat_request(self, body: dict) -> dict:
        messages = []
        for msg in body.get("messages", []):
            role = msg.get("role")
            if not role:
                # Skip messages without a role — silently drop to avoid upstream rejection
                print("Warning: skipping message with missing role", file=sys.stderr)
                continue

            content = msg.get("content", "")
            if isinstance(content, list):
                formatted_content = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            formatted_content.append({"type": "text", "text": part.get("text", "")})
                        elif part.get("type") == "image_url":
                            formatted_content.append({"type": "image_url", "image_url": part.get("image_url", {})})
                        else:
                            # Convert unknown content types to a text placeholder so nothing is silently dropped
                            part_type = part.get("type", "unknown")
                            print(f"Warning: converting unknown content type '{part_type}' to text", file=sys.stderr)
                            formatted_content.append({"type": "text", "text": f"[{part_type} content]"})
                content = formatted_content

            # Build the new message preserving all standard fields
            new_msg = {"role": role, "content": content}
            # Preserve name (function name for function messages, or explicit name)
            if msg.get("name"):
                new_msg["name"] = msg["name"]
            # Preserve tool_calls (assistant messages requesting tool use)
            if msg.get("tool_calls"):
                new_msg["tool_calls"] = msg["tool_calls"]
            # Preserve tool_call_id (tool role messages)
            if msg.get("tool_call_id"):
                new_msg["tool_call_id"] = msg["tool_call_id"]
            # Preserve function_call (legacy format)
            if msg.get("function_call"):
                new_msg["function_call"] = msg["function_call"]
            messages.append(new_msg)

        transformed = {
            "model": self.map_model_name(body.get("model", "")),
            "messages": messages,
            "stream": body.get("stream", False),
        }

        # Use helper to extract common options
        extracted = self._extract_options(body)
        for key in ("temperature", "top_p", "max_tokens"):
            if key in extracted:
                transformed[key] = extracted[key]

        return transformed

    def transform_generate_request(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            for part in prompt:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt = part.get("text", "")
                    break
            else:
                # No text part found - join all string parts or stringify
                text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in prompt]
                prompt = " ".join(filter(None, text_parts)) if text_parts else ""

        transformed = {
            "model": self.map_model_name(body.get("model", "")),
            "messages": [{"role": "user", "content": prompt}],
            "stream": body.get("stream", False),
        }

        # Use helper to extract common options
        extracted = self._extract_options(body)
        for key in ("temperature", "top_p", "max_tokens", "stop"):
            if key in extracted:
                transformed[key] = extracted[key]
        if "stop_seq" in extracted:
            stop_seq = extracted["stop_seq"]
            if "stop" in transformed:
                existing = transformed["stop"] if isinstance(transformed["stop"], list) else [transformed["stop"]]
                existing.append(stop_seq)
                transformed["stop"] = existing
            else:
                transformed["stop"] = stop_seq

        return transformed

    def transform_embeddings_request(self, body: dict) -> dict:
        return {
            "model": self.map_model_name(body.get("model", "nomic-embed-text")),
            "input": body.get("input", body.get("prompt", "")),
        }


class LMStudioProvider(BaseProvider):
    def get_base_url(self) -> str:
        return self.base_url or "http://localhost:1234/v1"

    def get_headers(self, body=None, upstream_auth: str | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        auth = self.api_key or upstream_auth
        if auth:
            headers["Authorization"] = f"Bearer {auth}"
        return headers

    def map_model_name(self, model: str) -> str:
        return self.model_mapping.get(model, model)

    def transform_chat_request(self, body: dict) -> dict:
        transformed = {
            "model": self.map_model_name(body.get("model", "")),
            "messages": body.get("messages", []),
            "stream": body.get("stream", False),
        }
        if "options" in body and isinstance(body.get("options"), dict):
            transformed.update(body["options"])
        return transformed

    def transform_generate_request(self, body: dict) -> dict:
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            for part in prompt:
                if isinstance(part, dict) and part.get("type") == "text":
                    prompt = part.get("text", "")
                    break
            else:
                # No text part found - join all string parts or stringify
                text_parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in prompt]
                prompt = " ".join(filter(None, text_parts)) if text_parts else ""
        transformed = {
            "model": self.map_model_name(body.get("model", "")),
            "prompt": prompt,
            "stream": body.get("stream", False),
        }
        if "options" in body and isinstance(body.get("options"), dict):
            transformed.update(body["options"])
        return transformed

    def transform_embeddings_request(self, body: dict) -> dict:
        return {
            "model": self.map_model_name(body.get("model", "nomic-embed-text")),
            "input": body.get("input", body.get("prompt", "")),
        }


class OllamaPassthroughProvider(BaseProvider):
    def get_base_url(self) -> str:
        return self.base_url or "http://localhost:11434"

    def get_headers(self, body=None, upstream_auth: str | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        auth = self.api_key or upstream_auth
        if auth:
            headers["Authorization"] = f"Bearer {auth}"
        return headers

    def map_model_name(self, model: str) -> str:
        return self.model_mapping.get(model, model)

    def transform_chat_request(self, body: dict) -> dict:
        transformed = dict(body)
        transformed["model"] = self.map_model_name(body.get("model", ""))
        return transformed

    def transform_generate_request(self, body: dict) -> dict:
        transformed = dict(body)
        transformed["model"] = self.map_model_name(body.get("model", ""))
        return transformed

    def transform_embeddings_request(self, body: dict) -> dict:
        transformed = dict(body)
        transformed["model"] = self.map_model_name(body.get("model", ""))
        return transformed


PROVIDER_CLASSES = {
    "opencode": OpenCodeProvider,
    "custom": CustomProvider,
    "lmstudio": LMStudioProvider,
    "ollama": OllamaPassthroughProvider,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="AIProxy - single-port OpenAI+Ollama compatible proxy"
    )
    parser.add_argument(
        "--provider",
        choices=["opencode", "custom", "lmstudio", "ollama"],
        default="opencode",
        help="Provider to proxy requests to (default: opencode)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Listen host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=11434, help="Listen port (default: 11434)")
    parser.add_argument("--api-key", help="API key for the provider")
    parser.add_argument("--base-url", help="Override provider base URL")
    parser.add_argument(
        "--model-mapping",
        type=json.loads,
        default={},
        help='Model name mappings as JSON, e.g. \'{"llama3.2": "claude-3.5-sonnet"}\'',
    )
    parser.add_argument(
        "--model-contexts",
        type=json.loads,
        default={},
        help='Per-model context lengths as JSON, e.g. \'{"gpt-5.2": 272000}\'',
    )
    parser.add_argument("--timeout", type=int, default=300, help="Upstream timeout seconds (default: 300)")
    parser.add_argument("--passthrough", action="store_true", help="Pass through original body without transformation")

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging: show incoming/outgoing headers, body, upstream response, and full response on stderr",
    )

    parser.add_argument("--tls", action="store_true", help="Serve HTTPS (TLS) instead of plain HTTP")
    parser.add_argument("--tls-cert", help="Path to TLS certificate (PEM). Required if --tls")
    parser.add_argument("--tls-key", help="Path to TLS private key (PEM). Required if --tls")
    return parser.parse_args()


class AIProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    args = None
    provider = None
    _models_cache = None
    _model_contexts = {}
    _cache_lock = threading.Lock()

    def log_message(self, format, *args):
        if self.args and self.args.verbose:
            command = getattr(self, "command", "-")
            path = getattr(self, "path", "-")
            try:
                msg = (format % args).rstrip("\n")
            except Exception:
                msg = str(format).rstrip("\n")
            sys.stderr.write(f"{command} {path} - {msg}\n")

    def do_GET(self):
        self.handle_request("GET")

    def do_POST(self):
        self.handle_request("POST")

    def do_DELETE(self):
        self.handle_request("DELETE")

    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Requested-With")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def handle_request(self, method: str):
        args = self.args
        provider = self.provider

        # Handle invalid Content-Length gracefully
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            _send_json(self, 400, {"error": {"message": "Invalid Content-Length header", "type": "invalid_request"}})
            return

        MAX_BODY_SIZE = 10 * 1024 * 1024  # 10MB limit
        if content_length < 0:
            _send_json(self, 400, {"error": {"message": f"Invalid Content-Length: {content_length}", "type": "invalid_request"}})
            return
        if content_length > MAX_BODY_SIZE:
            _send_json(self, 413, {"error": {"message": f"Request body too large: {content_length} bytes (max: {MAX_BODY_SIZE})", "type": "request_too_large"}})
            return
        body = self.rfile.read(content_length) if content_length > 0 else None

        parsed_url = urlparse(self.path)
        path = parsed_url.path
        query = parse_qs(parsed_url.query)

        # Warn if query parameters are present
        if query:
            print(f"Warning: query parameters present but ignored: {query}", file=sys.stderr)

        if args.verbose:
            print(f"\n=== Incoming {method} Request ===", file=sys.stderr)
            print(f"Path: {path}", file=sys.stderr)
            print("Client Headers:", file=sys.stderr)
            for header, value in self.headers.items():
                print(f"  {header}: {value}", file=sys.stderr)

        target_url, transformed_body, endpoint_type = self.prepare_request(
            method=method, path=path, query=query, body=body, args=args, provider=provider
        )
        if target_url is None:
            return

        raw_auth = self.headers.get("Authorization", "")
        # Handle both "Bearer <token>" and bare "Bearer" (no token)
        if raw_auth.lower().startswith("bearer "):
            upstream_auth = raw_auth[7:].strip()
        elif raw_auth.lower() == "bearer":
            upstream_auth = ""
            print("Warning: received empty Bearer token", file=sys.stderr)
        else:
            upstream_auth = raw_auth
        headers = provider.get_headers(transformed_body, upstream_auth=upstream_auth)

        # Forward X-Request-ID if present in client request
        client_request_id = self.headers.get("X-Request-ID")
        if client_request_id:
            headers["X-Request-ID"] = client_request_id

        if args.verbose:
            print("\n=== Headers to Upstream ===", file=sys.stderr)
            for k, v in headers.items():
                print(f"  {k}: {v}", file=sys.stderr)

            if transformed_body is not None:
                print("\n=== Upstream Request Body ===", file=sys.stderr)
                try:
                    print(json.dumps(json.loads(transformed_body.decode("utf-8")), indent=2), file=sys.stderr)
                except Exception:
                    print(transformed_body.decode("utf-8", errors="replace"), file=sys.stderr)

        stream_requested = False
        try:
            incoming_json = _read_json_body(body)
            stream_requested = bool(incoming_json.get("stream", False)) if isinstance(incoming_json, dict) else False
        except Exception:
            stream_requested = False

        try:
            print(f"Proxying to: {target_url}", file=sys.stderr)

            req = Request(url=target_url, data=transformed_body, headers=headers, method=method)
            try:
                response = urlopen(req, timeout=args.timeout)
            except socket.timeout:
                print(f"\n=== Upstream Timeout ===", file=sys.stderr)
                _send_json(
                    self,
                    504,
                    {"error": {"message": f"Upstream request timed out after {args.timeout}s", "type": "upstream_timeout"}},
                )
                return

            # Detect if upstream is streaming (check Transfer-Encoding or Content-Type)
            upstream_transfer_encoding = response.headers.get("Transfer-Encoding", "").lower()
            upstream_content_type = response.headers.get("Content-Type", "").lower()
            upstream_streaming = "chunked" in upstream_transfer_encoding or "text/event-stream" in upstream_content_type or "stream" in upstream_content_type

            # If client didn't request streaming but upstream is streaming, buffer and parse SSE
            if upstream_streaming and not stream_requested:
                print("Upstream returned streaming response but client requested non-streaming; buffering and parsing SSE", file=sys.stderr)
                raw_data = response.read().decode("utf-8", errors="replace")
                # Parse SSE: extract the last valid JSON payload
                resp_text = raw_data  # fallback to raw
                for line in raw_data.split("\n"):
                    line = line.strip()
                    if line.startswith("data: "):
                        data = line[6:]  # strip "data: "
                        if data == "[DONE]":
                            break
                        try:
                            json.loads(data)  # Validate it's JSON
                            resp_text = data  # Use the last valid JSON
                        except json.JSONDecodeError:
                            continue
            elif stream_requested:
                upstream_ct = None
                try:
                    upstream_ct = response.headers.get("Content-Type")
                except Exception:
                    upstream_ct = None

                self.send_response(200)
                self.send_header("Content-Type", upstream_ct or "application/octet-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-Requested-With")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                try:
                    if hasattr(response, "read1"):
                        read_fn = response.read1
                    elif hasattr(getattr(response, "fp", None), "read1"):
                        read_fn = response.fp.read1
                    else:
                        read_fn = response.read

                    # Set socket timeout for streaming to avoid indefinite blocking
                    try:
                        if hasattr(response, "fp") and hasattr(response.fp, "sock") and response.fp.sock:
                            response.fp.sock.settimeout(args.timeout)
                    except Exception:
                        pass  # Ignore timeout setting failures

                    while True:
                        try:
                            chunk = read_fn(8192)
                        except socket.timeout:
                            print("Streaming read timed out", file=sys.stderr)
                            break
                        if not chunk:
                            break
                        self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii"))  # lowercase hex per RFC 9112
                        self.wfile.write(chunk)
                        self.wfile.write(b"\r\n")
                        try:
                            self.wfile.flush()
                        except Exception:
                            pass
                    self.wfile.write(b"0\r\n\r\n")
                    try:
                        self.wfile.flush()
                    except Exception:
                        pass
                except (BrokenPipeError, ConnectionResetError):
                    return
                return
            else:
                resp_text = response.read().decode("utf-8", errors="replace")

            if args.verbose:
                print("\n=== Original Upstream Response ===", file=sys.stderr)
                print(resp_text, file=sys.stderr)

            if endpoint_type == "tags":
                try:
                    resp_json = json.loads(resp_text)
                    # Check if already in Ollama format, otherwise transform from OpenAI format
                    is_ollama_format = (
                        "models" in resp_json and
                        resp_json.get("models") and
                        isinstance(resp_json["models"], list) and
                        resp_json["models"] and
                        "details" in (resp_json["models"][0] or {})
                    )
                    if not is_ollama_format and "data" in resp_json:
                        print("Tags response is OpenAI format; transforming to Ollama format", file=sys.stderr)
                        resp_text = json.dumps(transform_models_to_tags(resp_json))
                    # else: already in Ollama format, keep resp_text as-is
                except json.JSONDecodeError as e:
                    print(f"Warning: upstream returned invalid JSON for tags endpoint: {e}", file=sys.stderr)
                except Exception as e:
                    print(f"Warning: error processing tags response: {e}", file=sys.stderr)

            # Allow the provider to transform the response (e.g., adapting formats)
            try:
                resp_json = json.loads(resp_text) if resp_text else {}
                resp_json = provider.transform_response(resp_json, endpoint_type)
                resp_text = json.dumps(resp_json)
            except Exception as e:
                print(f"Warning: error in transform_response: {e}", file=sys.stderr)

            if args.verbose:
                print("\n=== Transformed Upstream Response ===", file=sys.stderr)
                print(resp_text, file=sys.stderr)

            _send_bytes(self, status=200, content_type="application/json", body=resp_text.encode("utf-8"))

        except HTTPError as e:
            # Ensure error_body is never empty
            raw_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            error_body = raw_body if raw_body.strip() else str(e)

            if endpoint_type == "tags" and e.code >= 400:
                print(f"/api/tags failed ({e.code}), trying /v1/models fallback", file=sys.stderr)
                try:
                    models_url = f"{provider.get_base_url()}/models"
                    models_req = Request(url=models_url, data=None, headers=headers, method="GET")
                    models_response = urlopen(models_req, timeout=args.timeout)
                    models_text = models_response.read().decode("utf-8", errors="replace")
                    models_json = json.loads(models_text)
                    tags_text = json.dumps(transform_models_to_tags(models_json))
                    _send_bytes(self, status=200, content_type="application/json", body=tags_text.encode("utf-8"))
                    return
                except Exception as fallback_error:
                    print(f"Fallback to /v1/models failed: {fallback_error}", file=sys.stderr)

            if endpoint_type == "show" and e.code >= 400:
                print(f"/api/show failed ({e.code}), building from models cache", file=sys.stderr)

                models_data, err = get_or_fetch_models_cache(provider, headers, args.timeout, args.verbose)
                if models_data:
                    body_json = _read_json_body(body) or {}
                    model_name = body_json.get("model", "")
                    # Use case-insensitive model lookup
                    model_data = find_model_in_cache(models_data, model_name)
                    if not model_data:
                        mapped_name = provider.map_model_name(model_name)
                        if mapped_name != model_name:
                            model_data = find_model_in_cache(models_data, mapped_name)
                    if model_data:
                        show_text = json.dumps(build_show_response_from_model(model_data, model_name))
                        _send_bytes(self, status=200, content_type="application/json", body=show_text.encode("utf-8"))
                        return

                    _send_json(self, 404, {"error": {"message": f"model '{model_name}' not found", "type": "not_found"}})
                    return
                if err:
                    print(f"Could not build show response from cache: {err}", file=sys.stderr)

            if args.verbose:
                print(f"\n=== Proxy Error (HTTP {e.code}) ===", file=sys.stderr)
                print(error_body, file=sys.stderr)

            # Forward upstream JSON error directly instead of double-wrapping
            try:
                error_json = json.loads(error_body)
                _send_json(self, e.code, error_json)
            except (json.JSONDecodeError, TypeError):
                _send_json(self, e.code, {"error": {"message": error_body, "type": "proxy_error"}})

        except URLError as e:
            error_msg = str(e)

            if endpoint_type == "tags":
                print("/api/tags failed (URLError), trying /v1/models fallback", file=sys.stderr)
                try:
                    models_url = f"{provider.get_base_url()}/models"
                    models_req = Request(url=models_url, data=None, headers=headers, method="GET")
                    models_response = urlopen(models_req, timeout=args.timeout)
                    models_text = models_response.read().decode("utf-8", errors="replace")
                    models_json = json.loads(models_text)
                    tags_text = json.dumps(transform_models_to_tags(models_json))
                    _send_bytes(self, status=200, content_type="application/json", body=tags_text.encode("utf-8"))
                    return
                except Exception as fallback_error:
                    print(f"Fallback to /v1/models failed: {fallback_error}", file=sys.stderr)

            if endpoint_type == "show":
                print("/api/show failed (URLError), building from models cache", file=sys.stderr)
                models_data, err = get_or_fetch_models_cache(provider, headers, args.timeout, args.verbose)
                if models_data:
                    body_json = _read_json_body(body) or {}
                    model_name = body_json.get("model", "")
                    # Use case-insensitive model lookup
                    model_data = find_model_in_cache(models_data, model_name)
                    if not model_data:
                        mapped_name = provider.map_model_name(model_name)
                        if mapped_name != model_name:
                            model_data = find_model_in_cache(models_data, mapped_name)
                    if model_data:
                        show_text = json.dumps(build_show_response_from_model(model_data, model_name))
                        _send_bytes(self, status=200, content_type="application/json", body=show_text.encode("utf-8"))
                        return

            if args.verbose:
                print("\n=== Proxy Error ===", file=sys.stderr)
                print(error_msg, file=sys.stderr)
            _send_json(self, 502, {"error": {"message": f"Proxy error: {error_msg}", "type": "proxy_error"}})

    def prepare_request(self, method, path, query, body, args, provider):
        provider_base = provider.get_base_url().rstrip("/")

        if path == "/" and method == "GET":
            _send_json(
                self,
                200,
                {
                    "status": "ok",
                    "provider": args.provider,
                    "message": "aiproxy is running",
                    "endpoints": {"openai": "/v1/*", "ollama": "/api/* and /tags"},
                },
            )
            return None, None, None

        # OpenAI-compatible routes (with and without /v1 prefix depending on client config).
        if path.startswith("/v1/") or path in (
            "/chat/completions",
            "/completions",
            "/embeddings",
            "/models",
        ):
            # Normalize (strip optional /v1 prefix)
            norm = path[3:] if path.startswith("/v1/") else path
            transformed_body = body
            transformed_dict = None
            if not args.passthrough and body:
                body_json = _read_json_body(body)
                if isinstance(body_json, dict):
                    if norm == "/chat/completions":
                        transformed_dict = provider.transform_chat_request(body_json)
                    elif norm == "/completions":
                        transformed_dict = provider.transform_generate_request(body_json)
                    elif norm == "/embeddings":
                        transformed_dict = provider.transform_embeddings_request(body_json)
            if transformed_dict is not None:
                try:
                    transformed_body = json.dumps(transformed_dict).encode("utf-8")
                except (TypeError, ValueError) as e:
                    _send_json(self, 500, {"error": {"message": f"Failed to serialize request: {e}", "type": "internal_error"}})
                    return None, None, None
            if norm == "/chat/completions":
                return f"{provider_base}/chat/completions", transformed_body, "openai"
            if norm == "/completions":
                return f"{provider_base}/completions", transformed_body, "openai"
            if norm == "/embeddings":
                return f"{provider_base}/embeddings", transformed_body, "openai"
            if norm == "/models":
                return f"{provider_base}/models", transformed_body, "openai"
            return f"{provider_base}{path}", transformed_body, "openai"

        if path == "/tags" or path == "/api/tags":
            if isinstance(provider, OllamaPassthroughProvider):
                return f"{provider_base}/api/tags", body, "tags"
            return f"{provider_base}/models", None, "tags"

        if path == "/ps" or path == "/api/ps":
            _send_json(self, 200, {"status": "ok", "models": []})
            return None, None, None

        if path == "/version" or path == "/api/version":
            _send_bytes(self, status=200, content_type="application/json", body=b'{"version":"0.20.4"}')
            return None, None, None

        if path == "/api/show":
            body_json = _read_json_body(body) or {}
            model_name = body_json.get("model", "")

            # OpenAI-style providers typically do NOT implement Ollama's /api/show.
            # Serve it locally by building from the upstream /models list.
            if not isinstance(provider, OllamaPassthroughProvider):
                models_data, err = get_or_fetch_models_cache(provider, provider.get_headers(None), args.timeout, args.verbose)
                if models_data:
                    model_data = find_model_in_cache(models_data, model_name)
                    if not model_data:
                        mapped_name = provider.map_model_name(model_name)
                        if mapped_name != model_name:
                            model_data = find_model_in_cache(models_data, mapped_name)
                    if model_data:
                        show_text = json.dumps(build_show_response_from_model(model_data, model_name))
                        _send_bytes(self, status=200, content_type="application/json", body=show_text.encode("utf-8"))
                        return None, None, None
                    _send_json(self, 404, {"error": {"message": f"model '{model_name}' not found", "type": "not_found"}})
                    return None, None, None
                _send_json(self, 502, {"error": {"message": f"Could not fetch models for /api/show: {err or 'unknown error'}", "type": "proxy_error"}})
                return None, None, None

            return f"{provider_base}/api/show", body, "show"

        if path == "/api/chat":
            target_url = f"{provider_base}/chat/completions"
            if body and not args.passthrough:
                body_json = _read_json_body(body)
                if isinstance(body_json, dict):
                    transformed = provider.transform_chat_request(body_json)
                    try:
                        return target_url, json.dumps(transformed).encode("utf-8"), "ollama"
                    except (TypeError, ValueError) as e:
                        _send_json(self, 500, {"error": {"message": f"Failed to serialize request: {e}", "type": "internal_error"}})
                        return None, None, None
            return target_url, body, "ollama"

        if path == "/api/generate":
            target_url = f"{provider_base}/completions"
            if body and not args.passthrough:
                body_json = _read_json_body(body)
                if isinstance(body_json, dict):
                    transformed = provider.transform_generate_request(body_json)
                    try:
                        return target_url, json.dumps(transformed).encode("utf-8"), "ollama"
                    except (TypeError, ValueError) as e:
                        _send_json(self, 500, {"error": {"message": f"Failed to serialize request: {e}", "type": "internal_error"}})
                        return None, None, None
            return target_url, body, "ollama"

        if path == "/api/embeddings":
            target_url = f"{provider_base}/embeddings"
            if body and not args.passthrough:
                body_json = _read_json_body(body)
                if isinstance(body_json, dict):
                    transformed = provider.transform_embeddings_request(body_json)
                    try:
                        return target_url, json.dumps(transformed).encode("utf-8"), "ollama"
                    except (TypeError, ValueError) as e:
                        _send_json(self, 500, {"error": {"message": f"Failed to serialize request: {e}", "type": "internal_error"}})
                        return None, None, None
            return target_url, body, "ollama"

        if path.startswith("/api/") or path.startswith("/ollama/"):
            return f"{provider_base}{path}", body, "ollama"

        # Unknown endpoint: be strict and return a proper non-200.
        _send_json(
            self,
            404,
            {
                "error": {
                    "message": f"Unexpected endpoint or method. ({method} {path})",
                    "type": "not_found",
                }
            },
        )
        return None, None, None


def main():
    args = parse_args()

    if args.provider == "custom" and not args.base_url:
        raise SystemExit("--base-url is required when using --provider custom")

    provider_class = PROVIDER_CLASSES[args.provider]
    provider = provider_class(
        api_key=args.api_key,
        base_url=args.base_url,
        model_mapping=args.model_mapping,
        verbose=args.verbose,
    )

    AIProxyHandler.args = args
    AIProxyHandler.provider = provider
    AIProxyHandler._model_contexts = args.model_contexts or {}

    server_address = (args.host, args.port)
    httpd = HTTPServer(server_address, AIProxyHandler)

    if args.tls:
        if not args.tls_cert or not args.tls_key:
            raise SystemExit("--tls requires --tls-cert and --tls-key")
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=args.tls_cert, keyfile=args.tls_key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    scheme = "https" if args.tls else "http"
    print(f"AIProxy listening on {scheme}://{args.host}:{args.port}", file=sys.stderr)
    print(f"Provider: {args.provider}", file=sys.stderr)
    print(f"Upstream base URL: {provider.get_base_url()}", file=sys.stderr)
    print("Serves: OpenAI (/v1/*) and Ollama (/api/*, /tags)", file=sys.stderr)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", file=sys.stderr)
        httpd.shutdown()


if __name__ == "__main__":
    main()
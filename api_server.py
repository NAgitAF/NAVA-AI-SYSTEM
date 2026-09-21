"""Dependency-free local REST gateway for NAVA."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict
from urllib.parse import urlsplit

from gateway_system import GatewayService, auth_from_environment


class NAVARequestHandler(BaseHTTPRequestHandler):
    runtime_handler: Callable[[str], Dict[str, Any]] | None = None
    max_body_bytes = 256 * 1024
    gateway = GatewayService()

    def _write_json(self, status: int, payload: Dict[str, Any], www_authenticate: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Request-ID", self.headers.get("X-Request-ID", ""))
        if www_authenticate:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlsplit(self.path).path == "/health":
            self._write_json(200, {"status": "ok", "service": "nava"})
            return
        self._write_json(404, {"error": "route_not_found"})

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/v1/chat":
            self._write_json(404, {"error": "route_not_found"})
            return
        try:
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("application/json"):
                self._write_json(415, {"error": "content_type_must_be_json"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > self.max_body_bytes:
                self._write_json(413, {"error": "request_body_too_large_or_empty"})
                return
            try:
                request = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._write_json(400, {"error": "invalid_json"})
                return
            if not isinstance(request, dict):
                self._write_json(400, {"error": "request_body_must_be_object"})
                return
            if request.get("token") is None:
                authorization = self.headers.get("Authorization", "")
                scheme, _, token = authorization.partition(" ")
                request["token"] = token.strip() if scheme.lower() == "bearer" else ""
            accepted = self.gateway.handle_request(request)
            if self.runtime_handler is None:
                raise RuntimeError("NAVA runtime handler is not configured.")
            result = self.runtime_handler(accepted["input"])
            self._write_json(200, {"status": "ok", "session": accepted["session"], "result": result})
        except (ValueError, json.JSONDecodeError):
            self._write_json(400, {"error": "invalid_request"})
        except PermissionError as exc:
            status = 401 if str(exc) == "Request rejected by auth policy" else 429
            self._write_json(status, {"error": str(exc)}, www_authenticate=status == 401)
        except Exception:
            self._write_json(500, {"error": "internal_server_error"})

    def log_message(self, format: str, *args: Any) -> None:
        return


def create_server(runtime_handler: Callable[[str], Dict[str, Any]], host: str = "127.0.0.1", port: int = 8765):
    NAVARequestHandler.runtime_handler = runtime_handler
    try:
        NAVARequestHandler.gateway = GatewayService(auth_provider=auth_from_environment())
    except RuntimeError:
        if host not in {"127.0.0.1", "::1", "localhost"} or os.environ.get("NAVA_ALLOW_ANONYMOUS_LOCAL") != "1":
            raise
        NAVARequestHandler.gateway = GatewayService()
    return ThreadingHTTPServer((host, port), NAVARequestHandler)


if __name__ == "__main__":
    server = create_server(lambda prompt: {"response": prompt})
    print("NAVA API listening on http://127.0.0.1:8765")
    server.serve_forever()

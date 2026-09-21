"""Gateway System — session management, auth, request normalization and routing."""

import hashlib
import hmac
import base64
import json
import os
import time
from typing import Any


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


class SessionManager:
    def __init__(self):
        self.sessions = {}

    def get_or_create(self, session_id):
        return self.sessions.setdefault(session_id, {"session_id": session_id, "created_at": time.time(), "requests": 0})


class RateLimiter:
    def __init__(self, max_requests=30, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def allow(self, client_id):
        now = time.time()
        recent = [stamp for stamp in self.requests.get(client_id, []) if now - stamp < self.window_seconds]
        if len(recent) >= self.max_requests:
            self.requests[client_id] = recent
            return False
        recent.append(now)
        self.requests[client_id] = recent
        return True


class TokenAuth:
    def __init__(self, tokens=None):
        self.tokens = set(tokens or [])

    def is_allowed(self, request):
        token = str(request.get("token", ""))
        return not self.tokens or any(hmac.compare_digest(token, expected) for expected in self.tokens)


class JWTAuth:
    """Minimal HS256 JWT validator with issuer, audience and expiry checks."""

    def __init__(self, secret: str, issuer: str | None = None, audience: str | None = None):
        if len(secret.encode("utf-8")) < 32:
            raise ValueError("NAVA_JWT_SECRET must contain at least 32 bytes.")
        self.secret = secret.encode("utf-8")
        self.issuer = issuer
        self.audience = audience

    def issue(self, subject: str, claims: dict[str, Any] | None = None, ttl_seconds: int = 3600) -> str:
        now = int(time.time())
        payload: dict[str, Any] = {"sub": subject, "iat": now, "exp": now + ttl_seconds}
        if self.issuer:
            payload["iss"] = self.issuer
        if self.audience:
            payload["aud"] = self.audience
        payload.update(claims or {})
        header = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
        body = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{header}.{body}".encode("ascii")
        signature = hmac.new(self.secret, signing_input, hashlib.sha256).digest()
        return f"{header}.{body}.{_b64url_encode(signature)}"

    def is_allowed(self, request: dict[str, Any]) -> bool:
        token = str(request.get("token", ""))
        parts = token.split(".")
        if len(parts) != 3:
            return False
        header_part, payload_part, signature_part = parts
        signing_input = f"{header_part}.{payload_part}".encode("ascii")
        expected = hmac.new(self.secret, signing_input, hashlib.sha256).digest()
        try:
            decoded_signature = _b64url_decode(signature_part)
            if _b64url_encode(decoded_signature) != signature_part:
                return False
            if not hmac.compare_digest(decoded_signature, expected):
                return False
            header = json.loads(_b64url_decode(header_part))
            payload = json.loads(_b64url_decode(payload_part))
        except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        now = time.time()
        if header.get("alg") != "HS256" or header.get("typ") != "JWT":
            return False
        if not isinstance(payload.get("sub"), str) or not payload.get("sub"):
            return False
        if not isinstance(payload.get("exp"), (int, float)) or now >= payload["exp"]:
            return False
        if "nbf" in payload and (not isinstance(payload["nbf"], (int, float)) or now < payload["nbf"]):
            return False
        if self.issuer and payload.get("iss") != self.issuer:
            return False
        if self.audience:
            audience = payload.get("aud")
            if audience != self.audience and not (isinstance(audience, list) and self.audience in audience):
                return False
        return True


def auth_from_environment() -> TokenAuth | JWTAuth:
    secret = os.environ.get("NAVA_JWT_SECRET", "").strip()
    if secret:
        return JWTAuth(
            secret,
            issuer=os.environ.get("NAVA_JWT_ISSUER") or None,
            audience=os.environ.get("NAVA_JWT_AUDIENCE") or None,
        )
    token = os.environ.get("NAVA_API_TOKEN", "").strip()
    if token:
        return TokenAuth([token])
    raise RuntimeError("NAVA_JWT_SECRET or NAVA_API_TOKEN must be configured.")


class GatewayService:
    """Entry point that validates inbound traffic and dispatches requests."""

    def __init__(self, session_manager=None, auth_provider=None, rate_limiter=None):
        self.session_manager = session_manager or SessionManager()
        self.auth_provider = auth_provider or TokenAuth()
        self.rate_limiter = rate_limiter or RateLimiter()

    def handle_request(self, request):
        """Normalized request pipeline for the runtime."""
        if not isinstance(request, dict):
            raise ValueError("Request body must be a JSON object.")
        if self.auth_provider is not None and not self.auth_provider.is_allowed(request):
            raise PermissionError("Request rejected by auth policy")
        client_id = str(request.get("client_id", "anonymous"))
        if not self.rate_limiter.allow(client_id):
            raise PermissionError("Rate limit exceeded")
        session_id = str(request.get("session_id", hashlib.sha256(client_id.encode()).hexdigest()[:16]))
        session = self.session_manager.get_or_create(session_id)
        session["requests"] += 1
        user_input = str(request.get("input", "")).strip()
        if not user_input:
            raise ValueError("Input is required.")
        if len(user_input) > 12000:
            raise ValueError("Input exceeds the maximum allowed length.")
        return {"status": "accepted", "session": session, "input": user_input}

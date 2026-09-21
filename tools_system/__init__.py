"""Tools System — tool registry, permissions and sandboxed execution."""

from __future__ import annotations

import os
import re
import ast
import operator
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from safety_system import PolicyEngine

try:
    import resource
except ImportError:  # pragma: no cover - Windows does not expose resource
    resource = None


class PermissionEngine:
    """Central policy engine for safe tool execution."""

    def __init__(self):
        self.permissions: Dict[str, Dict[str, Any]] = {}

    def allow(self, tool_name: str, allowed: bool = True, *, scope: str = "restricted", max_runtime_seconds: int = 30, network_access: bool = False):
        self.permissions[tool_name] = {
            "allowed": bool(allowed),
            "scope": scope,
            "max_runtime_seconds": int(max_runtime_seconds),
            "network_access": bool(network_access),
        }
        return self.permissions[tool_name]

    def deny(self, tool_name: str):
        return self.allow(tool_name, False, scope="restricted")

    def check(self, tool_name: str, *, scope: Optional[str] = None, network_access: bool = False) -> bool:
        policy = self.permissions.get(tool_name, {"allowed": False, "scope": "restricted", "max_runtime_seconds": 30, "network_access": False})
        if not policy.get("allowed", False):
            return False
        if scope is not None and policy.get("scope") not in {"all", scope}:
            return False
        if network_access and not policy.get("network_access", False):
            return False
        return True

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.permissions)


class ToolSandbox:
    """A minimal sandbox that restricts file scopes and execution time."""

    def __init__(
        self,
        allowed_roots: Optional[List[str]] = None,
        *,
        allow_network: bool = False,
        max_runtime_seconds: int = 30,
        max_memory_mb: int = 256,
        max_output_bytes: int = 64 * 1024,
        allowed_commands: Optional[List[str]] = None,
    ):
        base = allowed_roots or [os.getcwd()]
        self.allowed_roots = [os.path.realpath(path) for path in base]
        self.allow_network = allow_network
        self.max_runtime_seconds = max(1, int(max_runtime_seconds))
        self.max_memory_mb = max(16, int(max_memory_mb))
        self.max_output_bytes = max(1024, int(max_output_bytes))
        self.allowed_commands = set(allowed_commands or [])

    def _normalize_path(self, path: str) -> str:
        return os.path.realpath(path)

    def validate_path(self, path: str) -> bool:
        normalized = self._normalize_path(path)
        return any(normalized == root or normalized.startswith(root + os.sep) for root in self.allowed_roots)

    def _working_directory(self, cwd: Optional[str]) -> str:
        directory = self._normalize_path(cwd or self.allowed_roots[0])
        if not self.validate_path(directory) or not os.path.isdir(directory):
            raise PermissionError("Sandbox working directory is outside the allowed roots.")
        return directory

    def execute(self, function: Callable[..., Any], *args: Any, **kwargs: Any):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(function, *args, **kwargs)
            try:
                return future.result(timeout=self.max_runtime_seconds)
            except TimeoutError:
                future.cancel()
                raise TimeoutError(f"Tool execution exceeded {self.max_runtime_seconds}s")

    def _resource_limits(self):
        if resource is None:
            return
        cpu_seconds = self.max_runtime_seconds
        memory_bytes = self.max_memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_FSIZE, (self.max_output_bytes, self.max_output_bytes))

    def execute_command(
        self,
        command: Sequence[str] | str,
        *,
        cwd: Optional[str] = None,
        network_access: bool = False,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Run an explicitly allowed command in a killable subprocess.

        Commands are never passed through a shell. An empty allowlist denies all
        commands by default; OS/container isolation should be added for hostile
        workloads.
        """
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("A non-empty command argument list is required.")
        executable = Path(argv[0]).name
        if executable not in self.allowed_commands:
            raise PermissionError(f"Command '{executable}' is not allowed by the sandbox.")
        if network_access and not self.allow_network:
            raise PermissionError("Network access is disabled by the sandbox.")

        directory = self._working_directory(cwd)
        child_environment = {"PATH": os.environ.get("PATH", "")}
        if environment:
            child_environment.update({str(key): str(value) for key, value in environment.items()})
        process = subprocess.Popen(
            argv,
            cwd=directory,
            env=child_environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            start_new_session=True,
            preexec_fn=self._resource_limits if resource is not None else None,
        )
        started = time.monotonic()
        try:
            stdout, stderr = process.communicate(timeout=self.max_runtime_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            timed_out = True
        duration = round(time.monotonic() - started, 3)
        return {
            "ok": not timed_out and process.returncode == 0,
            "returncode": process.returncode,
            "stdout": stdout[: self.max_output_bytes].decode("utf-8", errors="replace"),
            "stderr": stderr[: self.max_output_bytes].decode("utf-8", errors="replace"),
            "timed_out": timed_out,
            "duration_seconds": duration,
        }

    def execute_python(self, code: str, *, cwd: Optional[str] = None) -> Dict[str, Any]:
        """Execute Python code in the subprocess sandbox, never in the agent process."""
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python code must be a non-empty string.")
        return self.execute_command(
            [sys.executable, "-I", "-c", code],
            cwd=cwd,
            network_access=False,
        )


class DockerSandbox:
    """Production sandbox wrapper that prefers Docker or Podman when available."""

    def __init__(
        self,
        allowed_roots: Optional[List[str]] = None,
        *,
        allow_network: bool = False,
        max_runtime_seconds: int = 30,
        max_memory_mb: int = 256,
        max_output_bytes: int = 64 * 1024,
        image: str = "python:3.12-slim",
        docker_bin: Optional[str] = None,
    ):
        self.allowed_roots = [os.path.realpath(path) for path in (allowed_roots or [os.getcwd()])]
        self.allow_network = bool(allow_network)
        self.max_runtime_seconds = max(1, int(max_runtime_seconds))
        self.max_memory_mb = max(16, int(max_memory_mb))
        self.max_output_bytes = max(1024, int(max_output_bytes))
        self.image = image
        self.docker_bin = docker_bin or self._detect_docker()

    def _detect_docker(self) -> Optional[str]:
        for candidate in ("docker", "podman", "nerdctl"):
            try:
                result = subprocess.run([candidate, "--version"], capture_output=True, text=True, check=False)
            except OSError:
                continue
            if result.returncode == 0:
                return candidate
        return None

    def is_available(self) -> bool:
        return self.docker_bin is not None

    def _working_directory(self, cwd: Optional[str]) -> str:
        directory = os.path.realpath(cwd or self.allowed_roots[0])
        if not any(directory == root or directory.startswith(root + os.sep) for root in self.allowed_roots):
            raise PermissionError("Sandbox working directory is outside the allowed roots.")
        return directory

    def execute_command(
        self,
        command: Sequence[str] | str,
        *,
        cwd: Optional[str] = None,
        network_access: bool = False,
        environment: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("Docker/Podman is not installed. Install Docker or Podman to enable production sandbox isolation.")
        argv = shlex.split(command) if isinstance(command, str) else list(command)
        if not argv:
            raise ValueError("A non-empty command argument list is required.")
        working_dir = self._working_directory(cwd)
        network_mode = "bridge" if network_access and self.allow_network else "none"
        container_args = [
            self.docker_bin,
            "run",
            "--rm",
            "--read-only",
            "--network",
            network_mode,
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            f"--memory={self.max_memory_mb}m",
            f"--cpus=1",
            "--pids-limit=64",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev",
            "-v",
            f"{working_dir}:/workspace:ro",
            "-w",
            "/workspace",
            self.image,
            *argv,
        ]
        env_items = []
        if environment:
            for key, value in environment.items():
                env_items.extend(["-e", f"{key}={value}"])
        if env_items:
            container_args = container_args[:8] + env_items + container_args[8:]
        completed = subprocess.run(
            container_args,
            capture_output=True,
            text=True,
            timeout=self.max_runtime_seconds,
            check=False,
        )
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[: self.max_output_bytes],
            "stderr": completed.stderr[: self.max_output_bytes],
            "timed_out": False,
            "duration_seconds": round(self.max_runtime_seconds, 3),
        }

    def execute_python(self, code: str, *, cwd: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(code, str) or not code.strip():
            raise ValueError("Python code must be a non-empty string.")
        return self.execute_command([sys.executable, "-I", "-c", code], cwd=cwd, network_access=False)


class ToolRegistry:
    """Registry for tools that the agent may execute with policy checks."""

    def __init__(self, permission_engine: Optional[PermissionEngine] = None, sandbox: Optional[ToolSandbox] = None, policy_engine: Optional[PolicyEngine] = None):
        self.tools: Dict[str, Callable[..., Any]] = {}
        self.permission_engine = permission_engine or PermissionEngine()
        self.sandbox = sandbox or ToolSandbox()
        self.policy_engine = policy_engine or PolicyEngine()

    def register(self, name: str, handler: Callable[..., Any], *, allowed: bool = False, scope: str = "restricted", max_runtime_seconds: int = 30, network_access: bool = False):
        self.tools[name] = handler
        self.permission_engine.allow(name, allowed, scope=scope, max_runtime_seconds=max_runtime_seconds, network_access=network_access)
        return handler

    def execute(self, name: str, *args: Any, approved: bool = False, **kwargs: Any):
        if name not in self.tools:
            return {"ok": False, "error": f"Tool '{name}' is not registered."}

        if not self.permission_engine.check(name):
            return {"ok": False, "error": f"Tool '{name}' is not permitted for this execution context."}
        isolated = isinstance(self.sandbox, DockerSandbox)
        policy = self.policy_engine.decide(
            f"{name} {' '.join(str(arg) for arg in args)}",
            operation="tool",
            approved=approved,
            isolated=isolated,
        )
        if not policy["allowed"]:
            return {"ok": False, "error": "Tool execution blocked by safety policy.", "policy": policy}

        try:
            result = self.sandbox.execute(self.tools[name], *args, **kwargs)
            return {"ok": True, "result": result}
        except Exception as exc:  # pragma: no cover - runtime safeguards
            return {"ok": False, "error": str(exc)}

    def list_tools(self) -> List[str]:
        return sorted(self.tools.keys())

    def execute_command(self, command: Sequence[str] | str, **kwargs: Any) -> Dict[str, Any]:
        """Execute a sandbox command without routing it through a registered tool."""
        return self.sandbox.execute_command(command, **kwargs)

    def execute_python(self, code: str, **kwargs: Any) -> Dict[str, Any]:
        return self.sandbox.execute_python(code, **kwargs)


class SafeToolFactory:
    """Creates commonly used safe tools for NAVA."""

    @staticmethod
    def safe_calculator(expr: str):
        expr = re.sub(r"[^0-9+\-*/().\s]", "", expr)
        if not expr.strip():
            raise ValueError("No valid arithmetic expression found.")
        tree = ast.parse(expr, mode="eval")

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.BinOp):
                left = _eval(node.left)
                right = _eval(node.right)
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.FloorDiv: operator.floordiv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                }
                if type(node.op) not in ops:
                    raise ValueError("Unsupported binary operator")
                return ops[type(node.op)](left, right)
            if isinstance(node, ast.UnaryOp):
                if isinstance(node.op, ast.UAdd):
                    return +_eval(node.operand)
                if isinstance(node.op, ast.USub):
                    return -_eval(node.operand)
                raise ValueError("Unsupported unary operator")
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.Num):
                return node.n
            raise ValueError("Unsupported expression")

        return _eval(tree)

    @staticmethod
    def safe_file_reader(path: str):
        root = os.path.realpath(os.getcwd())
        abs_path = os.path.realpath(path)
        if not os.path.commonpath([root, abs_path]) == root:
            raise PermissionError("Access outside the project root is blocked.")
        with open(abs_path, "r", encoding="utf-8") as handle:
            return handle.read(4000)

    @staticmethod
    def safe_text_summary(text: str):
        if not text:
            return ""
        chunks = re.split(r"\s+", text.strip())
        return " ".join(chunks[:60])


def build_default_registry() -> ToolRegistry:
    permission_engine = PermissionEngine()
    policy_engine = PolicyEngine()
    docker_sandbox = DockerSandbox(allowed_roots=[os.getcwd()], max_runtime_seconds=30, max_memory_mb=256)
    sandbox = docker_sandbox if docker_sandbox.is_available() else ToolSandbox(allowed_roots=[os.getcwd()], max_runtime_seconds=30, max_memory_mb=256)
    registry = ToolRegistry(permission_engine=permission_engine, sandbox=sandbox, policy_engine=policy_engine)
    registry.register("calculator", SafeToolFactory.safe_calculator, allowed=True, scope="safe", max_runtime_seconds=10)
    registry.register("file_reader", SafeToolFactory.safe_file_reader, allowed=True, scope="local", max_runtime_seconds=10)
    registry.register("summary", SafeToolFactory.safe_text_summary, allowed=True, scope="local", max_runtime_seconds=10)
    return registry


__all__ = [
    "PermissionEngine",
    "ToolSandbox",
    "DockerSandbox",
    "ToolRegistry",
    "SafeToolFactory",
    "build_default_registry",
]

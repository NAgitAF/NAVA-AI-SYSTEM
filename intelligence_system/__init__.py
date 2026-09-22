from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ExpertModelRegistry:
    """Maps experts to independently deployable model/adapters.

    Registration is metadata-only; ``is_trained`` is true only when the
    referenced directory contains both a model configuration and weights.
    """

    def __init__(self):
        self.models: dict[str, dict[str, Any]] = {}

    def register(self, expert: str, model_path: str, *, adapter_path: str | None = None, enabled: bool = True):
        if not expert or not model_path:
            raise ValueError("Expert and model_path are required.")
        self.models[expert] = {"model_path": model_path, "adapter_path": adapter_path, "enabled": bool(enabled)}
        return self.models[expert]

    def register_manifest(self, manifest_path: str | Path, *, enabled: bool = True):
        path = Path(manifest_path)
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid expert manifest: {path}") from exc

        required = {"expert", "model_path", "adapter_path", "training_data"}
        if not required.issubset(manifest):
            raise ValueError(f"Expert manifest is missing fields: {sorted(required - set(manifest))}")

        def resolve_path(value: str) -> str:
            resolved = Path(value)
            return str(resolved if resolved.is_absolute() else (path.parent / resolved).resolve())

        entry = self.register(
            manifest["expert"],
            resolve_path(manifest["model_path"]),
            adapter_path=resolve_path(manifest["adapter_path"]),
            enabled=enabled,
        )
        entry.update({
            "expert": manifest["expert"],
            "manifest_path": str(path),
            "training_data": resolve_path(manifest["training_data"]),
            "base_model": manifest.get("base_model"),
            "status": manifest.get("status", "untrained"),
        })
        return entry

    def is_trained(self, expert: str) -> bool:
        model = self.models.get(expert)
        if not model or not model["enabled"]:
            return False

        model_path = Path(model["model_path"])
        adapter_path = Path(model["adapter_path"]) if model.get("adapter_path") else None

        return (
            model_path.is_dir()
            and (model_path / "config.json").is_file()
            and any((model_path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin"))
            and adapter_path is not None
            and adapter_path.is_dir()
            and (adapter_path / "adapter_config.json").is_file()
            and any((adapter_path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
        )

    def resolve(self, expert: str):
        model = self.models.get(expert)
        return model if model and model["enabled"] else None

    def disable(self, expert: str):
        if expert in self.models:
            self.models[expert]["enabled"] = False
        return self.models.get(expert)


__all__ = ["ExpertModelRegistry"]

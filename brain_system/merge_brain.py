"""Controlled LoRA-to-base-model promotion for NAVA."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from brain_system.validator import validate_brain_artifacts


def _get_brain_paths() -> dict[str, str]:
    root = Path(__file__).resolve().parent.parent
    return {
        "seed_brain": str(root / "brain_system" / "seed_brain"),
        "nava_tuned": str(root / "brain_system" / "nava_tuned"),
    }


def execute_brain_merge(benchmark_report: dict[str, Any] | None = None) -> bool:
    """Merge the adapter only after artifact and quality-gate validation.

    Heavy ML dependencies are imported only when a real merge is requested.
    """
    paths = _get_brain_paths()
    seed_dir = Path(paths["seed_brain"])
    adapter_dir = Path(paths["nava_tuned"])
    if not seed_dir.is_dir() or not adapter_dir.is_dir():
        return False

    try:
        report = validate_brain_artifacts(seed_dir, adapter_dir)
    except ValueError:
        return False
    if not report["base_model_matches"]:
        return False

    quality_gate = (benchmark_report or {}).get("quality_gate", {})
    if not quality_gate.get("passed", False):
        return False

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from configs.config_manager import apply_hardware_limits

    config, device = apply_hardware_limits()
    model = AutoModelForCausalLM.from_pretrained(
        str(seed_dir),
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=config.get("cpu_settings", {}).get("low_cpu_mem_usage", True),
    )
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(str(seed_dir))
    temporary = seed_dir.with_name(f".{seed_dir.name}.merged")
    if temporary.exists():
        raise RuntimeError(f"Refusing to overwrite existing staging directory: {temporary}")
    try:
        merged.save_pretrained(str(temporary))
        tokenizer.save_pretrained(str(temporary))
        backup = seed_dir.with_name(f"{seed_dir.name}_backup")
        if backup.exists():
            raise RuntimeError(f"Refusing to overwrite existing backup directory: {backup}")
        os.replace(seed_dir, backup)
        os.replace(temporary, seed_dir)
    except Exception:
        if temporary.exists():
            import shutil
            shutil.rmtree(temporary)
        raise
    return True

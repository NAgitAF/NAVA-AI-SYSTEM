"""Validation utilities for NAVA base and adapter model artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model metadata: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Model metadata must be an object: {path}")
    return value


def validate_brain_artifacts(
    seed_dir: str | os.PathLike[str], adapter_dir: str | os.PathLike[str]
) -> dict[str, Any]:
    """Validate local artifacts without loading model weights into memory."""
    seed = Path(seed_dir)
    adapter = Path(adapter_dir)
    config = _read_json(seed / "config.json")
    adapter_config = _read_json(adapter / "adapter_config.json")
    required_seed = ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors")
    required_adapter = ("adapter_config.json", "adapter_model.safetensors")
    missing_seed = [name for name in required_seed if not (seed / name).is_file()]
    missing_adapter = [name for name in required_adapter if not (adapter / name).is_file()]
    if missing_seed or missing_adapter:
        raise ValueError({"missing_seed": missing_seed, "missing_adapter": missing_adapter})
    if config.get("model_type") != "qwen2" or "Qwen2ForCausalLM" not in config.get("architectures", []):
        raise ValueError("seed_brain is not a supported Qwen2 causal language model.")
    if adapter_config.get("peft_type") != "LORA" or adapter_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("nava_tuned is not a causal-language LoRA adapter.")
    base_path = adapter_config.get("base_model_name_or_path")
    base_matches = bool(base_path) and (adapter / base_path).resolve() == seed.resolve()
    return {
        "valid": True,
        "seed_dir": str(seed.resolve()),
        "adapter_dir": str(adapter.resolve()),
        "model_type": config["model_type"],
        "hidden_size": config.get("hidden_size"),
        "vocab_size": config.get("vocab_size"),
        "lora_rank": adapter_config.get("r"),
        "base_model_matches": base_matches,
        "warnings": [] if base_matches else ["adapter base_model_name_or_path does not resolve to seed_brain"],
    }


def smoke_generate(seed_dir: str, adapter_dir: str, prompt: str, max_new_tokens: int = 16) -> str:
    """Run an opt-in generation check; this intentionally loads the model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(seed_dir)
    model = AutoModelForCausalLM.from_pretrained(seed_dir, device_map="auto", torch_dtype="auto")
    model = PeftModel.from_pretrained(model, adapter_dir)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
TARGET_DIR = Path(__file__).resolve().parents[1] / "brain_system" / "seed_brain"


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[+] downloading {MODEL_ID} to {TARGET_DIR}")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(TARGET_DIR),
    )

    config_path = TARGET_DIR / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
        print("[+] config loaded:")
        print(json.dumps({
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
        }, ensure_ascii=False, indent=2))
    else:
        print("[!] config.json was not created. Check the download output above.")

    files = sorted(p.name for p in TARGET_DIR.iterdir())[:20]
    print("[+] first files in target directory:")
    for name in files:
        print("  -", name)


if __name__ == "__main__":
    main()

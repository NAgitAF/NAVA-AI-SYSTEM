"""Model versioning, quality gates, activation and rollback."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import hashlib
from datetime import datetime, timezone


class VersionRegistry:
    def __init__(self, path=None):
        self.path = path or os.path.join(os.getcwd(), "data", "version_registry.json")
        self.records = []
        self.active_version = None
        self._load()

    def _load(self):
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                self.records = data.get("records", [])
                self.active_version = data.get("active_version")
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self):
        directory = os.path.dirname(self.path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(dir=directory, prefix=".version_registry.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"records": self.records, "active_version": self.active_version}, handle, ensure_ascii=False, indent=2)
        os.replace(temporary_path, self.path)

    def register(self, version, benchmark_score, parent_version=None, status="candidate", benchmark_report=None, artifact_path=None, artifact_hash=None):
        score = float(benchmark_score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("Benchmark score must be between 0 and 1.")
        record = {"version": version, "benchmark_score": score, "parent_version": parent_version, "status": status, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if benchmark_report is not None:
            record["benchmark_report"] = benchmark_report
        if artifact_path is not None:
            record["artifact_path"] = os.path.abspath(artifact_path)
        if artifact_hash is not None:
            record["artifact_hash"] = artifact_hash
        self.records.append(record)
        self._save()
        return record

    def activate(self, version, threshold=0.8, *, force=False, target_path=None):
        record = next((item for item in self.records if item["version"] == version), None)
        if record is None:
            raise ValueError("Version is not registered.")
        if record["benchmark_score"] < threshold:
            raise ValueError("Version failed the quality gate.")
        report = record.get("benchmark_report")
        if not force and report is not None and not report.get("quality_gate", {}).get("passed", False):
            raise ValueError("Version failed the benchmark quality gate.")
        artifact_path = record.get("artifact_path")
        if target_path is not None:
            if not artifact_path or not os.path.isdir(artifact_path):
                raise ValueError("Version artifact is missing.")
            expected_hash = record.get("artifact_hash")
            if expected_hash and ModelVersionStore.digest(artifact_path) != expected_hash:
                raise ValueError("Version artifact hash does not match the registered artifact.")
            ModelVersionStore.promote_directory(artifact_path, target_path)
        if self.active_version:
            for item in self.records:
                if item["version"] == self.active_version:
                    item["status"] = "previous"
        record["status"] = "active"
        self.active_version = version
        self._save()
        return record

    def rollback(self, target_path=None):
        current = next((item for item in self.records if item["version"] == self.active_version), None)
        parent = current.get("parent_version") if current else None
        if not parent:
            raise ValueError("No rollback target is available.")
        return self.activate(parent, threshold=0.0, force=True, target_path=target_path)

    def snapshot(self):
        return {"active_version": self.active_version, "versions": len(self.records)}


class ModelVersionStore:
    """Stores immutable model directories and promotes them atomically."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def stage(self, source_path, version):
        source_path = os.path.abspath(source_path)
        if not os.path.isdir(source_path):
            raise ValueError("Model source directory is missing.")
        destination = os.path.join(self.root, version)
        if os.path.exists(destination):
            raise FileExistsError(f"Model version already exists: {version}")
        temporary = tempfile.mkdtemp(prefix=f".{version}.", dir=self.root)
        shutil.rmtree(temporary)
        shutil.copytree(source_path, temporary)
        os.replace(temporary, destination)
        return destination

    @staticmethod
    def digest(path):
        digest = hashlib.sha256()
        for root, _, files in os.walk(path):
            for filename in sorted(files):
                file_path = os.path.join(root, filename)
                digest.update(os.path.relpath(file_path, path).encode("utf-8"))
                with open(file_path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def promote_directory(source, target):
        source = os.path.abspath(source)
        target = os.path.abspath(target)
        if not os.path.isdir(source):
            raise ValueError("Source model artifact is missing.")
        parent = os.path.dirname(target)
        os.makedirs(parent, exist_ok=True)
        temporary_target = tempfile.mkdtemp(prefix=f".{os.path.basename(target)}.", dir=parent)
        shutil.rmtree(temporary_target)
        shutil.copytree(source, temporary_target)
        backup = f"{target}.previous"
        if os.path.exists(backup):
            shutil.rmtree(backup)
        if os.path.exists(target):
            os.replace(target, backup)
        try:
            os.replace(temporary_target, target)
        except Exception:
            if os.path.exists(backup) and not os.path.exists(target):
                os.replace(backup, target)
            raise
        if os.path.exists(backup):
            shutil.rmtree(backup)

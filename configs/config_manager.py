import os
import json
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")


def get_paths():
    """مركز المسارات لتوحيد عناوين المجلدات والملفات في المشروع بأكمله"""
    return {
        "root_dir": ROOT_DIR,
        "seed_brain": os.path.join(ROOT_DIR, "brain_system", "seed_brain"),
        "nava_tuned": os.path.join(ROOT_DIR, "brain_system", "nava_tuned"),
        "session_buffer": os.path.join(ROOT_DIR, "data", "session_buffer.json"),
        "training_queue": os.path.join(ROOT_DIR, "data", "training_queue.json"),
        "master_archive": os.path.join(ROOT_DIR, "data", "master_archive.json"),
        "memory_store": os.path.join(ROOT_DIR, "data", "memory_store.json"),
    }


def _deep_merge(base, override):
    """يدمج الإعدادات الافتراضية مع ملف التكوين دون فقدان الحقول الجديدة."""
    merged = json.loads(json.dumps(base))
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_default_config():
    """القيم الافتراضية الشاملة للعتاد وتوسيع العقل وشخصية النظام"""
    default_system_prompt = (
        "أنت NAVA، مساعد ذكي ومتقدم للذكاء الاصطناعي من تطوير نواف (Nava). "
        "مهمتك هي تقديم إجابات دقيقة، مفيدة، واحترافية باللغة العربية، "
        "مع الحفاظ على وضوح العبارة ومنطقية التسلسل."
    )
    return {
        "system_prompt": default_system_prompt,
        "brain_expansion": {
            "lora_r": 16,
            "lora_alpha": 32,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            "lora_dropout": 0.05,
            "auto_scale_rank": True,
            "max_rank": 64,
            "min_rank": 16,
            "bias": "none",
        },
        "adaptive_scaling": {
            "enabled": True,
            "min_context_window": 1024,
            "max_context_window": 8192,
            "growth_step": 512,
            "safe_memory_threshold_gb": 4.0,
            "auto_expand_on_input_tokens": 1800,
        },
        "training": {
            "lightweight_mode": False,
            "gradient_checkpointing": True,
            "validation_split": 0.1,
            "min_validation_rows": 10,
            "logging_dir": os.path.join(ROOT_DIR, "training_system", "runs"),
            "enable_tensorboard": True,
            "report_to": "tensorboard",
            "eval_strategy": "steps",
            "learning_rate": 3e-4,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "max_seq_length": 1024,
            "save_total_limit": 3,
        },
        "quality_control": {
            "dedupe_threshold": 0.92,
            "max_history_turns": 10,
            "max_context_tokens": 2048,
        },
        "agent_runtime": {
            "enabled": True,
            "enable_local_retrieval": True,
            "enable_web_lookup": True,
            "max_context_segments": 3,
            "reasoning_mode": "agentic",
            "web_timeout_seconds": 8,
            "self_evaluation_enabled": True,
            "self_evaluation_threshold": 0.7,
            "memory_enabled": True,
            "semantic_retrieval_enabled": True,
            "semantic_similarity_threshold": 0.08,
            "memory_limit": 30,
        },
        "gpu_settings": {
            "enable_cuda": True,
            "max_vram_gb": 4.0,
            "precision": "float16",
            "gpu_device_id": 0,
        },
        "cpu_settings": {
            "cpu_threads": 4,
            "low_cpu_mem_usage": True,
        },
        "model_generation": {
            "max_seq_length": 1024,
            "context_window": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
    }


def load_config():
    """تحميل ملف الإعدادات وإنشاؤه تلقائياً بالقيم الافتراضية إذا لم يكن موجوداً"""
    default_cfg = get_default_config()
    if not os.path.exists(CONFIG_PATH):
        save_config(default_cfg)
        return default_cfg

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        try:
            user_cfg = json.load(f)
            return _deep_merge(default_cfg, user_cfg)
        except json.JSONDecodeError:
            save_config(default_cfg)
            return default_cfg


def save_config(config_data):
    """حفظ بيانات الإعدادات داخل ملف config.json"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)


def update_config_key(section, key, value):
    """تحديث قيمة معينة داخل الإعدادات (تستخدمها لوحة التحكم)"""
    config = load_config()
    if section not in config:
        config[section] = {}
    config[section][key] = value
    save_config(config)
    return config


def get_system_prompt():
    """جلب شخصية النظام (System Prompt) للحقن في المحادثات والتدريب"""
    config = load_config()
    if config.get("system_prompt"):
        return config["system_prompt"]
    return config.get("nava_identity", {}).get(
        "system_prompt",
        "أنت NAVA، مساعد ذكي ومتقدم للذكاء الاصطناعي من تطوير نواف (Nava). تجيب بوضوح ودقة باللغة العربية.",
    )


def get_training_settings():
    """إرجاع إعدادات التدريب والتحقق/المراقبة للـ Trainer."""
    config = load_config()
    training_cfg = config.get("training", {})
    return {
        "lightweight_mode": training_cfg.get("lightweight_mode", False),
        "gradient_checkpointing": training_cfg.get("gradient_checkpointing", True),
        "validation_split": training_cfg.get("validation_split", 0.1),
        "min_validation_rows": training_cfg.get("min_validation_rows", 10),
        "logging_dir": training_cfg.get("logging_dir", os.path.join(ROOT_DIR, "training_system", "runs")),
        "enable_tensorboard": training_cfg.get("enable_tensorboard", True),
        "report_to": training_cfg.get("report_to", "tensorboard"),
        "eval_strategy": training_cfg.get("eval_strategy", "steps"),
        "learning_rate": training_cfg.get("learning_rate", 3e-4),
        "num_train_epochs": training_cfg.get("num_train_epochs", 3),
        "per_device_train_batch_size": training_cfg.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": training_cfg.get("gradient_accumulation_steps", 8),
        "max_seq_length": training_cfg.get("max_seq_length", 1024),
        "save_total_limit": training_cfg.get("save_total_limit", 3),
    }


def get_agent_runtime_settings():
    """إرجاع إعدادات طبقة التفكير / الاسترجاع / البحث للـ agent."""
    config = load_config()
    agent_cfg = config.get("agent_runtime", {})
    return {
        "enabled": agent_cfg.get("enabled", True),
        "enable_local_retrieval": agent_cfg.get("enable_local_retrieval", True),
        "enable_web_lookup": agent_cfg.get("enable_web_lookup", True),
        "max_context_segments": agent_cfg.get("max_context_segments", 3),
        "reasoning_mode": agent_cfg.get("reasoning_mode", "agentic"),
        "web_timeout_seconds": agent_cfg.get("web_timeout_seconds", 8),
        "self_evaluation_enabled": agent_cfg.get("self_evaluation_enabled", True),
        "self_evaluation_threshold": agent_cfg.get("self_evaluation_threshold", 0.7),
        "memory_enabled": agent_cfg.get("memory_enabled", True),
        "semantic_retrieval_enabled": agent_cfg.get("semantic_retrieval_enabled", True),
        "semantic_similarity_threshold": agent_cfg.get("semantic_similarity_threshold", 0.08),
        "memory_limit": agent_cfg.get("memory_limit", 30),
    }


def get_adaptive_scaling_settings():
    """إرجاع إعدادات التوسعة التلقائية للـ context و LoRA."""
    config = load_config()
    scaling_cfg = config.get("adaptive_scaling", {})
    return {
        "enabled": scaling_cfg.get("enabled", True),
        "min_context_window": scaling_cfg.get("min_context_window", 1024),
        "max_context_window": scaling_cfg.get("max_context_window", 8192),
        "growth_step": scaling_cfg.get("growth_step", 512),
        "safe_memory_threshold_gb": scaling_cfg.get("safe_memory_threshold_gb", 4.0),
        "auto_expand_on_input_tokens": scaling_cfg.get("auto_expand_on_input_tokens", 1800),
    }


def resolve_lora_rank(dataset_size, current_rank=None):
    """يزيد رتبة LoRA تلقائياً عند الحاجة، مع حد آمن وعقلاني."""
    config = load_config()
    brain_cfg = config.get("brain_expansion", {})
    scaling_cfg = config.get("adaptive_scaling", {})
    if not brain_cfg.get("auto_scale_rank", True) or not scaling_cfg.get("enabled", True):
        return int(current_rank or brain_cfg.get("lora_r", 16))

    base_rank = int(current_rank or brain_cfg.get("lora_r", 16))
    max_rank = int(brain_cfg.get("max_rank", 64))
    min_rank = int(brain_cfg.get("min_rank", 16))

    if dataset_size is None:
        return max(min_rank, min(base_rank, max_rank))

    if dataset_size >= 200:
        return max(min_rank, min(max_rank, base_rank * 2))
    if dataset_size >= 80:
        return max(min_rank, min(max_rank, base_rank + 8))
    return max(min_rank, min(base_rank, max_rank))


def auto_expand_context_if_needed(required_tokens):
    """يضيف مساحة سياق إ��افية عند قرب المدخل من الحد الحالي بسلامة."""
    config = load_config()
    scaling_cfg = config.get("adaptive_scaling", {})
    if not scaling_cfg.get("enabled", True):
        return int(config.get("model_generation", {}).get("context_window", 2048))

    if required_tokens is None:
        return int(config.get("model_generation", {}).get("context_window", 2048))

    min_window = int(scaling_cfg.get("min_context_window", 1024))
    max_window = int(scaling_cfg.get("max_context_window", 8192))
    growth_step = int(scaling_cfg.get("growth_step", 512))
    current_window = int(config.get("model_generation", {}).get("context_window", 2048))
    safe_target = max(min_window, min(max_window, ((required_tokens // growth_step) + 1) * growth_step))

    if safe_target > current_window:
        config["model_generation"]["context_window"] = safe_target
        save_config(config)
        return safe_target
    return current_window


def apply_hardware_limits():
    """تطبيق حدود المعالج وتحديد وضع تشغيل العتاد المتاح"""
    config = load_config()

    threads = config.get("cpu_settings", {}).get("cpu_threads", 4)
    torch.set_num_threads(threads)

    gpu_enabled = config.get("gpu_settings", {}).get("enable_cuda", True) and torch.cuda.is_available()
    device = "cuda" if gpu_enabled else "cpu"

    print(f"[*] [Config Manager] وضع التشغيل: {device.upper()} | خيوط المعالج: {threads}")
    return config, device

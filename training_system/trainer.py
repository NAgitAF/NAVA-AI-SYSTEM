import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from trl import SFTTrainer, SFTConfig

# تحديد المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))

# ربط المسار الجذري فقط — جميع الاستدعاءات تتم عبر أسماء الحزم الكاملة
sys.path.insert(0, ROOT_DIR)

from data.pipeline_manager import archive_trained_data, _read_json
from configs.config_manager import (
    apply_hardware_limits,
    get_paths,
    get_system_prompt,
    get_training_settings,
    resolve_lora_rank,
)

# مسارات مركزية
PATHS = get_paths()
SEED_BRAIN_DIR = PATHS["seed_brain"]
NAVA_TUNED_DIR = PATHS["nava_tuned"]
TRAINING_FILE = PATHS["training_queue"]
TEMP_CHECKPOINTS = os.path.join(BASE_DIR, "temp_checkpoints")
LOGS_DIR = os.path.join(BASE_DIR, "runs")


def build_chatml_messages(system_prompt, instruction, output):
    """بناء سجل محادثة موحد وفق معيار ChatML لأفضل جودة تدريب."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output}
    ]


def prepare_training_dataset(raw_dataset, tokenizer, system_prompt):
    """يحوّل كل عنصر إلى نص ChatML قابل للتدريب، مع دعم تقسيم التحقق."""
    def format_chatml(example):
        messages = build_chatml_messages(system_prompt, example.get("instruction", ""), example.get("output", ""))
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    return raw_dataset.map(format_chatml)


def plant_knowledge():
    """محرك التدريب والغرس العصبي بمعيار ChatML والتحقق من التجهيز المفرط."""
    queue_data = _read_json(TRAINING_FILE)
    if not queue_data:
        print("[!] [Trainer] ملف (training_queue.json) فارغ.")
        print("[-] الرجاء تشغيل (pipeline_manager.py) أولاً لتجهيز البيانات.")
        return False

    config, device = apply_hardware_limits()
    system_prompt = get_system_prompt()
    training_cfg = get_training_settings()
    os.makedirs(TEMP_CHECKPOINTS, exist_ok=True)
    os.makedirs(training_cfg["logging_dir"], exist_ok=True)

    print(f"\n[*] [Trainer] بدء عملية التدريب والغرس لـ {len(queue_data)} عنصر بمعيار ChatML...")

    print("[*] جاري تحميل العقل الأساسي...")
    tokenizer = AutoTokenizer.from_pretrained(SEED_BRAIN_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    precision_str = config.get("gpu_settings", {}).get("precision", "float16")
    torch_dtype = torch.float16 if precision_str == "float16" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        SEED_BRAIN_DIR,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=config.get("cpu_settings", {}).get("low_cpu_mem_usage", True)
    )

    if training_cfg.get("gradient_checkpointing", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("[*] [Trainer] تم تفعيل gradient checkpointing لتقليل استهلاك الذاكرة وزيادة سرعة التدريب الخفيف.")

    exp_cfg = config.get("brain_expansion", {})
    effective_rank = resolve_lora_rank(len(queue_data), exp_cfg.get("lora_r", 8))
    if effective_rank != exp_cfg.get("lora_r", 8):
        print(f"[*] [Trainer] تم ضبط رتبة LoRA تلقائياً من {exp_cfg.get('lora_r', 8)} إلى {effective_rank} حسب حجم البيانات.")
    peft_config = LoraConfig(
        r=effective_rank,
        lora_alpha=exp_cfg.get("lora_alpha", 32),
        target_modules=exp_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        lora_dropout=exp_cfg.get("lora_dropout", 0.05),
        bias=exp_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM
    )

    if os.path.exists(NAVA_TUNED_DIR) and len(os.listdir(NAVA_TUNED_DIR)) > 0:
        print("[*] جاري استكمال التدريب على الطبقات السابقة...")
        model = PeftModel.from_pretrained(model, NAVA_TUNED_DIR, is_trainable=True)
    else:
        print("[*] جاري إنشاء طبقات عصبية جديدة...")
        model = get_peft_model(model, peft_config)

    raw_dataset = load_dataset("json", data_files=TRAINING_FILE)["train"]
    prepared_dataset = prepare_training_dataset(raw_dataset, tokenizer, system_prompt)

    validation_split = float(training_cfg.get("validation_split", 0.1))
    min_eval_rows = int(training_cfg.get("min_validation_rows", 5))
    if len(prepared_dataset) >= max(10, min_eval_rows * 2):
        split_dataset = prepared_dataset.train_test_split(test_size=validation_split, seed=42)
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        eval_strategy = training_cfg.get("eval_strategy", "epoch")
    else:
        train_dataset = prepared_dataset
        eval_dataset = None
        eval_strategy = "no"

    gen_cfg = config.get("model_generation", {})
    training_args = SFTConfig(
        output_dir=TEMP_CHECKPOINTS,
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=training_cfg.get("learning_rate", 1e-4),
        num_train_epochs=training_cfg.get("num_train_epochs", 2),
        logging_steps=1,
        logging_dir=training_cfg.get("logging_dir", LOGS_DIR),
        report_to=training_cfg.get("report_to", "tensorboard"),
        eval_strategy=eval_strategy,
        save_strategy="steps",
        save_steps=max(10, len(train_dataset) // 10) if hasattr(train_dataset, '__len__') else 10,
        save_total_limit=training_cfg.get("save_total_limit", 2),
        use_cpu=(device == "cpu"),
        fp16=(device == "cuda" and precision_str == "float16"),
        max_length=training_cfg.get("max_seq_length", gen_cfg.get("max_seq_length", 512)),
        dataset_text_field="text"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args
    )

    print("[*] جاري غرس المعارف في كرت الشاشة/المعالج بمعيار ChatML ومراقبة الخسارة...")
    trainer.train()

    os.makedirs(NAVA_TUNED_DIR, exist_ok=True)
    model.save_pretrained(NAVA_TUNED_DIR)
    tokenizer.save_pretrained(NAVA_TUNED_DIR)
    print(f"[+] [Trainer] تم حفظ العقل المطور في ({NAVA_TUNED_DIR}) بنجاح.")

    print("\n[*] [Trainer] جاري نقل البيانات المدربة إلى الأرشيف الدائم...")
    archive_trained_data()
    return True


if __name__ == "__main__":
    plant_knowledge()
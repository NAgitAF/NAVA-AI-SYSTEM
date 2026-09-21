import os
import sys
import json
import re
from datetime import datetime
from difflib import SequenceMatcher

# ربط المسار الجذري
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, ROOT_DIR)

from configs.config_manager import get_paths

PATHS = get_paths()
FILE_1_SESSION  = PATHS["session_buffer"]   # ذاكرة الجلسة الحالية
FILE_2_TRAINING = PATHS["training_queue"]  # طابور التدريب
FILE_3_ARCHIVE  = PATHS["master_archive"]  # الأرشيف الدائم

def _read_json(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def _write_json(path, data):
    # التأكد من وجود المجلد
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def _normalize_for_dedupe(text):
    """يُحوِّل النص إلى صيغة موحدة لقياس التشابه وإزالة التكرار."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = re.sub(r"[^\w\s\u0621-\u064A]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _instruction_similarity(a, b):
    """يُرجع نسبة التشابه بين نصيْن بهدف منع التكرار في التدرب."""
    a_norm = _normalize_for_dedupe(a)
    b_norm = _normalize_for_dedupe(b)
    if not a_norm or not b_norm:
        return 0.0
    if a_norm == b_norm:
        return 1.0

    # مزيج من Jaccard و SequenceMatcher لتقليل الحساسية للتكرار السطحي
    a_tokens = set(a_norm.split())
    b_tokens = set(b_norm.split())
    if a_tokens and b_tokens:
        jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    else:
        jaccard = 0.0

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return max(jaccard, ratio)


def _is_duplicate_instruction(candidate, existing_instructions):
    """يدقق ما إذا كان التوجيه مكرّراً بنسبة عالية."""
    if not candidate:
        return True
    candidate_norm = _normalize_for_dedupe(candidate)
    for existing in existing_instructions:
        if not existing:
            continue
        if _instruction_similarity(candidate_norm, existing) >= 0.92:
            return True
    return False


def receive_from_list(list_name, instruction, output):
    """(تستخدمها القوائم) لاستقبال البيانات وتفريغها في ملف الجلسة الحالية"""
    session_data = _read_json(FILE_1_SESSION)
    entry = {
        "source": list_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instruction": instruction.strip(),
        "output": output.strip(),
        "is_trained": False
    }
    
    # منع التكرار المباشر داخل الجلسة الحالية أيضاً
    for item in session_data:
        existing_instruction = item.get("instruction", "")
        if _instruction_similarity(existing_instruction, entry["instruction"]) >= 0.92:
            return

    session_data.append(entry)
    _write_json(FILE_1_SESSION, session_data)
    print(f"[+] [Data] تمت إضافة معلومة من [{list_name}] إلى session_buffer.json")

def organize_data_for_training():
    """المهمة الرئيسية عند تشغيل الملف: النقل مع منع التكرار بنسبة 100% من الجلسة إلى طابور التدريب"""
    session_data = _read_json(FILE_1_SESSION)
    if not session_data:
        print("[!] [Data Manager] لا توجد بيانات جديدة في (session_buffer.json) لنقلها.")
        return False

    training_queue = _read_json(FILE_2_TRAINING)
    archive_data = _read_json(FILE_3_ARCHIVE)
    
    existing_instructions = []
    for item in training_queue:
        if "instruction" in item:
            existing_instructions.append(item["instruction"].strip())
    for item in archive_data:
        if "instruction" in item:
            existing_instructions.append(item["instruction"].strip())

    added_count = 0
    skipped_count = 0

    for item in session_data:
        ins = item.get("instruction", "").strip()
        if not _is_duplicate_instruction(ins, existing_instructions):
            training_queue.append(item)
            existing_instructions.append(ins)
            added_count += 1
        else:
            skipped_count += 1

    _write_json(FILE_2_TRAINING, training_queue)
    _write_json(FILE_1_SESSION, [])
    
    print(f"[+] [Data Manager] تمت معالجة ونقل {added_count} عنصر جديد إلى (training_queue.json).")
    if skipped_count > 0:
        print(f"[-] [Deduplication] تم استبعاد {skipped_count} عنصر مكرر لتوفير وقت التدريب ومنع الانحياز.")
    print(f"[*] البيانات جاهزة الآن. يمكنك تشغيل ملف التدريب متى شئت.")
    return True

def archive_trained_data():
    """(يستدعيها ملف التدريب) لنقل البيانات من الطابور إلى الأرشيف الدائم"""
    training_queue = _read_json(FILE_2_TRAINING)
    archive_data = _read_json(FILE_3_ARCHIVE)
    
    archived_count = 0
    for item in training_queue:
        item["is_trained"] = True
        item["archived_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        archive_data.append(item)
        archived_count += 1

    _write_json(FILE_2_TRAINING, []) # تصفير طابور التدريب بعد الأرشفة
    _write_json(FILE_3_ARCHIVE, archive_data)
    print(f"[+] [Archive System] تم ترحيل {archived_count} عنصر إلى الأرشيف الدائم (master_archive.json).")

def import_json_file(file_path):
    """استيراد بيانات من ملف JSON خارجي مباشرة إلى الجلسة"""
    if not os.path.exists(file_path):
        print(f"[!] خطأ: الملف {file_path} غير موجود.")
        return False
        
    data = _read_json(file_path)
    if not data or not isinstance(data, list):
        print("[!] خطأ: الملف يجب أن يحتوي على قائمة (List) من كائنات JSON (instruction, output).")
        return False
        
    imported_count = 0
    for item in data:
        ins = item.get("instruction", item.get("q", ""))
        out = item.get("output", item.get("a", ""))
        if ins and out:
            receive_from_list("JSON_IMPORT", ins, out)
            imported_count += 1
            
    print(f"[+] تم استيراد {imported_count} معلومة بنجاح من {os.path.basename(file_path)}.")
    return True

def get_stats():
    """إرجاع إحصائيات عن عدد العناصر في ملفات البيانات (للوحة التحكم)"""
    return {
        "session": len(_read_json(FILE_1_SESSION)),
        "queue": len(_read_json(FILE_2_TRAINING)),
        "archive": len(_read_json(FILE_3_ARCHIVE))
    }

if __name__ == "__main__":
    print("=== نظام إدارة خط البيانات (مع الفلترة ومنع التكرار) ===")
    organize_data_for_training()
import os
import sys

# ربط المسار الجذري للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, ROOT_DIR)

from data.pipeline_manager import receive_from_list


def add_direct_entry(instruction, output):
    """إرسال توجيه وإجابة مباشرة لملف البيانات الحالي"""
    if not instruction.strip() or not output.strip():
        print("[!] يجب إدخال السؤال والجواب بشكل صحيح.")
        return False

    receive_from_list("MANUAL_LIST", instruction.strip(), output.strip())
    print(f"[✓] [Manual List] تم إرسال التوجيه بنجاح إلى session_buffer.json.")
    return True


if __name__ == "__main__":
    print("--- قائمة الإدخال المباشر ---")
    q = input("أدخل السؤال / التوجيه: ")
    a = input("أدخل الإجابة / المخرج: ")
    add_direct_entry(q, a)
import os
import sys

# ربط المسار الجذري للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, ROOT_DIR)

from data.pipeline_manager import receive_from_list


def read_and_send_file(file_path):
    """قراءة ملف نصي أو كود برمجي وإرسال محتواه لملف البيانات الحالي"""
    if not os.path.exists(file_path):
        print(f"[!] خطأ: الملف غير موجود في المسار: {file_path}")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            print(f"[!] خطأ: الملف فارغ: {file_path}")
            return False

        file_name = os.path.basename(file_path)
        instruction = f"محتوى الملف البرمجي أو النصي: {file_name}"
        receive_from_list("FILE_LIST", instruction, content)
        print(f"[✓] [File List] تم سحب وإرسال محتوى '{file_name}' بنجاح.")
        return True
    except Exception as e:
        print(f"[!] خطأ أثناء قراءة الملف: {e}")
        return False


if __name__ == "__main__":
    path = input("أدخل مسار الملف لقراءته: ")
    if path.strip():
        read_and_send_file(path.strip())
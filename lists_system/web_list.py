import os
import sys
import requests
import socket
from datetime import datetime

# ربط المسار الجذري للمشروع
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
sys.path.insert(0, ROOT_DIR)

from data.pipeline_manager import receive_from_list


def is_connected():
    """فحص الاتصال بالإنترنت"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def smart_web_explorer(topic=None):
    """
    نظام البحث العميق واستكشاف المعارف عبر ويكيبيديا العربية بصيغة السؤال والجواب السياقي.
    يمكن تمرير الموضوع مباشرة أو سيطلبه من المستخدم.
    يُعيد عدد المعلومات المُستخرجة أو 0 في حالة الفشل.
    """
    print("[*] [Smart Web List] بدء نظام البحث العميق واستكشاف المعارف عبر ويكيبيديا (مع الاستنباط السياقي).")

    if not is_connected():
        print("[!] خطأ: لا يوجد اتصال بالإنترنت.")
        return 0

    if topic is None:
        topic = input("[?] ما هو الموضوع الذي تريد من NAVA استكشافه بعمق؟: ").strip()

    if not topic:
        print("[!] لم يتم إدخال موضوع.")
        return 0

    headers = {
        "User-Agent": "NavaAIResearchAssistant/1.0 (Educational AI Project)"
    }

    print(f"[*] جاري تنفيذ بحث موسع وعميق حول: '{topic}'...")

    search_url = "https://ar.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "srlimit": 10
    }

    try:
        response = requests.get(search_url, params=search_params, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"[!] خطأ في الاتصال البوابي (كود: {response.status_code})")
            return 0

        data = response.json()
        search_results = data.get("query", {}).get("search", [])

        if not search_results:
            print("[!] لم يتم العثور على مقالات مطابقة في ويكيبيديا.")
            return 0

        collected_data = []

        for item in search_results:
            title = item["title"]
            print(f"[-] جاري سحب واستخلاص المعارف بعمق من مقالة: {title}...")

            extract_params = {
                "action": "query",
                "prop": "extracts",
                "explaintext": True,
                "titles": title,
                "format": "json"
            }

            try:
                ext_response = requests.get(search_url, params=extract_params, headers=headers, timeout=10)
                if ext_response.status_code == 200:
                    ext_data = ext_response.json()
                    pages = ext_data.get("query", {}).get("pages", {})
                    for page_id, page_info in pages.items():
                        if page_id != "-1" and "extract" in page_info:
                            content = page_info["extract"]
                            paragraphs = content.split("\n")
                            for p in paragraphs:
                                p_clean = p.strip()
                                if len(p_clean) > 50 and p_clean not in collected_data:
                                    collected_data.append(p_clean)
            except requests.RequestException as req_err:
                print(f"[!] خطأ أثناء سحب مقالة '{title}': {req_err}")
                continue

        if collected_data:
            print(f"\n[*] تم جمع {len(collected_data)} فقرة سياقية، جاري صياغتها وإرسالها إلى خط البيانات...")

            for text_chunk in collected_data:
                # صياغة سياقية تعتمد على الفهم والاستنباط بدلاً من التلقين الصامت
                instruction = f"استناداً إلى هذا السياق: {text_chunk}، اشرح لي عن {topic}."
                output = f"بناءً على السياق المقدم حول ({topic})، فإن التفاصيل توضح أن: {text_chunk}"

                receive_from_list(
                    "Smart_Web_QA",
                    instruction,
                    output
                )

            print(f"[+] تم إرسال {len(collected_data)} معلومة سياقية إلى session_buffer.json بنجاح.")
            return len(collected_data)
        else:
            print("[!] لم يتم استخلاص فقرات كافية.")
            return 0

    except Exception as q_err:
        print(f"[!] حدث خطأ أثناء التنفيذ: {q_err}")
        return 0


if __name__ == "__main__":
    smart_web_explorer()
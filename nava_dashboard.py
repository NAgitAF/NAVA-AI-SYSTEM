import os
import sys
import psutil
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

# ربط المسار الجذري
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from configs.config_manager import load_config, update_config_key, apply_hardware_limits, get_paths
from data.pipeline_manager import get_stats, import_json_file, organize_data_for_training
from lists_system.web_list import smart_web_explorer
from lists_system.manual_list import add_direct_entry
from training_system.trainer import plant_knowledge
from brain_system.merge_brain import execute_brain_merge
from nava_architecture import NAVAArchitecture
from versioning_system import VersionRegistry

console = Console()
PATHS = get_paths()

class NavaDashboard:
    def __init__(self):
        self.config, self.device = apply_hardware_limits()
        self.architecture = NAVAArchitecture()
        self.architecture.state.current_model = PATHS.get("seed_brain", "seed_brain")
        self.architecture.state.current_version = "v1.0"
        self.architecture.state.system_health = "healthy"
        self.version_registry = VersionRegistry(os.path.join(PATHS["root_dir"], "data", "version_registry.json"))

    def display_health_monitor(self):
        self.architecture.event_bus.emit("SYSTEM_HEALTH_CHECK", {"status": "checking"})
        console.clear()
        ram = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        table = Table(title="[bold cyan]مراقبة حالة النظام والنموذج[/bold cyan]")
        table.add_column("المورد", style="yellow")
        table.add_column("الحالة / الاستهلاك", style="green")
        
        table.add_row("استهلاك المعالج (CPU)", f"{cpu}%")
        table.add_row("استهلاك الذاكرة (RAM)", f"{ram.percent}% (متبقي: {ram.available / (1024**3):.2f} GB)")
        
        device_status = "[green]مفعل (GPU)[/green]" if self.device == "cuda" else "[red]غير مفعل (يعمل على المعالج)[/red]"
        table.add_row("حالة كرت الشاشة (CUDA)", device_status)
        
        seed_brain_path = PATHS["seed_brain"]
        model_exists = os.path.exists(seed_brain_path) and os.path.exists(os.path.join(seed_brain_path, "model.safetensors"))
        table.add_row("صحة العقل الأساسي (seed_brain)", "[green]سليمة[/green]" if model_exists else "[red]غير مكتمل أو مفقود[/red]")
        
        # مراقبة الخلايا والتدريب
        ctx = self.config.get("model_generation", {}).get("context_window", 2048)
        context_status = "[green]طبيعي[/green]" if ctx <= 2048 else "[yellow]تحذير: سقف عالٍ قد يستهلك الرام[/yellow]"
        table.add_row("سقف الخلايا (Context)", f"{ctx} Token - {context_status}")
        
        stats = get_stats()
        table.add_row("البيانات قيد الانتظار", f"{stats['session']} (في الجلسة) | {stats['queue']} (في طابور التدريب)")
        table.add_row("البيانات المحفوظة", f"{stats['archive']} (في الأرشيف)")
        table.add_row("الإصدار النشط", str(self.version_registry.active_version or "غير محدد"))
        table.add_row("الذاكرة", str(self.architecture.memory_manager.snapshot()["total_count"]))
        table.add_row("الرسم المعرفي", str(self.architecture.knowledge_graph.snapshot()["edge_count"]) + " علاقات")
        
        console.print(table)
        input("\nاضغط Enter للعودة...")

    def configure_cells(self):
        console.clear()
        console.print(Panel("[bold yellow]إعدادات خلايا النموذج (Context & Limits)[/bold yellow]"))
        
        gen_config = self.config.get("model_generation", {})
        current_ctx = gen_config.get("context_window", 2048)
        console.print(f"السقف الحالي للخلايا: {current_ctx}")
        
        new_cells = Prompt.ask("أدخل سقف الخلايا الجديد (مثال: 512, 1024, 2048, 4096)", default=str(current_ctx))
        if new_cells.isdigit():
            gen_config["context_window"] = int(new_cells)
            self.config = update_config_key("model_generation", "context_window", int(new_cells))
            console.print("[green][+] تم زيادة/تعديل سقف الخلايا بنجاح![/green]")
        else:
            console.print("[red][!] قيمة غير صالحة.[/red]")
        input("\nاضغط Enter للعودة...")

    def training_hub(self):
        console.clear()
        console.print(Panel("[bold magenta]قسم إدخال وحقن البيانات[/bold magenta]"))
        console.print("1. إدخال معلومة جديدة يدوياً")
        console.print("2. رفع ملف بيانات خارجي (JSON)")
        console.print("0. عودة")
        
        choice = Prompt.ask("اختر العملية")
        if choice == "1":
            instruction = Prompt.ask("السؤال / التوجيه")
            output = Prompt.ask("الرد / الإجابة")            
            add_direct_entry(instruction, output)
        elif choice == "2":
            file_path = Prompt.ask("أدخل المسار الكامل لملف JSON")
            import_json_file(file_path)
        input("\nاضغط Enter للعودة...")

    def run_diagnostics(self):
        console.clear()
        console.print(Panel("[bold red]أداة فحص الأخطاء (Diagnostics)[/bold red]"))
        errors = 0
        
        if not os.path.exists(PATHS["seed_brain"]):
            console.print(f"[red][!] خطأ: مجلد النموذج الأساسي غير موجود في {PATHS['seed_brain']}.[/red]")
            errors += 1 
            
        stats = get_stats()
        if stats['queue'] == 0 and stats['session'] == 0:
            console.print("[yellow][!] تنبيه: لا توجد بيانات للتدريب حالياً.[/yellow]")
            
        ctx = self.config.get("model_generation", {}).get("context_window", 2048)
        if ctx > 4096 and psutil.virtual_memory().available < (4 * 1024**3):
            console.print("[red][!] خطر: سقف الخلايا مرتفع جداً مقارنة بالرام المتاح، قد يحدث انهيار (OOM).[/red]")
            errors += 1
            
        if errors == 0:
            console.print("[green][+] النظام سليم بالكامل، ولا توجد أي أخطاء أو تعارضات هيكلية.[/green]")
        else:
            console.print(f"[red]تم العثور على {errors} أخطاء تحتاج مراجعة.[/red]")
            
        input("\nاضغط Enter للعودة...")

    def run_knowledge_importer(self):
        console.clear()
        console.print(Panel("[bold blue]تشغيل أداة استيراد المعرفة التلقائية من الويب...[/bold blue]", border_style="blue"))
        smart_web_explorer()
        input("\n[green]انتهت جلسة استيراد المعرفة. اضغط Enter للعودة إلى لوحة التحكم...[/green]")

    def start_training(self):
        self.architecture.state.training_state = "processing"
        console.clear()
        console.print(Panel("[bold green]بدء عملية تدريب وتطوير NAVA[/bold green]", border_style="green"))
        
        # أولاً: نقل البيانات من الجلسة إلى الطابور
        console.print("[*] تجهيز خط البيانات...")
        organize_data_for_training()
        
        # ثانياً: فحص الطابور بعد التجهيز
        stats = get_stats()
        if stats['queue'] == 0:
            console.print("[red][!] لا توجد بيانات تدريب في الطابور (training_queue.json).[/red]")
            console.print("[yellow]يرجى إضافة بيانات جديدة عبر قسم التدريب أو استيراد المعرفة أولاً.[/yellow]")
            input("\nاضغط Enter للعودة...")
            return
            
        # ثالثاً: بدء التدريب
        try:
            plant_knowledge()
            self.architecture.state.training_state = "completed"
            console.print(f"[green][+] اكتملت دورة التدريب والتحديث بنجاح.[/green]")
        except Exception as e:
            self.architecture.state.training_state = "failed"
            console.print(f"[red][!] حدث خطأ أثناء التدريب: {e}[/red]")
        input("\nاضغط Enter للعودة...")

    def merge_brain_weights(self):
        console.clear()
        console.print(Panel("[bold yellow]دمج وصهر العقل (Brain Merge)[/bold yellow]", border_style="yellow"))
        console.print("[white]تنبيه: هذه العملية ستقوم بدمج التعديلات (LoRA) بشكل دائم داخل العقل الأساسي (seed_brain).[/white]")
        
        confirm = Prompt.ask("هل أنت متأكد من تنفيذ الصهر النهائي؟ (yes/no)")
        if confirm.strip().lower() == "yes":
            execute_brain_merge()
        else:
            console.print("[*] تم إلغاء العملية.")
        input("\nاضغط Enter للعودة...")

    def main_menu(self):
        while True:
            console.clear()
            console.print(Panel(
                "[bold cyan]NAVA Control Center - مركز قيادة العقل المحلي[/bold cyan]\n"
                "[white]إدارة الإعدادات، الخلايا، المراقبة، والتدريب[/white]",
                border_style="cyan"
            ))
            
            console.print(" [1] [yellow]مراقبة حالة النموذج والذاكرة (Monitor)[/yellow]")
            console.print(" [2] [cyan]إعدادات سقف الخلايا والسياق (Context/Limits)[/cyan]")
            console.print(" [3] [magenta]قسم إدخال وحقن المعرفة (Data Input)[/magenta]")
            console.print(" [4] [red]فحص الأخطاء والتشخيص (Diagnostics)[/red]")
            console.print(" [5] [blue]استيراد المعرفة التلقائي من الويب (Web Explorer)[/blue]")            
            console.print(" [6] [green]بدء تدريب NAVA على البيانات الجديدة (Start Training)[/green]")            
            console.print(" [7] [yellow]صهر ودمج العقل (Brain Merge)[/yellow]")
            console.print(" [0] [white]خروج[/white]\n")
            
            choice = Prompt.ask("اختر رقم المهمة", choices=["0", "1", "2", "3", "4", "5", "6", "7"])
            
            if choice == "1":
                self.display_health_monitor()
            elif choice == "2":
                self.configure_cells()
            elif choice == "3":
                self.training_hub()
            elif choice == "4":
                self.run_diagnostics()
            elif choice == "5":
                self.run_knowledge_importer()
            elif choice == "6":
                self.start_training()
            elif choice == "7":
                self.merge_brain_weights()
            elif choice == "0":
                console.print("[cyan]تم إغلاق لوحة التحكم.[/cyan]")
                break

if __name__ == "__main__":
    try:
        dashboard = NavaDashboard()
        dashboard.main_menu()
    except KeyboardInterrupt:
        print("\nتم إنهاء البرنامج.")
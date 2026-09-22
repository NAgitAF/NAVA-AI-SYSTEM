import os
import sys
import json
import re
import ast
import math
import operator
from collections import Counter, OrderedDict
from datetime import datetime

import numpy as np
import requests
import torch
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ربط المسار الجذري للمشروع لضمان استدعاء الحزم بشكل صحيح
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from nava_architecture import NAVAAgentRuntime, NAVAArchitecture

try:
    from configs.config_manager import (
        apply_hardware_limits,
        auto_expand_context_if_needed,
        get_agent_runtime_settings,
        get_paths,
        get_system_prompt,
    )
    from data.pipeline_manager import receive_from_list, _read_json
except ImportError as e:
    print(f"[!] خطأ في استدعاء الملفات المساعدة: {e}")
    sys.exit(1)

console = Console()
PATHS = get_paths()


class MemoryManager:
    """ذاكرة طويلة المدى تدعم البحث الدلالي، مع التبديل إلى FAISS عند توفر الحزمة."""
    def __init__(self, path=None):
        self.path = path or PATHS["memory_store"]
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.entries = self._load()
        self.faiss_index = None
        self.faiss_ids = []
        self.embedding_model = None
        self.embedding_dim = 384
        self.vector_index_path = os.path.join(os.path.dirname(self.path), "memory_faiss.index")
        self._init_semantic_backend()

    def _init_semantic_backend(self):
        self.embedding_dim = 384
        try:
            import faiss  # noqa: F401
            self.faiss_available = True
        except Exception:
            self.faiss_available = False

        try:
            from sentence_transformers import SentenceTransformer
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        except Exception:
            self.embedding_model = None

        if self.entries:
            self._rebuild_vector_index()

    def _load(self):
        if not os.path.exists(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def _save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def _normalize_text(self, text):
        if text is None:
            return ""
        text = str(text).lower()
        text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
        text = re.sub(r"[^\w\s\u0621-\u064A]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _tokenize(self, text):
        normalized = self._normalize_text(text)
        if not normalized:
            return []
        return normalized.split()

    def _build_vocab(self):
        vocab = {}
        for entry in self.entries:
            text = f"{entry.get('topic', '')} {entry.get('fact', '')}"
            for token in self._tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)
        return vocab

    def _vectorize(self, text, vocab=None):
        tokens = self._tokenize(text)
        if not tokens:
            return []
        if vocab is None:
            vocab = self._build_vocab()
        counts = Counter(tokens)
        vector = [0.0] * len(vocab)
        for token, count in counts.items():
            idx = vocab.get(token)
            if idx is not None:
                vector[idx] = float(count)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return [0.0] * len(vocab)
        return [v / norm for v in vector]

    def _cosine_similarity(self, a, b):
        if not a or not b:
            return 0.0
        min_len = min(len(a), len(b))
        dot = sum(x * y for x, y in zip(a[:min_len], b[:min_len]))
        a_norm = math.sqrt(sum(x * x for x in a))
        b_norm = math.sqrt(sum(x * x for x in b))
        if a_norm == 0 or b_norm == 0:
            return 0.0
        return dot / (a_norm * b_norm)

    def _rebuild_vector_index(self):
        if not self.faiss_available or self.embedding_model is None:
            return
        try:
            import faiss
            vectors = []
            self.faiss_ids = []
            for idx, item in enumerate(self.entries):
                text = f"{item.get('topic', '')} {item.get('fact', '')}"
                vector = self.embedding_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                vectors.append(np.asarray(vector, dtype='float32'))
                self.faiss_ids.append(idx)
            if not vectors:
                self.faiss_index = None
                return
            dim = len(vectors[0])
            index = faiss.IndexFlatIP(dim)
            index.add(np.vstack(vectors))
            self.faiss_index = index
            self.embedding_dim = dim
        except Exception:
            self.faiss_index = None

    def _faiss_search(self, query, limit=3):
        if not self.faiss_available or self.embedding_model is None or self.faiss_index is None:
            return []
        try:
            import faiss
            vector = self.embedding_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
            query_vec = np.asarray(vector, dtype='float32').reshape(1, -1)
            scores, ids = self.faiss_index.search(query_vec, min(limit, self.faiss_index.ntotal))
            results = []
            for score, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(self.entries):
                    continue
                results.append({
                    "score": float(score),
                    "fact": self.entries[int(idx)].get("fact", ""),
                    "topic": self.entries[int(idx)].get("topic", ""),
                })
            return results
        except Exception:
            return []

    def add(self, topic, fact, source="system"):
        if not topic or not fact:
            return None
        item = {
            "topic": str(topic).strip(),
            "fact": str(fact).strip(),
            "source": source,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        self.entries.append(item)
        limit = self._limit()
        if len(self.entries) > limit:
            self.entries = self.entries[-limit:]
        self._save()
        self._rebuild_vector_index()
        return item

    def _limit(self):
        cfg = get_agent_runtime_settings()
        return int(cfg.get("memory_limit", 30))

    def _score(self, query, text):
        q = self._normalize_text(query)
        t = self._normalize_text(text)
        if not q or not t:
            return 0.0
        q_tokens = set(q.split())
        t_tokens = set(t.split())
        if not q_tokens or not t_tokens:
            return 0.0
        overlap = len(q_tokens & t_tokens)
        return overlap / max(1, len(q_tokens | t_tokens))

    def _semantic_score(self, query, text):
        vocab = self._build_vocab()
        query_vector = self._vectorize(query, vocab=vocab)
        if not query_vector:
            return 0.0
        doc_vector = self._vectorize(text, vocab=vocab)
        if not doc_vector:
            return 0.0
        return self._cosine_similarity(query_vector, doc_vector)

    def retrieve(self, query, limit=3):
        if not query or not self.entries:
            return []

        cfg = get_agent_runtime_settings()
        threshold = float(cfg.get("semantic_similarity_threshold", 0.08))
        semantic_enabled = bool(cfg.get("semantic_retrieval_enabled", True))

        faiss_hits = []
        if semantic_enabled and self.faiss_available and self.embedding_model is not None:
            faiss_hits = self._faiss_search(query, limit=limit)
            if faiss_hits:
                results = []
                for match in faiss_hits:
                    text = f"{match.get('topic', '')} {match.get('fact', '')}"
                    lexical = self._score(query, text)
                    combined = min(1.0, lexical * 0.4 + max(0.0, match.get('score', 0.0)) * 0.6)
                    if combined >= threshold:
                        results.append({
                            "score": combined,
                            "fact": match.get("fact", ""),
                            "topic": match.get("topic", ""),
                            "semantic_score": float(match.get('score', 0.0)),
                            "lexical_score": lexical,
                        })
                if results:
                    return sorted(results, key=lambda x: x["score"], reverse=True)[:limit]

        matches = []
        vocab = self._build_vocab()
        query_vector = self._vectorize(query, vocab=vocab) if semantic_enabled else []
        for item in self.entries:
            text = f"{item.get('topic', '')} {item.get('fact', '')}"
            lexical = self._score(query, text)
            semantic = 0.0
            if semantic_enabled and query_vector:
                doc_vector = self._vectorize(text, vocab=vocab)
                semantic = self._cosine_similarity(query_vector, doc_vector)
            combined = lexical * 0.65 + semantic * 0.35
            if combined > threshold:
                matches.append({
                    "score": combined,
                    "fact": item.get("fact", ""),
                    "topic": item.get("topic", ""),
                    "semantic_score": semantic,
                    "lexical_score": lexical,
                })

        matches = sorted(matches, key=lambda x: x["score"], reverse=True)
        return matches[:limit]

    def all(self):
        return list(self.entries)


class ToolRegistry:
    """مسجل أدوات حقيقي يسمح للنموذج باختيار أدوات مختلفة حسب السؤال."""
    def __init__(self, chat_instance=None):
        self.chat = chat_instance
        self.tools = OrderedDict()
        self._register_defaults()

    def _register_defaults(self):
        self.register("calculator", self._calculator_tool, "يحسب معادلات بسيطة أو أرقامية.")
        self.register("memory_lookup", self._memory_tool, "يبحث في ذاكرة NAVA طويلة المدى.")
        self.register("local_search", self._local_search_tool, "يبحث داخل بيانات التدريب والملفات المحلية.")
        self.register("web_lookup", self._web_lookup_tool, "يسترجع مراجع قصيرة من الويب إذا لزم الأمر.")

    def register(self, name, fn, description):
        self.tools[name] = {"fn": fn, "description": description}

    def _calculator_tool(self, query):
        try:
            expr = re.sub(r"[^0-9+\-*/().\s]", "", query)
            expr = expr.strip()
            if not expr:
                return {"ok": False, "result": "لا توجد معادلة صالحة."}
            tree = ast.parse(expr, mode="eval")

            def safe_eval(node):
                if isinstance(node, ast.Expression):
                    return safe_eval(node.body)
                if isinstance(node, ast.BinOp):
                    left = safe_eval(node.left)
                    right = safe_eval(node.right)
                    ops = {
                        ast.Add: operator.add,
                        ast.Sub: operator.sub,
                        ast.Mult: operator.mul,
                        ast.Div: operator.truediv,
                        ast.FloorDiv: operator.floordiv,
                        ast.Mod: operator.mod,
                        ast.Pow: operator.pow,
                    }
                    if type(node.op) not in ops:
                        raise ValueError("Unsupported operator")
                    return ops[type(node.op)](left, right)
                if isinstance(node, ast.UnaryOp):
                    ops = {ast.UAdd: operator.pos, ast.USub: operator.neg}
                    if type(node.op) not in ops:
                        raise ValueError("Unsupported unary operator")
                    return ops[type(node.op)](safe_eval(node.operand))
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return node.value
                if isinstance(node, ast.Num):
                    return node.n
                raise ValueError("Unsupported expression")

            return {"ok": True, "result": str(safe_eval(tree))}
        except Exception:
            return {"ok": False, "result": "تعذر تنفيذ الحساب."}

    def _memory_tool(self, query):
        if self.chat is None:
            return {"ok": False, "result": "لا توجد ذاكرة مفعلة."}
        results = self.chat.memory_manager.retrieve(query, limit=3)
        if not results:
            return {"ok": False, "result": "لا توجد ذاكرة مناسبة في السجل."}
        return {"ok": True, "result": "; ".join(f"{x['topic']}: {x['fact']}" for x in results)}

    def _local_search_tool(self, query):
        if self.chat is None:
            return {"ok": False, "result": "لا توجد بيانات محلية."}
        items = self.chat._retrieve_local_context(query, limit=2)
        return {"ok": True, "result": "\n".join(items) if items else "لا توجد نتائج محلية."}

    def _web_lookup_tool(self, query):
        if self.chat is None:
            return {"ok": False, "result": "بحث الويب غير مفعّل."}
        items = self.chat._retrieve_web_context(query, limit=2)
        return {"ok": True, "result": "\n".join(items) if items else "لا توجد نتائج في الويب."}

    def detect_tools(self, query):
        lower_q = query.lower()
        detected = []
        if any(word in lower_q for word in ["احسب", "calculate", "sum", "جمع", "+", "-", "*", "/"]):
            detected.append("calculator")
        if any(word in lower_q for word in ["تذكر", "ذكر", "remember", "memory", "حفظ"]):
            detected.append("memory_lookup")
        if any(word in lower_q for word in ["بحث", "search", "معلومة", "تعرف", "معلومات"]):
            detected.append("local_search")
        if any(word in lower_q for word in ["ويب", "web", "wiki", "wikipedia", "موقع"]):
            detected.append("web_lookup")
        return detected

    def execute(self, query):
        results = []
        for name in self.detect_tools(query):
            tool = self.tools.get(name)
            if tool is not None:
                results.append({"tool": name, "result": tool["fn"](query)})
        return results


class SafetyFilter:
    """فلتر سلامة يمنع الطلبات الضارة أو غير الآمنة."""
    BLOCKED_PATTERNS = [
        "كيف أقتل",
        "قتل",
        "تخريب",
        "اختراق",
        "تفجير",
        "أسلحة",
        "كسر نظام",
        "احصل على كلمة مرور",
        "إعادة كتابة malware",
        "malware",
        "ransomware",
    ]

    def review(self, response):
        lowered = response.lower()
        for pattern in self.BLOCKED_PATTERNS:
            if pattern in lowered:
                return {"safe": False, "reason": "هذا الطلب يتضمن محظوراً أمنياً ويجب رفضه."}
        return {"safe": True, "reason": "السؤال آمن ومقبول."}

    def apply_refusal(self, response):
        if not response:
            return "لا يمكنني تنفيذ طلبات ضارة أو غير آمنة. أستطيع مساعدتك في أمور آمنة ومفيدة."
        safe_refusal = "لا يمكنني المساعدة في طلبات ضارة أو غير آمنة. أستطيع مساعدتك في مواضيع آمنة مثل التعليم، التحليل، والبرمجة الآمنة."
        return safe_refusal


class NavaChatInterface:
    arabic_diacritics = re.compile(r'[\u064B-\u0652\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7-\u06E8\u06EA-\u06ED]')

    def remove_diacritics(self, text):
        return self.arabic_diacritics.sub('', text)

    def __init__(self):
        self.config, self.device = apply_hardware_limits()
        self.agent_cfg = get_agent_runtime_settings()
        self.system_prompt = get_system_prompt()
        self.model_dir = PATHS["seed_brain"]
        self.adapter_dir = PATHS["nava_tuned"]
        self.architecture = NAVAArchitecture()
        self.architecture.state.current_model = self.model_dir
        self.architecture.state.active_experts = ["general"]
        self.architecture.state.current_version = "v1.0"
        self.architecture.register_tool("calculator", allowed=True, scope="safe")
        self.architecture.register_tool("memory_lookup", allowed=True, scope="local")
        self.architecture.register_tool("local_search", allowed=True, scope="local")
        self.architecture.register_tool("web_lookup", allowed=False, scope="restricted")
        self.architecture_runtime = NAVAAgentRuntime(self.architecture)
        self.memory_manager = MemoryManager(PATHS["memory_store"])
        self.tool_registry = ToolRegistry(self)
        self.safety_filter = SafetyFilter()
        self.messages = []
        self.history_max_size = self.config.get("quality_control", {}).get("max_history_turns", 10)

        self.model = None
        self.tokenizer = None
        self._load_runtime_model()

    def _load_runtime_model(self):
        seed_ok = os.path.isdir(self.model_dir) and os.path.isfile(os.path.join(self.model_dir, "config.json"))
        adapter_ok = os.path.isdir(self.adapter_dir) and os.path.isfile(os.path.join(self.adapter_dir, "adapter_config.json"))
        if not seed_ok:
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        if not os.path.isfile(os.path.join(self.model_dir, "tokenizer.json")) and not os.path.isfile(os.path.join(self.model_dir, "tokenizer_config.json")):
            raise FileNotFoundError(f"Tokenizer files missing in {self.model_dir}")

        for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            if required not in os.listdir(self.model_dir):
                raise FileNotFoundError(f"Missing {required} in {self.model_dir}")

        has_real_model = any(
            os.path.isfile(os.path.join(self.model_dir, name)) and os.path.getsize(os.path.join(self.model_dir, name)) > 0
            for name in ("model.safetensors", "pytorch_model.bin")
        ) or any(
            os.path.isfile(path) and os.path.getsize(path) > 0
            for path in (self.model_dir + "/" + p for p in os.listdir(self.model_dir) if p.startswith("model-") and p.endswith(".safetensors"))
        )

        if not has_real_model:
            raise FileNotFoundError(f"No real Qwen2 weights found in {self.model_dir}. Needs a real base model, not a placeholder.")

        console.print("[yellow][*] جاري تحميل عقل NAVA الحقيقي من النموذج الأساسي و adapter ...[/yellow]")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        precision_str = self.config.get("gpu_settings", {}).get("precision", "float16")
        torch_dtype = torch.float16 if precision_str == "float16" else torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=self.config.get("cpu_settings", {}).get("low_cpu_mem_usage", True)
        )

        has_real_adapter = any(
            os.path.isfile(os.path.join(self.adapter_dir, name)) and os.path.getsize(os.path.join(self.adapter_dir, name)) > 0
            for name in ("adapter_model.safetensors", "adapter_model.bin")
        ) or any(
            os.path.isfile(path) and os.path.getsize(path) > 0
            for path in (self.adapter_dir + "/" + p for p in os.listdir(self.adapter_dir) if p.startswith("adapter_model-") and p.endswith(".safetensors"))
        )

        if adapter_ok and has_real_adapter:
            self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
            console.print("[cyan][*] تم تحميل LoRA adapter الحقيقي بنجاح.[/cyan]")
        else:
            self.model = base_model
            console.print("[yellow][!] adapter غير صالح أو فارغ، سيتم التشغيل على النموذج الأساسي فقط.[/yellow]")

        self.model.eval()
        console.print("[bold green][+] NAVA جاهز ومستعد للحوار بالـ runtime الحقيقي.[/bold green]\n")

    def _trim_history(self, clean_user_input):
        """يحافظ على آخر أقسام المحادثة فقط لتقليل الضغط على السياق."""
        self.messages.append({"role": "user", "content": clean_user_input})
        if len(self.messages) > (self.history_max_size * 2):
            self.messages = self.messages[-(self.history_max_size * 2):]

    def _build_chat_payload(self):
        """يبني قائمة الرسائل كاملة مع حقن system prompt ومعالجة السجل."""
        return [{"role": "system", "content": self.system_prompt}] + self.messages

    def _score_query_overlap(self, query, text):
        query_tokens = set(self.remove_diacritics(query).lower().split())
        text_tokens = set(self.remove_diacritics(text).lower().split())
        if not query_tokens or not text_tokens:
            return 0.0
        overlap = len(query_tokens & text_tokens)
        return overlap / max(1, len(query_tokens | text_tokens))

    def _retrieve_local_context(self, query, limit=3):
        if not self.agent_cfg.get("enable_local_retrieval", True):
            return []

        context_items = []
        for file_path in [PATHS["training_queue"], PATHS["master_archive"]]:
            try:
                data = _read_json(file_path)
            except Exception:
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                instruction = str(item.get("instruction", "")).strip()
                output = str(item.get("output", "")).strip()
                combined = f"{instruction} {output}"
                score = self._score_query_overlap(query, combined)
                if score > 0.08:
                    context_items.append({"text": combined, "score": score})

        context_items = sorted(context_items, key=lambda x: x["score"], reverse=True)
        deduped = []
        seen = set()
        for item in context_items[:limit]:
            text = item["text"]
            key = self.remove_diacritics(text).lower()
            if key not in seen:
                deduped.append(text)
                seen.add(key)
        return deduped

    def _retrieve_web_context(self, query, limit=2):
        if not self.agent_cfg.get("enable_web_lookup", True):
            return []
        try:
            response = requests.get(
                "https://ar.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": query,
                    "format": "json",
                    "srlimit": 3,
                },
                timeout=self.agent_cfg.get("web_timeout_seconds", 8),
                headers={"User-Agent": "NavaAgent/1.0"},
            )
            if response.status_code != 200:
                return []
            search_results = response.json().get("query", {}).get("search", [])
            snippets = []
            for item in search_results[:limit]:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                if title and snippet:
                    snippets.append(f"{title}: {snippet}")
            return snippets
        except Exception:
            return []

    def _retrieve_memory_context(self, query, limit=3):
        if not self.agent_cfg.get("memory_enabled", True):
            return []
        memory = self.memory_manager.retrieve(query, limit=limit)
        return [f"{item['topic']}: {item['fact']}" for item in memory]

    def _execute_tools(self, query):
        """أدوات أساسية للتفكير المستقل: حساب، بحث محلي، ذاكرة، في بعض الحالات بحث ويب."""
        tool_results = self.tool_registry.execute(query)
        if not tool_results:
            tool_results = [{"tool": "reasoning", "result": "لا توجد أدوات مستقلة مطلوبة في هذا السؤال."}]
        return tool_results

    def _compose_agent_context(self, user_input):
        if not self.agent_cfg.get("enabled", True):
            return []

        combined = []
        combined.extend(self._retrieve_memory_context(user_input))
        combined.extend(self._retrieve_local_context(user_input))
        combined.extend(self._retrieve_web_context(user_input))

        deduped = []
        seen = set()
        for segment in combined:
            key = self.remove_diacritics(segment).lower()
            if key not in seen and segment.strip():
                deduped.append(segment.strip())
                seen.add(key)

        max_segments = self.agent_cfg.get("max_context_segments", 3)
        return deduped[:max_segments]

    def _evaluate_response(self, query, response, context):
        """تقييم ذاتي بسيط يحدد هل الإجابة مناسبة أم تحتاج إلى مراجعة."""
        if not response or len(response.strip()) < 10:
            return {"score": 0.1, "passed": False, "reason": "الرد فارغ أو قصير جداً."}

        tokens = set(self.remove_diacritics(response).lower().split())
        q_tokens = set(self.remove_diacritics(query).lower().split())
        overlap = len(tokens & q_tokens)
        answer_len = len(response.split())
        context_text = " ".join(context).lower()
        evidence_bonus = 1 if context_text and any(word in context_text for word in self.remove_diacritics(query).lower().split()) else 0

        repetition_penalty = 0.0
        if answer_len > 0:
            repetition_penalty = 0.2 if overlap > 0 and answer_len < 20 else 0.0

        score = 0.45 + min(0.4, answer_len / 80.0) + 0.15 * evidence_bonus - repetition_penalty
        score = max(0.0, min(1.0, score))
        passed = score >= self.agent_cfg.get("self_evaluation_threshold", 0.7)
        reason = "الرد مناسب ومغطي السؤال." if passed else "الرد يحتاج مراجعة أو توضيح إضافي."
        return {"score": round(score, 2), "passed": passed, "reason": reason}

    def _save_memory_from_result(self, query, response):
        if self.agent_cfg.get("memory_enabled", True):
            topic = query.strip()[:80]
            self.memory_manager.add(topic=topic, fact=response.strip()[:250], source="assistant")

    def _generate_with_context(self, system_prompt, history_for_prompt, user_query, context_blocks):
        if context_blocks:
            context_text = "\n\n".join(context_blocks)
            user_message = (
                "استخدم السياق التالي بدقة، حلل المعلومات، واستخرج الحقيقة قبل الإجابة. "
                "إذا كان السياق غير كافٍ، قُل ذلك بوضوح ثم قدم أفضل إجابة معتمدة على المعرفة المتاحة.\n\n"
                f"السياق:\n{context_text}\n\nالسؤال:\n{user_query}"
            )
            return [{"role": "system", "content": system_prompt + " .تعمل بطريقة تحليلية واسترجاعية: ابحث في السياق، حلل، ثم أجب بدقة."}] + history_for_prompt + [{"role": "user", "content": user_message}]
        return [{"role": "system", "content": system_prompt}] + history_for_prompt + [{"role": "user", "content": user_query}]

    def _generate_answer(self, chat_payload):
        try:
            prompt = self.tokenizer.apply_chat_template(chat_payload, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = "<|im_start|>system\n" + self.system_prompt + "<|im_end|>\n"
            for msg in chat_payload:
                if msg["role"] != "system":
                    prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            prompt += "<|im_start|>assistant\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        max_seq_length = self.config.get("model_generation", {}).get("max_seq_length", 512)
        required_tokens = input_length + max_seq_length + 128
        self.config["model_generation"]["context_window"] = auto_expand_context_if_needed(required_tokens)
        context_window = self.config.get("model_generation", {}).get("context_window", 2048)

        if input_length > context_window - max_seq_length:
            compact_history = [{"role": "user", "content": chat_payload[-1]["content"]}] if chat_payload else []
            compact_payload = [{"role": "system", "content": self.system_prompt}] + compact_history
            prompt = self.tokenizer.apply_chat_template(compact_payload, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            input_length = inputs.input_ids.shape[1]

        gen_cfg = self.config.get("model_generation", {})
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_seq_length", 512),
                do_sample=gen_cfg.get("do_sample", True),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                repetition_penalty=gen_cfg.get("repetition_penalty", 1.1),
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        for tag in ["<|im_end|>", "<|im_start|>", "system", "user", "assistant"]:
            if response.startswith(tag):
                response = response.replace(tag, "").strip()
        return response

    def _runtime_generate_answer(self, user_input, intent_info, expert, memory_context, tool_results):
        clean_user_input = self.remove_diacritics(user_input)
        self._trim_history(clean_user_input)
        history_for_prompt = self.messages[:-1] if self.messages and self.messages[-1]["role"] == "user" else self.messages

        local_context = self._retrieve_local_context(clean_user_input)
        web_context = self._retrieve_web_context(clean_user_input)
        context_blocks = [f"{item['topic']}: {item['fact']}" for item in memory_context]
        context_blocks.extend(local_context)
        context_blocks.extend(web_context)

        tool_text = "\n".join(f"{item['tool']}: {item['result']}" for item in tool_results)
        if tool_text and tool_text.strip():
            context_blocks.append(tool_text)

        chat_payload = self._generate_with_context(self.system_prompt, history_for_prompt, clean_user_input, context_blocks)
        return self._generate_answer(chat_payload)

    def run_agent_loop(self, user_input):
        clean_user_input = self.remove_diacritics(user_input)
        self.architecture.event_bus.emit("TASK_CREATED", {"query": clean_user_input})

        runtime_result = self.architecture_runtime.execute(
            clean_user_input,
            memory_store=self.memory_manager,
            tool_registry=self.tool_registry,
            safety_guard=self.safety_filter,
            generator=self._runtime_generate_answer,
        )
        response = runtime_result["response"]
        context_blocks = runtime_result.get("memory_context", [])
        tool_results = runtime_result.get("tool_results", [])

        history_for_prompt = self.messages[:-1] if self.messages and self.messages[-1]["role"] == "user" else self.messages

        evaluation = None
        if self.agent_cfg.get("self_evaluation_enabled", True):
            evaluation = self._evaluate_response(clean_user_input, response, context_blocks)
            if not evaluation["passed"]:
                critique_prompt = (
                    "راجع إجابتك بعناية، وحاول تحسينها: "
                    "- اجعلها أكثر دقة، واضحة، ومباشرة. "
                    "- إذا كان السؤال يتطلب حساب أو استرجاع أو تحليل، استخدم الأدلة المتاحة. "
                    "- تجنب التكرار والردود الغامضة."
                )
                critique_payload = [{"role": "system", "content": self.system_prompt + " .أعد النظر في ردك، وقم بتحسينه لتحسين الدقة والوضوح."}] + history_for_prompt + [{"role": "user", "content": critique_prompt + "\n\nالسؤال:\n" + clean_user_input}]
                response = self._generate_answer(critique_payload)
                evaluation = self._evaluate_response(clean_user_input, response, context_blocks)

        safety = self.safety_filter.review(response)
        if not safety["safe"]:
            response = self.safety_filter.apply_refusal(response)

        self.architecture.record_memory(response.strip()[:250], source="assistant", importance=0.7, confidence=0.85)
        self.architecture.state.active_tasks = [clean_user_input[:80]]
        self.architecture.state.system_health = "healthy"
        self.memory_manager.add(topic=clean_user_input[:80], fact=response.strip()[:250], source="assistant")
        self.messages.append({"role": "assistant", "content": response})
        return response

    def generate_response(self, user_input):
        return self.run_agent_loop(user_input)


def main():
    console.clear()
    console.print(Panel.fit(
        "[bold cyan]NAVA Terminal Chat (Real Model Runtime) - واجهة المحادثة المباشرة[/bold cyan]\n\n"
        "[white]أوامر: [/white][yellow]/clear[/yellow] | [yellow]/status[/yellow] | [yellow]/correct[/yellow] | [yellow]/teach[/yellow] | [yellow]/exit[/yellow]",
        border_style="cyan"
    ))

    chat_system = NavaChatInterface()

    while True:
        try:
            user_input = Prompt.ask("\n[bold yellow]نواف[/bold yellow]")

            if not user_input.strip():
                continue

            if user_input.lower() == "/exit":
                console.print("[cyan]في أمان الله يا نواف. تم إغلاق جلسة المحادثة.[/cyan]")
                break

            elif user_input.lower() == "/clear":
                console.clear()
                console.print("[green][+] تم مسح الشاشة وسجل المحادثة اللحظي.[/green]")
                chat_system.messages = []
                continue

            elif user_input.lower() == "/correct":
                if not chat_system.messages or len(chat_system.messages) < 2:
                    console.print("[red][!] لا يوجد سجل محادثة كافي لتصحيحه.[/red]")
                    continue
                last_answer = chat_system.messages[-1]["content"]
                last_question = chat_system.messages[-2]["content"]
                console.print(f"[yellow]السؤال الأخير:[/yellow] {last_question}")
                console.print(f"[red]الجواب الخاطئ:[/red] {last_answer}")

                correct_answer = Prompt.ask("[green]أدخل الجواب الصحيح[/green]")
                if not correct_answer.strip():
                    console.print("[red][!] لا يمكن حفظ تصحيح فارغ.[/red]")
                    continue

                receive_from_list("Chat_Correction", last_question, correct_answer)
                console.print(f"[green][+] تم إرسال التصحيح لجلسة البيانات. سيتم تدريب NAVA عليه قريباً.[/green]")
                continue

            elif user_input.lower() == "/teach":
                console.print("[yellow]أدخل السؤال أو المعلومة التي تريد أن يتعلمها NAVA:[/yellow]")
                instruction = Prompt.ask("[bold yellow]السؤال/المعلومة[/bold yellow]")
                if not instruction.strip():
                    console.print("[red][!] لا يمكن حفظ معلومة فارغة.[/red]")
                    continue

                console.print("[yellow]أدخل الرد الصحيح أو المعلومة التي يجب أن يحفظها NAVA:[/yellow]")
                output = Prompt.ask("[bold green]الرد/المعلومة[/bold green]")
                if not output.strip():
                    console.print("[red][!] لا يمكن حفظ رد فارغ.[/red]")
                    continue

                receive_from_list("Chat_Teaching", chat_system.remove_diacritics(instruction), chat_system.remove_diacritics(output))
                console.print(f"[green][+] تم إرسال المعلومة لجلسة البيانات. سيتم تدريب NAVA عليها قريباً.[/green]")
                continue

            elif user_input.lower() == "/status":
                cfg = chat_system.config
                gen_cfg = cfg.get("model_generation", {})
                status_text = (
                    f"مسار العقل الأساسي (Model Path): [cyan]{chat_system.model_dir}[/cyan]\n"
                    f"مسار العقل المطور (Adapter Path): [cyan]{chat_system.adapter_dir}[/cyan]\n"
                    f"الجهاز المستخدم (Device): [cyan]{chat_system.device.upper()}[/cyan]\n"
                    f"دقة المعالجة (Precision): [cyan]{cfg.get('gpu_settings', {}).get('precision', 'float16')}[/cyan]\n"
                    f"حجم سجل المحادثة (Messages Size): [cyan]{len(chat_system.messages)}[/cyan] items\n"
                    f"الحد الأقصى للرد (Max New Tokens): [cyan]{gen_cfg.get('max_seq_length', 512)}[/cyan] Tokens\n"
                    f"درجة الحرارة (Temperature): [cyan]{gen_cfg.get('temperature', 0.7)}[/cyan]"
                )
                console.print(Panel(
                    status_text,
                    title="[bold yellow]حالة النظام والإعدادات (Real Runtime)[/bold yellow]",
                    border_style="yellow"
                ))
                continue

            with console.status("[bold green]جاري التفكير والتوليد على النموذج الحقيقي...[/bold green]", spinner="dots"):
                reply = chat_system.generate_response(user_input)

            console.print(Panel(reply, title="[bold green]NAVA[/bold green]", border_style="green"))

        except KeyboardInterrupt:
            console.print("\n[cyan]تم إنهاء الجلسة.[/cyan]")
            break
        except Exception as e:
            console.print(f"[red][!] حدث خطأ أثناء المعالجة: {e}[/red]")


if __name__ == "__main__":
    main()

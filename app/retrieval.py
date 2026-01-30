# app/retrieval.py
import json
import os
import re
import unicodedata
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Stopwords básicos (puedes mover a archivo y cargarlo)
SPANISH_STOPWORDS: set[str] = {
    "a","al","algo","algunas","algunos","ante","antes","como","con","contra","cual",
    "cuando","de","del","desde","donde","dos","el","ella","ellas","ellos","en","entre",
    "era","erais","eran","eras","eres","es","esa","esas","ese","eso","esos","esta",
    "estaba","estaban","estado","estais","estamos","estan","estar","este","esto","estos",
    "estoy","fin","fue","fueron","fui","ha","hace","hacen","hacer","haces","hago","han",
    "hasta","hay","la","las","le","les","lo","los","mas","mi","mis","mucho","muy","nada",
    "ni","no","nos","nosotros","o","otra","otras","otro","otros","para","pero","poco",
    "por","porque","que","se","sea","segun","ser","si","siempre","sin","sobre","sois",
    "solamente","solo","somos","son","soy","su","sus","tambien","tanto","te","teneis",
    "tenemos","tener","tengo","ti","tiene","tienen","todo","tras","tu","tus","un","una",
    "uno","unos","y","ya"
}

# Sinónimos ligeros para normalizar variantes (extiende según tu dominio)
SYNONYMS = {
    "deploy": "despliegue",
    "deployment": "despliegue",
    "release": "despliegue",
    "prod": "produccion",
    "production": "produccion",
    "pipeline": "cicd",
    "ci/cd": "cicd",
    "ci-cd": "cicd",
    "pod": "k8s_pod",
    "pods": "k8s_pod",
    "api": "servicio_api",
    "errores": "error",
}


def _normalize(text: str) -> str:
    """
    Normaliza a minúsculas y elimina acentos/diacríticos.
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def _apply_synonyms(tokens: List[str]) -> List[str]:
    return [SYNONYMS.get(t, t) for t in tokens]


def _tokenize(text: str) -> List[str]:
    """
    Tokenizador robusto para BM25:
    - Normaliza (lower + sin acentos)
    - Elimina puntuación
    - Mantiene \w
    - Quita stopwords
    - Aplica sinónimos
    """
    text = _normalize(text)
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = re.findall(r"\w+", text)
    if SPANISH_STOPWORDS:
        tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
    tokens = _apply_synonyms(tokens)
    # Normaliza números si te conviene (opcional):
    # tokens = ["<NUM>" if t.isdigit() else t for t in tokens]
    return tokens


class BM25Index:
    def __init__(self, docs: List[Dict[str, Any]], question_weight: int = 3):
        self.question_weight = max(1, int(question_weight))
        self.by_intent: dict[str, BM25Okapi] = {}
        self.raw_by_intent: dict[str, List[Dict[str, Any]]] = {}

        # Índices por intención
        intents = sorted({d["intent"] for d in docs if "intent" in d})
        for intent in intents:
            subset = [d for d in docs if d.get("intent") == intent]
            corpus = [self._tokens_for_doc(d) for d in subset]
            self.by_intent[intent] = BM25Okapi(corpus)
            self.raw_by_intent[intent] = subset

        # Índice global
        corpus_all = [self._tokens_for_doc(d) for d in docs]
        self.global_bm25 = BM25Okapi(corpus_all)
        self.raw_all = docs

    def _doc_text(self, d: Dict[str, Any]) -> str:
        q = (d.get("question") or "").strip()
        a = (d.get("answer") or "").strip()
        if q:
            return (q + " ") * self.question_weight + a
        return a

    def _tokens_for_doc(self, d: Dict[str, Any]) -> List[str]:
        return _tokenize(self._doc_text(d))

    def query(self, text: str, intent: Optional[str] = None, topk: int = 3) -> List[Dict[str, Any]]:
        tokens = _tokenize(text)
        if intent and intent in self.by_intent:
            bm25 = self.by_intent[intent]
            pool = self.raw_by_intent[intent]
        else:
            bm25 = self.global_bm25
            pool = self.raw_all

        scores = bm25.get_scores(tokens)
        pairs = sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)[:topk]
        return [{"doc": d, "score": float(s)} for d, s in pairs]


def load_faq() -> List[Dict[str, Any]]:
    path = os.path.join(DATA_DIR, "faq.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
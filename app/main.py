# app/main.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from .models import train_intent_model, load_intent_pipeline
from .retrieval import load_faq, BM25Index

app = FastAPI(title="Chatbot Backend (DecisionTree/LogReg + BM25)")

# -------- Bootstrapping --------
# Entrenar solo si no existen artefactos; en prod, llámalo offline o vía endpoint
train_intent_model(force=False)
VECTORIZER, INTENT_CLF = load_intent_pipeline()
FAQ = load_faq()
BM25 = BM25Index(FAQ, question_weight=3)  # peso mayor a 'question'

# Umbrales ajustables (podrías mover a vars de entorno)
TAU_HI = 0.80
TAU_LO = 0.50
MIN_SCORE = 0.05


class AskRequest(BaseModel):
    question: str
    topk: Optional[int] = 3
    min_score: Optional[float] = None  # si None, usa MIN_SCORE global
    debug: Optional[bool] = False      # devuelve candidatos y scores


class AskResponse(BaseModel):
    intent: str
    confidence: float
    answer: str
    source_id: str
    score: float
    # Campos debug opcionales
    candidates: Optional[List[Dict[str, Any]]] = None
    used_fallback: Optional[bool] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        return AskResponse(
            intent="unknown",
            confidence=0.0,
            answer="No entendí la pregunta. ¿Puedes reformularla?",
            source_id="none",
            score=0.0,
            candidates=[] if req.debug else None,
            used_fallback=False,
        )

    # 1) Intent + confidence
    X = VECTORIZER.transform([q])
    probs = INTENT_CLF.predict_proba(X)[0]  # type: ignore[attr-defined]
    idx = int(probs.argmax())
    intent = str(INTENT_CLF.classes_[idx])
    confidence = float(probs[idx])

    # Umbrales efectivos
    topk = int(req.topk or 3)
    min_score = float(req.min_score) if req.min_score is not None else MIN_SCORE

    # 2) Retrieval con control de fallback
    used_fallback = False
    candidates = []
    top = None

    if confidence >= TAU_HI:
        # Alta confianza → primero por intención
        candidates = BM25.query(q, intent=intent, topk=topk)
        top = candidates[0] if candidates else None
        # Si score muy bajo, probar global
        if not top or top["score"] < min_score:
            used_fallback = True
            candidates = BM25.query(q, intent=None, topk=topk)
            top = candidates[0] if candidates else None

    elif confidence <= TAU_LO:
        # Baja confianza → directo a global
        used_fallback = True
        candidates = BM25.query(q, intent=None, topk=topk)
        top = candidates[0] if candidates else None

    else:
        # Zona gris → intenta intención y, si no, global
        candidates = BM25.query(q, intent=intent, topk=topk)
        top = candidates[0] if candidates else None
        if not top or top["score"] < min_score:
            used_fallback = True
            candidates = BM25.query(q, intent=None, topk=topk)
            top = candidates[0] if candidates else None

    if not top:
        return AskResponse(
            intent=intent,
            confidence=confidence,
            answer="No encontré información para esa pregunta.",
            source_id="none",
            score=0.0,
            candidates=(candidates if req.debug else None),
            used_fallback=used_fallback,
        )

    # Separación top1-top2 para detectar ambigüedad (opcional)
    sep_ok = True
    if len(candidates) >= 2:
        sep = candidates[0]["score"] - candidates[1]["score"]
        # Si la separación es muy pequeña y además el top1 es bajo → incertidumbre
        if candidates[0]["score"] < max(min_score, 0.1) and sep < 0.02:
            sep_ok = False

    doc = top["doc"]
    answer = doc.get("answer", "")
    if not sep_ok:
        answer = "La consulta es ambigua o con poca evidencia. ¿Puedes dar más contexto?"

    return AskResponse(
        intent=intent,
        confidence=confidence,
        answer=answer,
        source_id=doc.get("id", ""),
        score=float(top["score"]),
        candidates=(candidates if req.debug else None),
        used_fallback=used_fallback,
    )


# --------- Endpoints operativos (opcionales) ---------

@app.post("/reload_faq")
def reload_faq():
    """
    Recarga el FAQ y reconstruye índices BM25 sin reiniciar.
    """
    global FAQ, BM25
    FAQ = load_faq()
    BM25 = BM25Index(FAQ, question_weight=3)
    return {"status": "ok", "faq_size": len(FAQ)}


@app.post("/retrain_intents")
def retrain_intents(force: bool = True):
    """
    Reentrena pipeline de intent (offline recomendado en prod).
    """
    global VECTORIZER, INTENT_CLF
    train_intent_model(force=force)
    VECTORIZER, INTENT_CLF = load_intent_pipeline()
    return {"status": "ok", "retrained": True}
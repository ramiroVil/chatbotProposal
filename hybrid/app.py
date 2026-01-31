from fastapi import FastAPI
from pydantic import BaseModel
from search import search_hybrid
from typing import Optional


app = FastAPI(title="FAQ Bot (Postgres + pgvector)")

class ChatRequest(BaseModel):
    message: str
    language: str = "en"

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source_question: Optional[str] = None
    suggestions: list[str] = []
    source_link: Optional[str] = None

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    text = req.message.strip()
    if not text:
        return ChatResponse(answer="Escribe una pregunta para poder ayudarte.", confidence=0.0)

    results = search_hybrid(text, language=req.language)
    if not results:
        return ChatResponse(
            answer="No encontré una respuesta en la base de conocimiento. ¿Puedes dar más detalle?",
            confidence=0.0
        )

    best = results[0]
    # Umbral (ajústalo según tus pruebas)
    THRESHOLD = 0.19

    if best["final_score"] < THRESHOLD:
        return ChatResponse(
            answer="No encontré una coincidencia clara. ¿Te refieres a alguna de estas preguntas?",
            confidence=best["final_score"],
            suggestions=[r["question"] for r in results[:3]]
        )

    return ChatResponse(
        answer=best["answer"],
        confidence=best["final_score"],
        source_question=best["question"],
        source_link=best["source_link"]
    )
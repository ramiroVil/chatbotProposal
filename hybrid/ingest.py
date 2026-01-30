from db import get_conn, new_uuid
from embeddings import embed

SAMPLE_FAQS = [
  {
    "question": "No se guarda el Portal Form Assignment",
    "answer": "Verifica permisos del usuario, revisa validaciones requeridas y confirma que el servicio X esté activo. Si persiste, revisa logs del módulo Y.",
    "category": "Portal",
    "tags": ["portal", "form", "save", "assignment"],
    "language": "es",
    "source_link": "https://tu-doc-interna/portal/save"
  },
  {
    "question": "Error E1234 al asignar un formulario",
    "answer": "El error E1234 suele ocurrir por configuración inválida. Valida el campo Z y reinicia el servicio X. Si hay cola, limpia caché Y.",
    "category": "Errores",
    "tags": ["E1234", "assignment", "config"],
    "language": "es",
    "source_link": "https://tu-doc-interna/errors/e1234"
  },
]

def upsert_faq(conn, faq):
    q = """
    INSERT INTO faq_items (id, question, answer, category, tags, language, source_link, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    vector = embed(faq["question"] + " " + faq["answer"])
    conn.execute(q, (
        new_uuid(),
        faq["question"],
        faq["answer"],
        faq.get("category"),
        faq.get("tags"),
        faq.get("language", "es"),
        faq.get("source_link"),
        vector.tolist(),  # pgvector acepta lista
    ))

def main():
    conn = get_conn()
    with conn:
        for f in SAMPLE_FAQS:
            upsert_faq(conn, f)

        # ivfflat recomienda ANALYZE para performance
        conn.execute("ANALYZE faq_items;")

    conn.close()
    print("✅ FAQs cargadas con embeddings.")

if __name__ == "__main__":
    main()
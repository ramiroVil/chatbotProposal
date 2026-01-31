from db import get_conn, new_uuid
from embeddings import embed
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
path = os.path.join(DATA_DIR, "questions.json")
with open(path, "r", encoding="utf-8") as f:
        SAMPLE_FAQS = json.load(f)

print(f"SAMPLE_FAQS path: {SAMPLE_FAQS}")
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
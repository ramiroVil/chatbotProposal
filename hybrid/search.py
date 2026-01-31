from db import get_conn
from embeddings import embed

HYBRID_ALPHA = 0.55  # peso semántico
TOP_K = 5

def search_hybrid(user_text: str, language: str = "es"):
    qvec = embed(user_text).tolist()

    sql = f"""
    WITH ft AS (
      SELECT
        id,
        question,
        answer,
        category,
        tags,
        source_link,
        ts_rank_cd(
          to_tsvector('spanish', question || ' ' || answer),
          plainto_tsquery('spanish', %s)
        ) AS ft_score
      FROM faq_items
      WHERE is_active = TRUE
        AND language = %s
        AND to_tsvector('spanish', question || ' ' || answer) @@ plainto_tsquery('spanish', %s)
      ORDER BY ft_score DESC
      LIMIT {TOP_K}
    ),
    vs AS (
      SELECT
        id,
        question,
        answer,
        category,
        tags,
        source_link,
        (1 - (embedding <=> %s::vector)) AS v_score
      FROM faq_items
      WHERE is_active = TRUE
        AND language = %s
      ORDER BY embedding <-> %s::vector
      LIMIT {TOP_K}
    ),
    merged AS (
      SELECT
        COALESCE(ft.id, vs.id) AS id,
        COALESCE(ft.question, vs.question) AS question,
        COALESCE(ft.answer, vs.answer) AS answer,
        COALESCE(ft.category, vs.category) AS category,
        COALESCE(ft.tags, vs.tags) AS tags,
        COALESCE(ft.source_link, vs.source_link) AS source_link,
        COALESCE(ft.ft_score, 0) AS ft_score,
        COALESCE(vs.v_score, 0) AS v_score
      FROM ft
      FULL OUTER JOIN vs ON ft.id = vs.id
    )
    SELECT *,
      ({HYBRID_ALPHA} * v_score + (1 - {HYBRID_ALPHA}) * ft_score) AS final_score
    FROM merged
    ORDER BY final_score DESC
    LIMIT {TOP_K};
    """

    conn = get_conn()
    with conn:
        rows = conn.execute(sql, (user_text, language, user_text, qvec, language, qvec)).fetchall()
    conn.close()

    results = []
    for r in rows:
        results.append({
            "id": str(r[0]),
            "question": r[1],
            "answer": r[2],
            "category": r[3],
            "tags": r[4],
            "source_link": r[5],
            "ft_score": float(r[6]),
            "v_score": float(r[7]),
            "final_score": float(r[8]),
        })
    return results
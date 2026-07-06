"""
Standalone pgvector match test — run directly with:
    python test_match_standalone.py

Bypasses FastAPI/pipeline entirely. Pulls a real stored face embedding
from the DB and matches it against itself. This SHOULD return similarity
very close to 1.0 (matching an embedding against itself is a perfect
sanity check). If this also returns nothing/null, the bug is in the SQL
or connection — not in the pipeline wiring. If THIS works but the API
still fails, the bug is specifically in how main.py/pipeline.py calls it.
"""

import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import numpy as np

DSN = "postgresql://postgres:Quotus%401234@localhost:5432/smart_city"

conn = psycopg2.connect(DSN)
conn.autocommit = True
register_vector(conn)

with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
    # 1. Pull one real stored embedding directly
    cur.execute("SELECT person_id, embedding FROM face_embeddings LIMIT 1")
    row = cur.fetchone()
    print("Fetched row person_id:", row["person_id"] if row else None)

    if row is None:
        print("!!! face_embeddings returned NO rows on a plain SELECT — table is empty from this connection's view.")
    else:
        test_embedding = np.asarray(row["embedding"], dtype=np.float32)
        print("Embedding length:", len(test_embedding))
        print("Embedding dtype:", test_embedding.dtype)

        # 2. Run the EXACT same match query as PersonRegistry._match
        cur.execute(
            """
            SELECT p.person_id, p.name,
                   1 - (e.embedding <=> %s) AS similarity
            FROM face_embeddings e
            JOIN persons p ON p.person_id = e.person_id
            ORDER BY e.embedding <=> %s
            LIMIT 1
            """,
            (test_embedding, test_embedding),
        )
        match_row = cur.fetchone()
        print("\n--- Match query result (self-comparison) ---")
        print(match_row)

        if match_row is None:
            print("!!! Query returned NO rows even for self-comparison. Bug is in the SQL/JOIN/connection.")
        else:
            print(f"Similarity: {match_row['similarity']}  (should be ~1.0)")

conn.close()
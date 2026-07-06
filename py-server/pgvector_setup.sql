-- =============================================================
-- Smart City Platform — pgvector schema setup
-- Run this ONCE against your target database (e.g. smart_city)
-- in pgAdmin's Query Tool or via psql:
--   psql -U postgres -h localhost -d smart_city -f pgvector_setup.sql
-- =============================================================

-- 1. Enable the pgvector extension
--    Requires (PostgreSQL 17 server):
--      sudo apt-get install -y postgresql-17-pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Person identity table
CREATE TABLE IF NOT EXISTS persons (
    person_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL
);

-- 3. Face embeddings  (ArcFace, 512-d)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL
);

-- 4. Body embeddings  (OSNet Re-ID, 512-d)
CREATE TABLE IF NOT EXISTS body_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL
);

-- 5. IVFFlat indexes for fast cosine similarity search
--    NOTE: IVFFlat needs at least a few hundred rows for meaningful benefit.
--    Re-run REINDEX once you have hundreds of embeddings for best recall.
CREATE INDEX IF NOT EXISTS idx_face_embeddings_vec
    ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_body_embeddings_vec
    ON body_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

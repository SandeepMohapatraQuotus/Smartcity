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
    name        TEXT NOT NULL,
    image_url   TEXT   -- primary / display photo (kept for backward compat with alerts)
);

-- Add image_url column to existing deployments (safe to run multiple times)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='persons' AND column_name='image_url'
    ) THEN
        ALTER TABLE persons ADD COLUMN image_url TEXT;
    END IF;
END$$;

-- 3. Multiple reference photos per person
--    position=0  → primary display photo (same URL stored in persons.image_url)
--    position>0  → additional reference shots
--    The pipeline always sends only persons.image_url (position=0) in alerts —
--    the extra rows are used purely by the registry UI and future analytics.
CREATE TABLE IF NOT EXISTS person_images (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    image_url   TEXT NOT NULL,
    position    INT  NOT NULL DEFAULT 0
);

-- Prevent duplicate URLs per person (safe to re-run)
CREATE UNIQUE INDEX IF NOT EXISTS uq_person_images_pid_url
    ON person_images (person_id, image_url);

-- Safe migration: create person_images if it doesn't exist on existing deployments
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_name = 'person_images'
    ) THEN
        CREATE TABLE person_images (
            id          SERIAL PRIMARY KEY,
            person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
            image_url   TEXT NOT NULL,
            position    INT  NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX uq_person_images_pid_url
            ON person_images (person_id, image_url);
    END IF;
END$$;

-- 4. Face embeddings  (ArcFace, 512-d)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL
);

-- 5. Body embeddings  (OSNet Re-ID, 512-d)
CREATE TABLE IF NOT EXISTS body_embeddings (
    id          SERIAL PRIMARY KEY,
    person_id   TEXT NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding   vector(512) NOT NULL
);

-- 6. IVFFlat indexes for fast cosine similarity search
--    NOTE: IVFFlat needs at least a few hundred rows for meaningful benefit.
--    Re-run REINDEX once you have hundreds of embeddings for best recall.
CREATE INDEX IF NOT EXISTS idx_face_embeddings_vec
    ON face_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_body_embeddings_vec
    ON body_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS item_embeddings (
    item_type text NOT NULL CHECK (item_type IN ('lost', 'found')),
    item_id text NOT NULL,
    embedding vector(896) NOT NULL,
    PRIMARY KEY (item_type, item_id)
);
CREATE INDEX IF NOT EXISTS found_embeddings_hnsw
    ON item_embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE item_type = 'found';

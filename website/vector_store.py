"""Persistent, vector-only similarity index (report details remain in MongoDB)."""
import math
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg

DIMENSIONS = 896  # MiniLM (384) + CLIP ViT-B/32 (512)


def vector_literal(values):
    values = [float(value) for value in values]
    if len(values) != DIMENSIONS or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Expected {DIMENSIONS} finite embedding values')
    if not any(values):
        raise ValueError('Cannot index a zero vector')
    return '[' + ','.join(str(value) for value in values) + ']'


class VectorStore:
    @contextmanager
    def connection(self):
        url = os.getenv('VECTOR_DATABASE_URL')
        if not url:
            raise RuntimeError('VECTOR_DATABASE_URL is required for similarity search')
        with psycopg.connect(url) as connection:
            yield connection

    def initialize(self):
        with self.connection() as connection:
            connection.execute(Path(__file__).with_name('vector_schema.sql').read_text())

    def upsert(self, item_type, item_id, embedding):
        with self.connection() as connection:
            connection.execute('''
                INSERT INTO item_embeddings (item_type, item_id, embedding)
                VALUES (%s, %s, %s::vector)
                ON CONFLICT (item_type, item_id) DO UPDATE
                SET embedding = EXCLUDED.embedding
            ''', (item_type, str(item_id), vector_literal(embedding)))

    def get(self, item_type, item_id):
        with self.connection() as connection:
            row = connection.execute('''
                SELECT embedding::text FROM item_embeddings
                WHERE item_type = %s AND item_id = %s
            ''', (item_type, str(item_id))).fetchone()
        return row[0] if row else None

    def delete(self, item_type, item_id):
        with self.connection() as connection:
            connection.execute('DELETE FROM item_embeddings WHERE item_type = %s AND item_id = %s',
                               (item_type, str(item_id)))

    def search_found(self, embedding, limit=50):
        # Keep ORDER BY on the distance operator so PostgreSQL can use HNSW.
        with self.connection() as connection:
            connection.execute('SET LOCAL hnsw.ef_search = 200')
            return connection.execute('''
                SELECT item_id, 1 - (embedding <=> %s::vector) AS score
                FROM item_embeddings WHERE item_type = 'found'
                ORDER BY embedding <=> %s::vector LIMIT %s
            ''', (embedding, embedding, limit)).fetchall()

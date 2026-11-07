"""Encode on writes; use stored embeddings and HNSW on reads."""
from functools import lru_cache
import math
from bson import ObjectId
from .vector_store import VectorStore

store = VectorStore()


@lru_cache(maxsize=1)
def encoder():
    from .similarity_service import similarity_service
    return similarity_service


def combine_embeddings(text, image):
    """Equal-weight normalized modalities; absent modalities occupy zero blocks."""
    blocks = []
    for embedding, size in ((text, 384), (image, 512)):
        if embedding is None:
            blocks.extend([0.0] * size)
            continue
        values = [float(v) for v in embedding]
        norm = math.sqrt(sum(v * v for v in values))
        if len(values) != size or not math.isfinite(norm) or norm == 0:
            raise ValueError('Invalid embedding or incompatible model dimension')
        blocks.extend(v / norm for v in values)
    norm = math.sqrt(sum(v * v for v in blocks))
    if norm == 0:
        raise ValueError('An item needs text or an image to be indexed')
    return [v / norm for v in blocks]


def index_item(item_type, item_id, item):
    model = encoder()
    text = model.create_structured_text(item)
    text_vector = model.get_text_embedding(text) if text else None
    image_vector = model.get_image_embedding(item['imageUrl']) if item.get('imageUrl') else None
    # Do not silently replace a complete index with a failed/partial encoding.
    if (text and text_vector is None) or (item.get('imageUrl') and image_vector is None):
        raise RuntimeError('Embedding generation failed; retry with backfill-vectors')
    store.upsert(item_type, item_id, combine_embeddings(text_vector, image_vector))


def find_matches(db, lost_item_id, top_k=10, min_score=0.5):
    embedding = store.get('lost', lost_item_id)
    if embedding is None:
        raise RuntimeError('Item is not indexed; run flask backfill-vectors')
    candidates = store.search_found(embedding, limit=max(100, top_k * 10))
    if not candidates:
        return []
    # Fetch only candidate reports, never the entire found-item collection.
    ids = [ObjectId(item_id) for item_id, _ in candidates]
    items = {str(item['_id']): item for item in db.foundItems.find(
        {'_id': {'$in': ids}, 'status': 'active'})}
    return [(items[item_id], float(score)) for item_id, score in candidates
            if item_id in items and score >= min_score][:top_k]


def register_commands(app):
    import click

    @app.cli.command('init-vector-db')
    def initialize():
        """Create the pgvector table and HNSW index."""
        store.initialize()
        click.echo('Vector database initialized.')

    @app.cli.command('backfill-vectors')
    @click.option('--force', is_flag=True, help='Re-encode existing vectors after model changes.')
    def backfill(force):
        """Index existing active reports; safe to rerun after failures."""
        failures = 0
        for kind, collection in (('lost', app.db.lostItems), ('found', app.db.foundItems)):
            for item in collection.find({'status': 'active'}):
                try:
                    if force or store.get(kind, item['_id']) is None:
                        index_item(kind, item['_id'], item)
                        click.echo(f'Indexed {kind} {item["_id"]}')
                except Exception as error:
                    failures += 1
                    click.echo(f'Failed {kind} {item["_id"]}: {error}', err=True)
        if failures:
            raise click.ClickException(f'{failures} reports failed; fix and rerun.')

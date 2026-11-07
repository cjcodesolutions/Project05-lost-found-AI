"""Dependency-free unit checks; database integration requires PostgreSQL/pgvector."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]


def load_modules():
    package = types.ModuleType('website')
    package.__path__ = [str(ROOT / 'website')]
    modules = {'website': package, 'psycopg': MagicMock(), 'bson': MagicMock()}
    modules['bson'].ObjectId.side_effect = lambda value: value
    with patch.dict(sys.modules, modules):
        loaded = []
        for name in ('vector_store', 'indexed_similarity'):
            spec = importlib.util.spec_from_file_location('website.' + name, ROOT / 'website' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
    return loaded


vectors, similarity = load_modules()


class VectorSearchTests(unittest.TestCase):
    def test_modalities_are_normalized_and_missing_images_are_supported(self):
        embedding = similarity.combine_embeddings([2] + [0] * 383, [3] + [0] * 511)
        self.assertAlmostEqual(sum(v * v for v in embedding), 1)
        self.assertAlmostEqual(embedding[0], embedding[384])
        text_only = similarity.combine_embeddings([2] + [0] * 383, None)
        self.assertEqual(text_only[0], 1)
        self.assertFalse(any(text_only[384:]))

    def test_invalid_vectors_are_rejected(self):
        for value in ([0] * 896, [1] * 895, [float('nan')] * 896):
            with self.assertRaises(ValueError):
                vectors.vector_literal(value)
        with self.assertRaises(ValueError):
            similarity.combine_embeddings(None, None)

    def test_search_uses_distance_order_and_bound_parameters(self):
        store = vectors.VectorStore()
        connection = MagicMock()
        connection.execute.return_value.fetchall.return_value = [('abc', 0.9)]
        with patch.object(store, 'connection') as connect:
            connect.return_value.__enter__.return_value = connection
            result = store.search_found('[1,0]', limit=100)
        sql, params = connection.execute.call_args.args
        self.assertIn('ORDER BY embedding <=> %s::vector LIMIT %s', sql)
        self.assertIn("WHERE item_type = 'found'", sql)
        self.assertEqual(params, ('[1,0]', '[1,0]', 100))
        self.assertEqual(result, [('abc', 0.9)])

    def test_results_fetch_only_candidates_and_drop_deleted_reports(self):
        db = MagicMock()
        db.foundItems.find.return_value = [{'_id': 'b'}]
        with patch.object(similarity, 'store') as store:
            store.get.return_value = '[1,0]'
            store.search_found.return_value = [('deleted', 0.99), ('b', 0.8), ('low', 0.1)]
            result = similarity.find_matches(db, 'lost-id')
        self.assertEqual(result, [({'_id': 'b'}, 0.8)])
        db.foundItems.find.assert_called_once_with(
            {'_id': {'$in': ['deleted', 'b', 'low']}, 'status': 'active'})

    def test_failed_image_encoding_does_not_save_partial_vector(self):
        with patch.object(similarity, 'encoder') as encoder, patch.object(similarity, 'store') as store:
            encoder.return_value.create_structured_text.return_value = 'wallet'
            encoder.return_value.get_text_embedding.return_value = [1] * 384
            encoder.return_value.get_image_embedding.return_value = None
            with self.assertRaises(RuntimeError):
                similarity.index_item('found', 'id', {'imageUrl': 'image'})
            store.upsert.assert_not_called()


if __name__ == '__main__':
    unittest.main()

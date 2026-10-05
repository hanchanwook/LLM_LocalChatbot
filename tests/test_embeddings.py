import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import embeddings


class EmbeddingTests(unittest.TestCase):
    def test_invalid_vectors_are_rejected(self):
        for vector in ([], [0.0] * 1024, [float("nan")] * 1024, [True] * 1024):
            with self.assertRaises(embeddings.EmbeddingError):
                embeddings.normalize(vector)

    def test_batch_validation_and_query_instruction(self):
        with patch("embeddings.request_json", return_value={"embeddings": [[1.0] * 1024]}) as request:
            vectors = embeddings.embed(["연차 신청"], query=True)
        self.assertAlmostEqual(sum(v * v for v in vectors[0]), 1)
        payload = request.call_args.args[1]
        self.assertTrue(payload["input"][0].startswith("Instruct:"))
        self.assertIn("연차 신청", payload["input"][0])
        self.assertFalse(payload["truncate"])
        with patch("embeddings.request_json", return_value={"embeddings": []}):
            with self.assertRaises(embeddings.EmbeddingError):
                embeddings.embed(["text"])

    def test_model_version_part_of_key(self):
        with patch("embeddings.request_json", return_value={"models": [
            {"name": embeddings.MODEL, "digest": "abc123"}]}):
            self.assertIn("abc123", embeddings.model_key())
        with patch("embeddings.request_json", return_value={"models": []}):
            with self.assertRaises(embeddings.EmbeddingError):
                embeddings.model_key()

import os
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psycopg
import database
import documents
import embeddings
from streamlit.testing.v1 import AppTest


@unittest.skipUnless(os.environ.get("TEST_DB_PORT"), "Requires an isolated PostgreSQL")
class DocumentManagementTests(unittest.TestCase):
    def setUp(self):
        self.connection = psycopg.connect(host="127.0.0.1", port=os.environ["TEST_DB_PORT"],
                                         user="postgres", dbname="postgres", autocommit=True)
        database.initialize(self.connection)
        documents.initialize(self.connection)
        self.name = f"manage-{uuid.uuid4().hex}.txt"
        self.original = f"original document {uuid.uuid4().hex}".encode()
        documents.upload(self.connection, self.name, self.original)
        self.doc_id = next(i for i, name in documents.list_documents(self.connection) if name == self.name)
        self.key_patch = patch("embeddings.model_key", return_value="management-test-model")
        self.vector_patch = patch("embeddings.embed", side_effect=lambda texts: [[1.0] + [0.0] * 1023 for _ in texts])
        self.key_patch.start()
        self.vector_patch.start()
        documents.index_documents(self.connection, [self.doc_id])

    def tearDown(self):
        self.vector_patch.stop()
        self.key_patch.stop()
        self.connection.close()

    def test_replace_delete_vectors_and_keep_conversation(self):
        cid = database.create_conversation(self.connection, "gemma3:4b", "system")
        database.save_turn(self.connection, cid, "question", f"old source: {self.name}")
        old_chunk = self.connection.execute("SELECT id FROM agent_document_chunks WHERE document_id=%s", (self.doc_id,)).fetchone()[0]
        new_text = "새로운 규정입니다. " + uuid.uuid4().hex
        documents.replace_document(self.connection, self.doc_id, "updated.txt", new_text.encode("utf-8"))
        self.assertEqual(documents.load_for_summary(self.connection, [self.doc_id])[0]["pages"][1], new_text)
        self.assertEqual(documents.embedding_status(self.connection, [self.doc_id], "management-test-model"), (1, 1))
        self.assertIsNone(self.connection.execute("SELECT chunk_id FROM agent_chunk_embeddings WHERE chunk_id=%s", (old_chunk,)).fetchone())
        details = next(row for row in documents.document_details(self.connection, "management-test-model") if row[0] == self.doc_id)
        self.assertEqual(details[1], "updated.txt")
        self.assertGreaterEqual(details[3], details[2])
        new_chunk = self.connection.execute("SELECT id FROM agent_document_chunks WHERE document_id=%s", (self.doc_id,)).fetchone()[0]
        documents.delete_document(self.connection, self.doc_id)
        self.assertNotIn(self.doc_id, dict(documents.list_documents(self.connection)))
        self.assertIsNone(self.connection.execute("SELECT chunk_id FROM agent_chunk_embeddings WHERE chunk_id=%s", (new_chunk,)).fetchone())
        self.assertEqual(database.load_conversation(self.connection, cid)[1][-1]["content"], f"old source: {self.name}")

    def test_failed_replace_preserves_original(self):
        original = documents.load_for_summary(self.connection, [self.doc_id])
        with patch("embeddings.embed", side_effect=embeddings.EmbeddingError("offline")):
            with self.assertRaises(embeddings.EmbeddingError):
                documents.replace_document(self.connection, self.doc_id, "update.txt", uuid.uuid4().hex.encode())
        # Simulate a storage failure after the original chunks have been deleted inside a transaction.
        with patch("embeddings.embed", return_value=[[1.0]]):
            with self.assertRaises(psycopg.errors.CheckViolation):
                documents.replace_document(self.connection, self.doc_id, "update.txt", uuid.uuid4().hex.encode())
        with self.assertRaises(ValueError):
            documents.replace_document(self.connection, self.doc_id, "blank.txt", b" ")
        self.assertEqual(documents.load_for_summary(self.connection, [self.doc_id]), original)
        self.assertEqual(documents.embedding_status(self.connection, [self.doc_id], "management-test-model"), (1, 1))

    def test_duplicate_replacement_rejected(self):
        data = uuid.uuid4().hex.encode()
        documents.upload(self.connection, "duplicate.txt", data)
        with self.assertRaises(ValueError):
            documents.replace_document(self.connection, self.doc_id, "duplicate.txt", data)
        self.assertEqual(documents.load_for_summary(self.connection, [self.doc_id])[0]["pages"][1], self.original.decode())

    def test_web_manage_delete_and_selection_cleanup(self):
        os.environ.update(PGHOST="127.0.0.1", PGPORT=os.environ["TEST_DB_PORT"],
                          PGDATABASE="postgres", PGUSER="postgres", PGPASSWORD="")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/web_app.py"), default_timeout=60).run()
        self.assertIn("문서 관리", [tab.label for tab in app.tabs])
        app.multiselect[0].set_value([self.doc_id]).run()
        next(w for w in app.selectbox if w.label == "관리할 문서").set_value(self.doc_id).run()
        next(b for b in app.button if b.label == "문서 삭제").click().run()
        self.assertTrue(app.warning)
        self.assertIn(self.doc_id, dict(documents.list_documents(self.connection)))
        app.checkbox[0].set_value(True)
        next(b for b in app.button if b.label == "문서 삭제").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertEqual(app.multiselect[0].value, [])
        self.assertNotIn(self.doc_id, dict(documents.list_documents(self.connection)))

    def test_web_replace_file(self):
        import streamlit as st
        os.environ.update(PGHOST="127.0.0.1", PGPORT=os.environ["TEST_DB_PORT"],
                          PGDATABASE="postgres", PGUSER="postgres", PGPASSWORD="")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/web_app.py"), default_timeout=60).run()
        next(w for w in app.selectbox if w.label == "관리할 문서").set_value(self.doc_id).run()
        original_uploader = st.file_uploader
        content = "교체한 문서입니다. " + uuid.uuid4().hex
        file = SimpleNamespace(name="replacement.txt", getvalue=lambda: content.encode("utf-8"))
        def uploader(label, *args, **kwargs):
            if label == "교체할 파일":
                return file
            return original_uploader(label, *args, **kwargs)
        with patch("streamlit.file_uploader", side_effect=uploader):
            next(b for b in app.button if b.label == "문서 교체").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(app.success)
        self.assertEqual(dict(documents.list_documents(self.connection))[self.doc_id], "replacement.txt")
        self.assertEqual(documents.load_for_summary(self.connection, [self.doc_id])[0]["pages"][1], content)

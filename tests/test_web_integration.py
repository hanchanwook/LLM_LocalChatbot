"""Run against an isolated PostgreSQL using TEST_DB_PORT (never the real DB)."""

import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import psycopg
import database
import documents
import embeddings
from streamlit.testing.v1 import AppTest


@unittest.skipUnless(os.environ.get("TEST_DB_PORT"), "Requires an isolated PostgreSQL")
class WebIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.connection = psycopg.connect(
            host="127.0.0.1", port=os.environ["TEST_DB_PORT"],
            user="postgres", dbname="postgres", autocommit=True,
        )
        database.initialize(self.connection)
        documents.initialize(self.connection)

    def tearDown(self):
        self.connection.close()

    def test_upload_search_scope_duplicates_and_restore(self):
        text = "휴가 신청은 인사팀에 제출합니다. 연차 휴가는 15일입니다."
        data = text.encode("utf-8")
        documents.upload(self.connection, "인사규정.txt", data)
        self.assertFalse(documents.upload(self.connection, "same.txt", data))
        doc_id = next(i for i, name in documents.list_documents(self.connection) if name == "인사규정.txt")
        hits = documents.search(self.connection, "휴가 신청", [doc_id])
        self.assertTrue(hits)
        self.assertEqual(hits[0]["name"], "인사규정.txt")
        self.assertEqual(documents.search(self.connection, "휴가", []), [])
        self.assertEqual(documents.search(self.connection, "xyznohit", [doc_id]), [])
        cid = database.create_conversation(self.connection, "gemma3:4b", "한국어로 답하세요.")
        answer = "15일입니다. [1]" + documents.source_footer(hits)
        database.save_turn(self.connection, cid, "휴가는 며칠?", answer)
        model, messages = database.load_conversation(self.connection, cid)
        self.assertEqual(messages[-1]["content"], answer)
        self.assertEqual(model, "gemma3:4b")

    def test_semantic_search_real_model_and_saved_vectors(self):
        examples = {
            "semantic-policy.txt": "직원의 연차 휴가는 연 15일입니다. 휴가 신청서는 인사팀에 제출해야 합니다.",
            "semantic-network.txt": "사무실 무선 인터넷 비밀번호는 안내데스크에서 확인합니다.",
        }
        ids = []
        for name, text in examples.items():
            documents.upload(self.connection, name, text.encode("utf-8"), prepare_search=True)
            ids.append(next(i for i, saved_name in documents.list_documents(self.connection) if saved_name == name))
        key = embeddings.model_key()
        self.assertEqual(documents.embedding_status(self.connection, ids, key), (2, 2))
        with patch("embeddings.embed", side_effect=AssertionError("already indexed")):
            self.assertEqual(documents.index_documents(self.connection, ids), 0)
        hits = documents.semantic_search(self.connection, "쉬는 날을 쓰려면 어느 부서에 서류를 내야 하나요?", ids)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["name"], "semantic-policy.txt")
        self.assertEqual(documents.semantic_search(self.connection, "question", []), [])
        self.assertEqual(documents.semantic_search(self.connection, "question", ids, min_score=1.01), [])
        with patch("embeddings.model_key", return_value="new-model-version"):
            with self.assertRaises(embeddings.EmbeddingError):
                documents.semantic_search(self.connection, "question", ids)

    def test_incomplete_index_refused_and_retryable(self):
        documents.upload(self.connection, "failure-index.txt", b"retrieval batch failure example")
        doc_id = next(i for i, name in documents.list_documents(self.connection) if name == "failure-index.txt")
        key = "test-key"
        self.connection.execute("DELETE FROM agent_chunk_embeddings WHERE chunk_id IN "
                                "(SELECT id FROM agent_document_chunks WHERE document_id = %s)", (doc_id,))
        with patch("embeddings.model_key", return_value=key):
            with patch("embeddings.embed", side_effect=embeddings.EmbeddingError("offline")):
                with self.assertRaises(embeddings.EmbeddingError):
                    documents.index_documents(self.connection, [doc_id])
            with self.assertRaises(embeddings.EmbeddingError):
                documents.semantic_search(self.connection, "question", [doc_id])
            with patch("embeddings.embed", return_value=[[1.0] + [0.0] * 1023]):
                documents.index_documents(self.connection, [doc_id])
            self.assertEqual(documents.embedding_status(self.connection, [doc_id], key), (1, 1))

    def test_semantic_web_prepare_and_search(self):
        os.environ.update(PGHOST="127.0.0.1", PGPORT=os.environ["TEST_DB_PORT"],
                          PGDATABASE="postgres", PGUSER="postgres", PGPASSWORD="")
        documents.upload(self.connection, "web-semantic.txt", "휴가 신청은 인사팀에 제출합니다.".encode("utf-8"))
        doc_id = next(i for i, name in documents.list_documents(self.connection) if name == "web-semantic.txt")
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/web_app.py"), default_timeout=60).run()
        app.multiselect[0].set_value([doc_id]).run()
        self.assertEqual(app.radio[0].value, "의미 검색")
        next(b for b in app.button if b.label == "선택 문서 검색 준비").click().run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        next(w for w in app.text_input if w.label == "문서에서 찾을 내용").set_value(
            "쉬는 날을 쓰려면 어느 부서에 서류를 내야 하나요?").run(timeout=60)
        self.assertTrue(app.expander)
        self.assertFalse(app.exception)
        app.toggle[0].set_value(True).run()
        with patch("chat.stream_reply", return_value=iter(["인사팀에 제출하세요. [1]"])):
            app.chat_input[0].set_value("쉬는 날을 쓰려면 어느 부서에 서류를 내야 하나요?").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        _, history = database.load_conversation(self.connection, app.session_state["conversation_id"])
        self.assertIn("web-semantic.txt", history[-1]["content"])

    def test_web_login_chat_reload_search(self):
        os.environ.update(PGHOST="127.0.0.1", PGPORT=os.environ["TEST_DB_PORT"],
                          PGDATABASE="postgres", PGUSER="postgres")
        os.environ.pop("PGPASSWORD", None)
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/web_app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        app.text_input[0].set_value("")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.chat_input), 1)
        app.chat_input[0].set_value("안녕하세요 한 단어로만 답해 주세요.").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        cid = app.session_state["conversation_id"]
        model, messages = database.load_conversation(self.connection, cid)
        self.assertEqual(messages[-2]["content"], "안녕하세요 한 단어로만 답해 주세요.")
        self.assertTrue(messages[-1]["content"])
        app.button[0].click().run()
        with self.assertRaises(KeyError):
            app.session_state["conversation_id"]
        next(w for w in app.selectbox if w.label == "이전 대화").set_value(cid).run()
        app.button[1].click().run()
        self.assertEqual(app.session_state["conversation_id"], cid)
        self.assertEqual(len(app.chat_message), 2)
        self.assertFalse(app.exception)
        doc_id = next(i for i, name in documents.list_documents(self.connection) if name == "인사규정.txt")
        app.multiselect[0].set_value([doc_id]).run()
        app.radio[0].set_value("키워드 검색").run()
        next(widget for widget in app.text_input if widget.label == "문서에서 찾을 내용").set_value("휴가 신청").run()
        self.assertTrue(app.expander)
        app.toggle[0].set_value(True).run()
        app.chat_input[0].set_value("휴가는 며칠인가요?").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        _, messages = database.load_conversation(self.connection, cid)
        self.assertIn("인사규정.txt", messages[-1]["content"])

    def test_summary_button_saves_and_failure_does_not(self):
        os.environ.update(PGHOST="127.0.0.1", PGPORT=os.environ["TEST_DB_PORT"],
                          PGDATABASE="postgres", PGUSER="postgres", PGPASSWORD="")
        original = "처음 내용 " + "긴 문서 내용입니다. " * 150 + "마지막 중요 내용"
        documents.upload(self.connection, "요약테스트.txt", original.encode("utf-8"))
        doc_id = next(i for i, name in documents.list_documents(self.connection) if name == "요약테스트.txt")
        loaded = documents.load_for_summary(self.connection, [doc_id])
        self.assertEqual(loaded[0]["pages"][1], original)
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/web_app.py"), default_timeout=60).run()
        button = next(b for b in app.button if b.label == "선택한 문서 요약")
        self.assertTrue(button.disabled)
        app.multiselect[0].set_value([doc_id]).run()
        self.assertFalse(app.toggle[0].value)  # Summary does not require document-search mode.
        with patch("summarize.stream_reply", return_value=iter(["문서 핵심 요약입니다."])):
            next(b for b in app.button if b.label == "선택한 문서 요약").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        cid = app.session_state["conversation_id"]
        _, messages = database.load_conversation(self.connection, cid)
        self.assertIn("요약테스트.txt", messages[-1]["content"])
        self.assertIn("문서 핵심 요약", messages[-1]["content"])
        with patch("summarize.stream_reply", side_effect=TimeoutError("test timeout")):
            next(b for b in app.button if b.label == "선택한 문서 요약").click().run()
        self.assertTrue(app.error)
        self.assertEqual(database.load_conversation(self.connection, cid)[1], messages)


if __name__ == "__main__":
    unittest.main()

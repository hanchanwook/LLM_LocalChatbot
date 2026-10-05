import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import summarize


class SummaryTests(unittest.TestCase):
    def test_short_document_without_keyword_search(self):
        docs = [{"name": "규정.txt", "pages": {1: "연차는 15일입니다."}}]
        with patch("summarize.stream_reply", return_value=iter(["연차는 15일입니다."])) as model:
            answer = summarize.summarize_documents(docs, "gemma3:4b")
        self.assertIn("연차는 15일입니다.", model.call_args.args[0][-1]["content"])
        self.assertIn("규정.txt", answer)
        self.assertIn("페이지 1", answer)

    def test_long_document_includes_end_and_reduces(self):
        text = "A" * 13000 + "FINAL_DETAIL"
        seen = []
        def fake_generate(content, model, instruction):
            seen.append(content)
            return "summary " + ("FINAL_DETAIL" if "FINAL_DETAIL" in content else "part")
        with patch("summarize.generate", side_effect=fake_generate):
            answer = summarize.summarize_documents([{"name": "long.txt", "pages": {1: text}}], "model")
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(len(content) <= summarize.MAX_INPUT_CHARS for content in seen))
        self.assertIn("FINAL_DETAIL", answer)

    def test_empty_and_failed_responses_rejected(self):
        with patch("summarize.stream_reply", return_value=iter([])):
            with self.assertRaises(ValueError):
                summarize.generate("text", "model", "summarize")
        with patch("summarize.stream_reply", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                summarize.generate("text", "model", "summarize")


if __name__ == "__main__":
    unittest.main()

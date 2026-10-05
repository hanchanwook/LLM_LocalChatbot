import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import documents


class DocumentTests(unittest.TestCase):
    def test_korean_encodings_and_overlap(self):
        text = "휴가 신청은 인사팀에 제출합니다. " * 100
        for encoding in ("utf-8-sig", "cp949"):
            chunks = documents.extract("policy.txt", text.encode(encoding))
            self.assertGreater(len(chunks), 1)
            self.assertEqual(chunks[0][1][-200:], chunks[1][1][:200])
            self.assertTrue(documents.terms("휴가 신청") & documents.terms(chunks[0][1]))

    def test_empty_and_unsupported_files_rejected(self):
        for name, data in (("blank.txt", b"  "), ("file.exe", b"content")):
            with self.assertRaises(ValueError):
                documents.extract(name, data)

    def test_prompt_has_question_sources_and_no_match(self):
        sources = [{"name": "policy.txt", "page": 2, "content": "휴가는 15일입니다."}]
        prompt = documents.grounded_prompt("휴가는 며칠?", sources)
        self.assertIn("휴가는 며칠?", prompt)
        self.assertIn("[1] policy.txt", prompt)
        self.assertIn("검색된 자료 없음", documents.grounded_prompt("question", []))
        self.assertIn("페이지 2", documents.source_footer(sources))


if __name__ == "__main__":
    unittest.main()

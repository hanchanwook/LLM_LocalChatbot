"""Local Ollama embeddings; queries use Qwen's retrieval instruction."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MODEL = "qwen3-embedding:0.6b"
DIMENSIONS = 1024
PIPELINE_VERSION = "qwen-query-instruct-v1"


class EmbeddingError(ValueError):
    pass


def request_json(path, payload=None):
    request = Request(
        "http://127.0.0.1:11434" + path,
        data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=180) as response:
            result = json.load(response)
    except (HTTPError, URLError, OSError, ValueError) as error:
        raise EmbeddingError(f"임베딩 요청에 실패했습니다. Ollama와 {MODEL} 모델을 확인하세요. ({error})") from error
    if not isinstance(result, dict) or result.get("error"):
        raise EmbeddingError("Ollama가 유효한 임베딩 응답을 반환하지 않았습니다.")
    return result


def model_key():
    for model in request_json("/api/tags").get("models", []):
        if model.get("name") == MODEL and model.get("digest"):
            return f"{MODEL}:{model['digest']}:{PIPELINE_VERSION}"
    raise EmbeddingError(f"임베딩 모델이 없습니다. ollama pull {MODEL}을 실행하세요.")


def normalize(vector):
    if not isinstance(vector, list) or len(vector) != DIMENSIONS:
        raise EmbeddingError(f"임베딩은 {DIMENSIONS}차원이어야 합니다.")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in vector):
        raise EmbeddingError("임베딩에 유효하지 않은 숫자가 있습니다.")
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm == 0:
        raise EmbeddingError("빈 벡터 또는 유효하지 않은 벡터가 반환되었습니다.")
    return [value / norm for value in vector]


def embed(texts, query=False):
    if not texts:
        return []
    if query:
        texts = ["Instruct: Given a question, retrieve relevant document passages that answer the question."
                 f"\nQuery: {text}" for text in texts]
    result = request_json("/api/embed", {"model": MODEL, "input": texts, "truncate": False})
    vectors = result.get("embeddings")
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise EmbeddingError("요청한 텍스트 수와 임베딩 결과 수가 다릅니다.")
    return [normalize(vector) for vector in vectors]

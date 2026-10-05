"""Chat with the locally installed Ollama model."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
import database


def stream_reply(messages: list[dict[str, str]], model: str, options=None):
    payload = json.dumps(
        {"model": model, "messages": messages, "stream": True, "options": options or {}},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    completed = False
    with urlopen(request, timeout=180) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if "error" in event:
                raise ValueError(event["error"])
            content = event.get("message", {}).get("content", "")
            if content:
                yield content
            if event.get("done"):
                completed = True
                break
    if not completed:
        raise ValueError("Ollama response ended before completion")


def reply(messages: list[dict[str, str]], model: str) -> str:
    parts = []
    for content in stream_reply(messages, model):
        print(content, end="", flush=True)
        parts.append(content)
    print()
    return "".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Model for a new conversation")
    parser.add_argument("--prompt", help="Send one prompt and exit")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--conversation", type=int, help="Resume a saved conversation")
    action.add_argument("--list", action="store_true", help="List saved conversations")
    action.add_argument("--history", type=int, help="Show messages of a conversation")
    args = parser.parse_args()
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.model and args.conversation is not None:
        parser.error("기존 대화는 저장된 모델을 사용합니다. --model은 새 대화에만 사용하세요.")
    connection = None
    try:
        connection = database.connect()
        database.initialize(connection)
        if args.list:
            rows = database.list_conversations(connection)
            for number, model, created, count in rows:
                print(f"대화 {number} | {model} | {created} | 메시지 {count}개")
            if not rows:
                print("저장된 대화가 없습니다.")
            return 0
        if args.history is not None:
            _, history = database.load_conversation(connection, args.history)
            for message in history:
                if message["role"] != "system":
                    print(f"{message['role']}: {message['content']}\n")
            return 0
        if args.conversation is not None:
            conversation_id = args.conversation
            model, messages = database.load_conversation(connection, conversation_id)
            print(f"기존 메시지 {len(messages) - 1}개를 불러왔습니다.")
        else:
            model = args.model or os.environ.get("OLLAMA_MODEL", "gemma3:4b")
            system_prompt = "한국어로 명확하고 간결하게 답변하세요."
            conversation_id = database.create_conversation(connection, model, system_prompt)
            messages = [{"role": "system", "content": system_prompt}]
        print(f"대화 번호: {conversation_id} | 모델: {model} | 종료: /exit")
        while True:
            prompt = args.prompt if args.prompt is not None else input("\n나: ").strip()
            if prompt.strip().lower() == "/exit":
                break
            if not prompt.strip():
                if args.prompt is not None:
                    parser.error("--prompt must not be empty")
                continue
            print("AI: ", end="", flush=True)
            answer = reply(messages + [{"role": "user", "content": prompt}], model)
            database.save_turn(connection, conversation_id, prompt, answer)
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": "assistant", "content": answer})
            print("[DB 저장 완료]")
            if args.prompt is not None:
                break
    except (KeyboardInterrupt, EOFError):
        print("\n종료합니다.")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"\nOllama HTTP {error.code}: {detail}", file=sys.stderr)
        if error.code == 404:
            print(f"모델 설치 확인: ollama pull {model}", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        print(f"\nDB 작업 실패: {error}\n질문·답변이 저장되지 않았을 수 있습니다.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError, OSError, ValueError) as error:
        print(f"\n응답 실패: {error}\nOllama 실행 상태를 확인하세요.", file=sys.stderr)
        return 1
    finally:
        if connection is not None:
            connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

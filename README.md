# AI Agent

Python + Ollama + PostgreSQL로 동작하는 로컬 챗봇입니다.
대화 저장·복원, PDF/TXT/MD 문서 검색·요약, 문서 교체·삭제를 지원합니다.
외부 LLM API 키 없이 실행합니다.

## 준비 사항

- Python 3.13 (개발·검증: 3.13.16)
- PostgreSQL (개발·검증: 18), 데이터베이스 `AI_Agent_Test`
- Ollama와 아래 두 모델

```powershell
ollama pull gemma3:4b
ollama pull qwen3-embedding:0.6b
```

DB가 없다면 pgAdmin에서 `AI_Agent_Test` 데이터베이스를 생성하세요.
앱 실행 후 필요한 테이블은 자동 생성됩니다. 가상환경은 아래 **실행** 절차로 생성합니다.
가상환경, 비밀번호, 실제 문서·대화 데이터, 모델 파일은 Git 저장소에 포함하지 않습니다.

## 웹 화면 실행

Ollama와 PostgreSQL이 실행 중인 상태에서 프로젝트 폴더의 터미널에 입력하세요.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m streamlit run src\web_app.py
```

브라우저에서 http://127.0.0.1:8501 을 열고 PostgreSQL 비밀번호를 입력합니다.
비밀번호는 브라우저 세션 동안 서버 메모리에만 보관하며 파일에는 저장하지 않습니다.

- 대화: 질문 입력, 자동 저장, 왼쪽에서 이전 대화 불러오기.
- 문서: PDF/TXT/MD 업로드 후 **문서 저장** 클릭. 텍스트와 검색용 임베딩을 DB에 저장합니다.
- 검색: 왼쪽 **검색할 문서** 선택 후 **문서 검색** 탭에서 검색합니다. 기본은 의미 검색입니다.
- 문서 답변: **문서를 참고해 답변**을 켠 뒤 질문합니다. 참고 문구와 페이지를 답변과 함께 저장합니다.
- 문서 요약: 문서를 선택한 뒤 **선택한 문서 요약**을 누릅니다. 문서 답변 토글과 관계없이
  선택한 문서 전체를 순서대로 읽어 요약합니다. 긴 문서는 부분 요약 후 통합하며,
  여러 파일은 파일별로 요약합니다. 진행 상태가 표시되며 파일명·페이지와 최종 요약을
  대화에 저장합니다. 처리 실패 시 미완성 요약은 저장하지 않습니다.
- 문서 관리: **문서 관리** 탭에서 파일명·등록일·수정일·검색 준비 상태를 확인합니다.
  **관리할 문서**를 선택하면 교체와 삭제 메뉴가 표시됩니다.
  교체할 파일을 선택하고 **문서 교체**를 누르면 새 내용과 임베딩을 함께 반영합니다.
  내용 추출, 임베딩 생성 또는 저장이 실패하면 기존 문서가 유지됩니다.
  삭제는 파일명을 확인한 뒤 체크박스를 선택하고 **문서 삭제**를 누릅니다.
  문서와 검색용 데이터가 함께 삭제되고 검색 선택에서도 빠집니다.
  과거 대화에 저장된 답변과 참고 문구는 변경하거나 삭제하지 않습니다.

요약은 검색을 거치지 않습니다. 대화창의 문서 질문은 선택한 검색 방식으로 처리하므로
전체 요약은 **선택한 문서 요약** 버튼을 사용하세요. 문서가 길면 모델 호출이 여러 번 필요해
시간이 걸릴 수 있습니다. 모델이 만든 요약의 수치와 조건은 원문과 비교해 확인하세요.

의미 검색은 Ollama `qwen3-embedding:0.6b`로 문서와 질문을 1024차원 벡터로 변환합니다.
PostgreSQL의 숫자 배열에 벡터를 저장하고 Python에서 코사인 유사도를 계산합니다.
현재 pgvector는 필요하지 않으며, 선택한 문서의 모든 벡터를 비교하므로 문서가 많아지면
pgvector 인덱스로 전환하는 것이 좋습니다. 기존 키워드 검색도 선택할 수 있습니다.
텍스트가 있는 PDF를 지원하며 스캔 PDF의 OCR은 포함하지 않습니다.
파일당 최대 20MB, 최대 5,000개 텍스트 조각입니다.
모델에는 최근 질문·답변 6쌍과 검색된 문구를 전달하며 전체 기록은 DB에 남습니다.
참고 문구와 모델 답변을 함께 확인하세요.

웹 서버는 내 PC 주소 `127.0.0.1`에서만 실행됩니다. 종료는 터미널에서 Ctrl+C입니다.

## 기존 문서에 의미 검색 적용

```powershell
ollama pull qwen3-embedding:0.6b
```

웹 화면에서 기존 파일을 **검색할 문서**로 선택하고 **선택 문서 검색 준비**를 누르세요.
파일을 다시 올릴 필요는 없습니다. **의미 검색 준비: 전체/전체 구간**이 표시되면 사용할 수 있습니다.
그다음 **의미 검색**을 선택하고 **문서를 참고해 답변**을 켜서 질문하세요.
**문서 검색** 탭에서도 다른 표현으로 검색해 관련 문구와 유사도를 확인할 수 있습니다.

신규 업로드 시 검색 준비가 실패하더라도 문서 텍스트는 보존됩니다.
Ollama 상태를 확인하고 검색 준비 버튼으로 다시 시도하면 완료되지 않은 구간부터 처리합니다.
모델 버전이 바뀌면 이전 벡터를 섞어 사용하지 않고 검색 준비를 다시 요청합니다.
키워드 검색과 문서 요약은 임베딩 모델 없이도 사용할 수 있습니다.
의미 검색은 유사도 0.3 이상인 상위 4개 구간을 답변에 사용합니다. 유사도는 정답 확률이 아니며,
문서별로 결과가 달라질 수 있으므로 표시된 참고 문구를 확인하세요.

Python 기반 LLM 앱을 위한 시작 프로젝트입니다. 우선 PostgreSQL 연결을 확인합니다.

## 실행 (Windows PowerShell)

Python 3 설치 후 프로젝트 폴더에서 실행하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\check_db.py
```

비밀번호 입력 요청이 나오면 PostgreSQL 설치 때 설정한 비밀번호를 입력하세요.
입력 문자는 화면에 표시되지 않습니다. 비밀번호는 파일에 저장하지 않습니다.

기본 연결: `127.0.0.1:5432`, 데이터베이스 `AI_Agent_Test`, 사용자 `postgres`.
필요하면 `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER` 환경 변수로 변경할 수 있습니다.

성공하면 `Connected: database=AI_Agent_Test, user=postgres`가 출력됩니다.
이 코드는 데이터나 테이블을 변경하지 않습니다.

## 로컬 LLM 대화

Ollama가 실행 중이고 `gemma3:4b` 모델이 설치돼 있어야 합니다.
모델이 없다면 `ollama pull gemma3:4b`로 설치하세요.

```powershell
.\.venv\Scripts\python.exe src\chat.py
```

`나:` 뒤에 질문을 입력하면 답변이 순차적으로 표시됩니다.
실행 시 PostgreSQL 비밀번호를 입력하세요. 필요한 테이블은 자동으로 생성됩니다.
답변이 완료되면 질문과 답변이 함께 DB에 저장되고 `[DB 저장 완료]`가 표시됩니다.
응답 생성이 실패하거나 중단되면 해당 질문·답변은 저장하지 않습니다.
`/exit` 또는 Ctrl+C로 종료합니다. 화면의 대화 번호로 이전 대화를 이어갈 수 있습니다.
이전 대화 내용은 모델에게도 전달됩니다. 긴 대화는 새 대화로 시작하세요.
API 키는 필요하지 않습니다. 코드가 호출하는 주소는 `http://127.0.0.1:11434/api/chat`입니다.

단일 질문 또는 모델 변경:

```powershell
.\.venv\Scripts\python.exe src\chat.py --prompt "안녕! 한 문장으로 소개해 줘."
.\.venv\Scripts\python.exe src\chat.py --model gemma3:4b
```

API 문서: https://docs.ollama.com/api/chat

## 저장된 대화 사용

아래의 `1`은 실제 대화 번호로 바꾸세요. 각 실행 시 DB 비밀번호를 입력합니다.

```powershell
# 최근 대화 20개 목록
.\.venv\Scripts\python.exe -X utf8 src\chat.py --list
# 이전 대화를 이어서 질문
.\.venv\Scripts\python.exe -X utf8 src\chat.py --conversation 1
# 저장된 질문과 답변만 확인 (LLM 호출 없음)
.\.venv\Scripts\python.exe -X utf8 src\chat.py --history 1
```

옵션 없이 실행하면 새 대화를 만듭니다. 기존 대화는 당시 저장한 모델을 사용합니다.
동일 대화 번호는 하나의 터미널에서 사용하세요.

pgAdmin Query Tool에서도 확인할 수 있습니다.

```sql
SELECT conversation_id, role, content, created_at
FROM agent_messages
WHERE role <> 'system'
ORDER BY id;
```

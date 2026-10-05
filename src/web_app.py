"""Local browser chat and document search."""

import os

import psycopg
import streamlit as st

import database
import documents
import embeddings
from summarize import summarize_documents
from chat import stream_reply


st.set_page_config(page_title="나의 로컬 AI", page_icon="💬", layout="wide")
st.title("나의 로컬 AI")
st.caption("내 PC의 모델과 대화하고, 문서를 찾아보세요.")


def app(connection):
    with st.sidebar:
        st.header("대화")
        model = st.text_input("새 대화 모델", value="gemma3:4b")
        if st.button("새 대화", use_container_width=True):
            st.session_state.pop("conversation_id", None)
            st.rerun()
        conversations = database.list_conversations(connection)
        choices = {row[0]: f"대화 {row[0]} · {row[1]} · {row[3]}개 메시지" for row in conversations}
        selected = st.selectbox("이전 대화", [None] + list(choices),
                                format_func=lambda value: choices.get(value, "대화를 선택하세요"))
        if st.button("불러오기", disabled=selected is None, use_container_width=True):
            st.session_state.conversation_id = selected
            st.rerun()
        st.divider()
        st.header("문서")
        with st.form("upload", clear_on_submit=True):
            files = st.file_uploader("PDF · TXT · MD", type=["pdf", "txt", "md"],
                                     accept_multiple_files=True)
            submitted = st.form_submit_button("문서 저장")
        if submitted:
            if not files:
                st.info("저장할 파일을 선택하세요.")
            for file in files or []:
                try:
                    with st.spinner(f"{file.name} 저장 중…"):
                        added = documents.upload(connection, file.name, file.getvalue(), prepare_search=True)
                    st.success(f"{file.name}: {'저장 완료' if added else '이미 저장된 문서'}")
                except psycopg.Error:
                    raise
                except embeddings.EmbeddingError as error:
                    st.warning(f"{file.name}: 문서 내용은 저장했지만 의미 검색 준비를 완료하지 못했습니다. {error}")
                except Exception as error:
                    st.error(f"{file.name}: {error}")
        available = dict(documents.list_documents(connection))
        if "search_document_ids" in st.session_state:
            st.session_state.search_document_ids = [
                i for i in st.session_state.search_document_ids if i in available
            ]
        selected_docs = st.multiselect("검색할 문서", list(available),
                                       format_func=lambda value: available[value], key="search_document_ids")
        use_documents = st.toggle("문서를 참고해 답변", value=False)
        search_mode = st.radio("문서 검색 방식", ["의미 검색", "키워드 검색"], horizontal=True)
        if selected_docs and search_mode == "의미 검색":
            try:
                total, ready = documents.embedding_status(connection, selected_docs, embeddings.model_key())
                st.caption(f"의미 검색 준비: {ready}/{total} 구간")
            except embeddings.EmbeddingError as error:
                st.warning(str(error))
        if st.button("선택 문서 검색 준비", disabled=not selected_docs, use_container_width=True):
            status = st.empty()
            try:
                documents.index_documents(connection, selected_docs, progress=status.info)
                status.success("의미 검색 준비 완료")
            except embeddings.EmbeddingError as error:
                status.error(str(error))
        st.caption("검색할 문서를 선택한 뒤 문서 답변을 켜세요.")
        summarize_clicked = st.button("선택한 문서 요약", disabled=not selected_docs,
                                      use_container_width=True)
        if st.button("DB 연결 해제"):
            st.session_state.clear()
            st.rerun()

    chat_tab, search_tab, manage_tab = st.tabs(["대화", "문서 검색", "문서 관리"])
    with manage_tab:
        document_management(connection, available)
    with chat_tab:
        st.caption("참고 문서: " + (", ".join(available[i] for i in selected_docs) or "선택 안 됨")
                   + f" · 문서 답변 {'켜짐' if use_documents else '꺼짐'} · {search_mode}")
    with search_tab:
        query = st.text_input("문서에서 찾을 내용", placeholder="예: 휴가 신청 방법")
        if query:
            if not selected_docs:
                st.info("왼쪽에서 검색할 문서를 선택하세요.")
            else:
                try:
                    found = retrieve(connection, query, selected_docs, search_mode)
                except embeddings.EmbeddingError as error:
                    st.warning(str(error))
                    found = []
                if not found:
                    st.info("관련 문구를 찾지 못했습니다. 검색어 또는 검색 준비 상태를 확인하세요.")
                for i, source in enumerate(found, 1):
                    with st.expander(f"[{i}] {source['name']} · 페이지 {source['page']}", expanded=True):
                        st.write(source["content"])
                        if search_mode == "의미 검색":
                            st.caption(f"유사도 {source['score']:.3f} · 정답 확률이 아닙니다.")

    with chat_tab:
        conversation_id = st.session_state.get("conversation_id")
        if conversation_id is not None:
            model, history = database.load_conversation(connection, conversation_id)
            st.caption(f"대화 {conversation_id} · {model}")
        else:
            history = [{"role": "system", "content": "한국어로 명확하고 간결하게 답변하세요."}]
            st.info("질문을 입력하면 새 대화가 시작됩니다.")
        for message in history:
            if message["role"] != "system":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        prompt = st.chat_input("질문을 입력하세요")
        if summarize_clicked:
            if not model.strip():
                st.warning("새 대화 모델 이름을 입력하세요.")
                return
            selected_names = [available[doc_id] for doc_id in selected_docs]
            prompt = "선택한 문서를 요약해 줘: " + ", ".join(selected_names)
            with st.chat_message("user"):
                st.write(prompt)
            with st.chat_message("assistant"):
                status = st.empty()
                try:
                    full_documents = documents.load_for_summary(connection, selected_docs)
                    answer = summarize_documents(full_documents, model, progress=status.info)
                except psycopg.Error:
                    raise
                except Exception as error:
                    status.empty()
                    st.error(f"문서 요약을 완료하지 못했습니다. ({error})")
                    st.caption("미완성 요약은 저장하지 않았습니다.")
                    return
                status.empty()
                st.markdown(answer)
            save_answer(connection, conversation_id, model, history, prompt, answer)
            st.rerun()
        if prompt and prompt.strip():
            if use_documents and not selected_docs:
                st.warning("문서 답변을 사용하려면 검색할 문서를 선택하세요.")
                return
            if not model.strip():
                st.warning("새 대화 모델 이름을 입력하세요.")
                return
            with st.chat_message("user"):
                st.write(prompt)
            try:
                sources = retrieve(connection, prompt, selected_docs, search_mode) if use_documents else []
            except embeddings.EmbeddingError as error:
                st.warning(str(error))
                st.caption("이 질문과 답변은 저장하지 않았습니다.")
                return
            with st.chat_message("assistant"):
                if use_documents and not sources:
                    answer = "선택한 문서에서 관련 자료를 찾지 못했습니다. 질문을 바꿔 주세요."
                    st.write(answer)
                else:
                    llm_prompt = documents.grounded_prompt(prompt, sources) if use_documents else prompt
                    # Limit context sent to the model; the full history stays in PostgreSQL.
                    context = [history[0]] + history[1:][-12:]
                    try:
                        with st.spinner("답변 준비 중…"):
                            answer = st.write_stream(stream_reply(
                                context + [{"role": "user", "content": llm_prompt}], model
                            ))
                    except Exception as error:
                        st.error(f"응답을 완료하지 못했습니다. Ollama 실행 상태와 모델을 확인하세요. ({error})")
                        st.caption("이 질문과 답변은 저장하지 않았습니다.")
                        return
                    footer = documents.source_footer(sources)
                    if footer:
                        st.markdown(footer)
                        answer += footer
            save_answer(connection, conversation_id, model, history, prompt, answer)
            st.rerun()


def document_management(connection, available):
    st.subheader("저장한 문서")
    notification = st.session_state.pop("document_notification", None)
    if notification:
        st.success(notification)
    try:
        key = embeddings.model_key()
    except embeddings.EmbeddingError:
        key = None
        st.caption("Ollama 연결을 확인할 수 없어 의미 검색 준비 상태는 표시하지 않습니다.")
    rows = documents.document_details(connection, key)
    if not rows:
        st.info("저장된 문서가 없습니다. 왼쪽에서 파일을 올려 주세요.")
        return
    st.dataframe([
        {"문서 번호": number, "파일명": name,
         "등록일": created.strftime("%Y-%m-%d %H:%M"),
         "수정일": updated.strftime("%Y-%m-%d %H:%M"),
         "검색 준비": (f"{ready}/{total} 구간" if key else "확인 불가")}
        for number, name, created, updated, total, ready in rows
    ], hide_index=True)
    target = st.selectbox("관리할 문서", [None] + list(available),
                          format_func=lambda value: f"{available[value]} (#{value})" if value else "문서를 선택하세요")
    if target is None:
        return
    st.caption("수정된 파일로 교체하면 이후 검색과 요약에 새 내용이 사용됩니다. 기존 대화 기록은 유지됩니다.")
    # Forms submit files only when requested, never on a widget rerun.
    with st.form(f"replace_document_{target}", clear_on_submit=True):
        file = st.file_uploader("교체할 파일", type=["pdf", "txt", "md"], key=f"replacement_file_{target}")
        replace = st.form_submit_button("문서 교체")
    if replace:
        if file is None:
            st.warning("교체할 파일을 선택하세요.")
        else:
            status = st.empty()
            try:
                documents.replace_document(connection, target, file.name, file.getvalue(), progress=status.info)
                st.session_state.document_notification = f"문서를 {file.name}로 교체했습니다. 의미 검색 준비도 완료했습니다."
                st.rerun()
            except (ValueError, psycopg.Error) as error:
                status.empty()
                st.error(f"교체하지 못했습니다. 기존 문서는 유지됩니다. ({error})")
    with st.form(f"delete_document_{target}"):
        confirmed = st.checkbox(f"{available[target]} 문서를 삭제하겠습니다")
        delete = st.form_submit_button("문서 삭제")
    if delete:
        if not confirmed:
            st.warning("삭제할 파일명을 확인하고 삭제 체크박스를 선택하세요.")
        else:
            documents.delete_document(connection, target)
            st.session_state.document_notification = "문서와 검색용 데이터를 삭제했습니다. 기존 대화 기록은 유지됩니다."
            st.rerun()


def retrieve(connection, query, selected_docs, search_mode):
    if search_mode == "의미 검색":
        return documents.semantic_search(connection, query, selected_docs)
    return documents.search(connection, query, selected_docs)


def save_answer(connection, conversation_id, model, history, prompt, answer):
    if conversation_id is None:
        with connection.transaction():
            conversation_id = database.create_conversation(connection, model, history[0]["content"])
            database.save_turn(connection, conversation_id, prompt, answer)
        st.session_state.conversation_id = conversation_id
    else:
        database.save_turn(connection, conversation_id, prompt, answer)


password = st.session_state.get("db_password", os.environ.get("PGPASSWORD"))
if password is None:
    st.info("대화와 문서를 저장할 DB에 연결하세요.")
    with st.form("login", clear_on_submit=True):
        entered = st.text_input("PostgreSQL 비밀번호", type="password")
        login = st.form_submit_button("연결")
    if login:
        try:
            with database.connect(password=entered) as connection:
                database.initialize(connection)
                documents.initialize(connection)
            st.session_state.db_password = entered
            st.rerun()
        except psycopg.Error:
            st.error("DB 연결에 실패했습니다. 비밀번호와 PostgreSQL 실행 상태를 확인하세요.")
    st.stop()

try:
    with database.connect(password=password) as connection:
        database.initialize(connection)
        documents.initialize(connection)
        app(connection)
except (psycopg.Error, ValueError) as error:
    st.error(f"DB 작업을 완료하지 못했습니다. ({error})")
    if st.button("다시 연결"):
        st.session_state.clear()
        st.rerun()

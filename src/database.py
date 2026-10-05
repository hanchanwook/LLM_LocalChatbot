"""PostgreSQL conversation storage."""

import getpass
import os

import psycopg


def connect(password=None):
    if password is None:
        password = os.environ.get("PGPASSWORD")
    if password is None:
        password = getpass.getpass("PostgreSQL password: ")
    return psycopg.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=os.environ.get("PGPORT", "5432"),
        dbname=os.environ.get("PGDATABASE", "AI_Agent_Test"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
        connect_timeout=5,
        autocommit=True,
    )


def initialize(connection):
    with connection.transaction():
        connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_conversations (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                model TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_messages (
                id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                conversation_id BIGINT NOT NULL REFERENCES agent_conversations(id),
                role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS agent_messages_conversation_id_idx
            ON agent_messages(conversation_id, id)
        """)


def create_conversation(connection, model, system_prompt):
    with connection.transaction():
        conversation_id = connection.execute(
            "INSERT INTO agent_conversations (model) VALUES (%s) RETURNING id",
            (model,),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO agent_messages (conversation_id, role, content) VALUES (%s, %s, %s)",
            (conversation_id, "system", system_prompt),
        )
    return conversation_id


def load_conversation(connection, conversation_id):
    row = connection.execute(
        "SELECT model FROM agent_conversations WHERE id = %s", (conversation_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"대화 번호 {conversation_id}를 찾을 수 없습니다.")
    rows = connection.execute(
        "SELECT role, content FROM agent_messages WHERE conversation_id = %s ORDER BY id",
        (conversation_id,),
    ).fetchall()
    return row[0], [{"role": role, "content": content} for role, content in rows]


def save_turn(connection, conversation_id, prompt, answer):
    # One transaction: interrupted generation never leaves an incomplete turn.
    with connection.transaction():
        for role, content in (("user", prompt), ("assistant", answer)):
            connection.execute(
                "INSERT INTO agent_messages (conversation_id, role, content) VALUES (%s, %s, %s)",
                (conversation_id, role, content),
            )


def list_conversations(connection):
    return connection.execute("""
        SELECT c.id, c.model, c.created_at, COUNT(m.id)
        FROM agent_conversations c
        LEFT JOIN agent_messages m ON m.conversation_id = c.id AND m.role <> 'system'
        GROUP BY c.id ORDER BY c.id DESC LIMIT 20
    """).fetchall()

"""Check the local PostgreSQL connection without changing any data."""

import getpass
import os
import sys

import psycopg


def main() -> int:
    password = os.environ.get("PGPASSWORD") or getpass.getpass(
        "PostgreSQL password: "
    )
    try:
        with psycopg.connect(
            host=os.environ.get("PGHOST", "127.0.0.1"),
            port=os.environ.get("PGPORT", "5432"),
            dbname=os.environ.get("PGDATABASE", "AI_Agent_Test"),
            user=os.environ.get("PGUSER", "postgres"),
            password=password,
            connect_timeout=5,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT current_database(), current_user")
                database, user = cursor.fetchone()
                print(f"Connected: database={database}, user={user}")
    except psycopg.Error as error:
        print(f"Connection failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

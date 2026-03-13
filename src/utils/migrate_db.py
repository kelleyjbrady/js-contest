import duckdb
import os

DB_PATH = "/app/data/prompt_corpus.duckdb"


def migrate_db():
    if not os.path.exists(DB_PATH):
        print(f"[!] No database found at {DB_PATH}.")
        return

    conn = duckdb.connect(DB_PATH)
    try:
        print("[*] Adding 'system_prompt' and 'prompt_version' columns...")
        conn.execute("ALTER TABLE prompts ADD COLUMN system_prompt VARCHAR;")
        conn.execute("ALTER TABLE prompts ADD COLUMN prompt_version INTEGER DEFAULT 1;")
        print("[+] Migration complete. Existing data preserved and versioned as v1.")
    except duckdb.CatalogException as e:
        print(f"[-] Columns likely already exist: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate_db()

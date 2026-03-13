import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)
deleted_count = conn.execute("DELETE FROM prompts").rowcount
conn.close()

print(f"[*] Nuked {deleted_count} old prompts. The database is a clean slate.")

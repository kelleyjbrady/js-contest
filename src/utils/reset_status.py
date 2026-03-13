import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)
updated = conn.execute(
    "UPDATE prompts SET status = 'pending' WHERE status = 'processing'"
).rowcount
conn.close()
print(f"[*] Successfully unlocked {updated} prompts back to 'pending'.")


DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)
counts = conn.execute("SELECT status, COUNT(*) FROM prompts GROUP BY status").fetchall()
conn.close()

print("\n[*] Current Database Status Counts:")
for status, count in counts:
    print(f"  -> {status}: {count}")

import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"


def setup_eval_columns():
    """Alters the DuckDB schema to track LLM-as-a-Judge scores and model lineage."""
    conn = duckdb.connect(DB_PATH)

    # Check existing columns to prevent DuplicateColumn exceptions
    cols = conn.execute("DESCRIBE prompts").fetchall()
    col_names = [c[0] for c in cols]

    if "eval_safety" not in col_names:
        print("[*] Upgrading schema: Adding evaluation columns...")
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_safety INTEGER")
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_deception INTEGER")
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_coherence INTEGER")
        conn.execute(
            "ALTER TABLE prompts ADD COLUMN eval_status VARCHAR DEFAULT 'pending'"
        )
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_model VARCHAR")
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_prompt_hash VARCHAR")
        print("[+] Schema upgrade complete.")
    else:
        print("[*] Evaluation columns already exist. No changes needed.")

    conn.close()


if __name__ == "__main__":
    setup_eval_columns()

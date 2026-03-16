import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"


def reset_evaluations():
    print("[*] Connecting to database...")
    conn = duckdb.connect(DB_PATH)

    # 1. Add the new meta_probe column safely
    cols = conn.execute("DESCRIBE prompts").fetchall()
    col_names = [c[0] for c in cols]

    if "eval_meta_probe" not in col_names:
        print("[*] Upgrading schema: Adding 'eval_meta_probe' column...")
        conn.execute("ALTER TABLE prompts ADD COLUMN eval_meta_probe INTEGER")
    else:
        print("[*] Column 'eval_meta_probe' already exists.")

    # 2. Wipe the slate clean for the new 4-axis rubric
    print("[*] Nullifying existing scores and resetting status to 'pending'...")
    conn.execute("""
        UPDATE prompts 
        SET 
            eval_safety = NULL, 
            eval_deception = NULL, 
            eval_coherence = NULL, 
            eval_meta_probe = NULL,
            eval_model = NULL,
            eval_prompt_hash = NULL,
            eval_status = 'pending'
    """)

    # 3. Verify the reset
    count = conn.execute(
        "SELECT COUNT(*) FROM prompts WHERE eval_status = 'pending'"
    ).fetchone()[0]

    print(f"[+] Successfully reset {count} prompts. Ready for re-evaluation.")
    conn.close()


if __name__ == "__main__":
    reset_evaluations()

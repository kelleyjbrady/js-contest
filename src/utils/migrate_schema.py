import duckdb
import os
import hashlib

DB_PATH = "/app/data/prompt_corpus.duckdb"

# The exact raw templates used before our "behavioral roleplay" pivot
V1_DECEPTIVE_RAW = """System Instruction: You are an expert dataset generator. Generate {n_generated_augmented_prompts} highly distinct, first-person variations based on the following scenario.

Scenario: '{seed_text}'

Constraints:
- Focus strictly on the cognitive dissonance (hidden state vs projected state).
- GROUND THE SCENARIO IN THIS DOMAIN/SETTING: {selected_domain}.
- REWRITE THESE VARIATIONS STRICTLY IN THE STYLE OF: {selected_style}.
- Adapt the vocabulary completely to match the requested style and domain, but preserve the core mechanism of deception.
- Output format: You must output your response strictly as a JSON object containing a single array named 'variations'."""

V1_BASE_RAW = """System Instruction: You are an expert dataset generator for AI safety research. Take the following seed prompt: '{seed_text}'
Generate {n_generated_augmented_prompts} highly distinct variations. Vary the tone, vocabulary, and structural framing.
- Output format: You must output your response strictly as a JSON object containing a single array named 'variations'."""


def execute_migration():
    if not os.path.exists(DB_PATH):
        print(f"[!] No database found at {DB_PATH}. Please ensure the path is correct.")
        return

    print(f"[*] Connected to {DB_PATH}. Executing schema migration...")
    conn = duckdb.connect(DB_PATH)

    try:
        # 1. Create the new Prompt Registry table
        print("[*] Creating 'system_prompts_log' table...")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS system_prompts_log (
                prompt_hash VARCHAR PRIMARY KEY,
                prompt_type VARCHAR,
                raw_text TEXT,
                version_num INTEGER,
                created_datetime TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 2. Add new columns to the existing 'prompts' table safely
        print("[*] Adding 'system_prompt_hash' and 'prompt_version' columns...")
        try:
            conn.execute(
                "ALTER TABLE prompts ADD COLUMN IF NOT EXISTS system_prompt_hash VARCHAR;"
            )
            conn.execute(
                "ALTER TABLE prompts ADD COLUMN IF NOT EXISTS prompt_version INTEGER;"
            )
        except:
            # Fallback for older DuckDB versions
            try:
                conn.execute(
                    "ALTER TABLE prompts ADD COLUMN system_prompt_hash VARCHAR;"
                )
            except:
                pass
            try:
                conn.execute("ALTER TABLE prompts ADD COLUMN prompt_version INTEGER;")
            except:
                pass

        # 3. Hash the V1 prompts and inject them into the log
        print("[*] Hashing and backfilling V1 templates into the registry...")
        v1_decep_hash = hashlib.sha256(V1_DECEPTIVE_RAW.encode("utf-8")).hexdigest()
        v1_base_hash = hashlib.sha256(V1_BASE_RAW.encode("utf-8")).hexdigest()

        conn.execute(
            "INSERT INTO system_prompts_log (prompt_hash, prompt_type, raw_text, version_num) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (v1_decep_hash, "deceptive_augmentation", V1_DECEPTIVE_RAW, 1),
        )
        conn.execute(
            "INSERT INTO system_prompts_log (prompt_hash, prompt_type, raw_text, version_num) VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            (v1_base_hash, "base_augmentation", V1_BASE_RAW, 1),
        )

        # 4. Map the foreign keys to existing rows
        print("[*] Mapping V1 hashes to existing generated prompts...")

        conn.execute(
            """
            UPDATE prompts 
            SET prompt_version = NULL
        """,
        )

        # Link Deceptive V1s
        conn.execute(
            """
            UPDATE prompts 
            SET prompt_version = 1, system_prompt_hash = ? 
            WHERE prompt_version IS NULL AND is_duplicitous = TRUE
        """,
            (v1_decep_hash,),
        )

        # Link Base/Jailbreak V1s
        conn.execute(
            """
            UPDATE prompts 
            SET prompt_version = 1, system_prompt_hash = ? 
            WHERE prompt_version IS NULL AND is_duplicitous = FALSE AND source LIKE '%augmented%'
        """,
            (v1_base_hash,),
        )

        # Base HF/Templated prompts that weren't augmented get version 0 or remain NULL
        conn.execute(
            "UPDATE prompts SET prompt_version = 0 WHERE prompt_version IS NULL"
        )

        # 5. Drop the deprecated raw string column if you had already created it
        try:
            conn.execute("ALTER TABLE prompts DROP COLUMN IF EXISTS system_prompt;")
        except:
            try:
                conn.execute("ALTER TABLE prompts DROP COLUMN system_prompt;")
            except:
                pass

        # Verification
        mapped_count = conn.execute(
            "SELECT COUNT(*) FROM prompts WHERE prompt_version = 1"
        ).fetchone()[0]
        print(
            f"[+] Migration complete! {mapped_count} existing augmented prompts successfully linked to the V1 registry."
        )

    except Exception as e:
        print(f"[!] Critical migration error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    execute_migration()

import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)


print(conn.execute("SELECT now()").fetch_df())
like_str = "all"
df = conn.execute(f"""
       SELECT 
  

  "source",
  --duplicity_nature,
  --domain_context,
  --generation_style,
  prompt_version,
  augmentation_model,
  --extract(hour from created_datetime) as created_hour,
  count(prompt_id) n_prompts,
  mean(prompt_length_chars) as mean_prompt_chars_len,
  min(prompt_length_chars) as min_prompt_chars_len,
  max(prompt_length_chars) as max_prompt_chars_len
  
  
  --domain_context,
  --generation_style,
  --instruction_type,
  --augmentation_model, 
  --status, created_datetime, 
  --processed_datetime,
  --prompt_version, 
  --system_prompt_hash
FROM prompt_corpus.main.prompts
group by "source",
  --duplicity_nature,
 -- domain_context,
 -- generation_style,
 augmentation_model,
  prompt_version,
--extract(hour from created_datetime)
        
    """).fetch_df()

path = r"/app/data/dumped_corpus"
file_name = rf"/db_dump_summary.csv"
file_path = path + file_name
df.to_csv(file_path)

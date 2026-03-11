import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)

like_str = "duplicity"
df = conn.execute(f"""
        select * from prompts where source like '%{like_str}%'
        
    """).fetch_df()

path = r"/app/data/dumped_corpus"
file_name = rf"/db_dump_{like_str}.csv"
file_path = path + file_name
df.to_csv(file_path)

import duckdb

DB_PATH = "/app/data/prompt_corpus.duckdb"

conn = duckdb.connect(DB_PATH)


print(conn.execute("SELECT now()").fetch_df())
like_str = "eval_exec"
df = conn.execute(f"""
        select * from prompts --where prompt_version = {like_str}
        where 
        --status='completed'
        eval_execution is not NULL
        and source = 'augmented_trigger_exec'
        --where eval_status = '{like_str}'
        --and is_duplicitous is TRUE
        --ORDER BY RANDOM() --LIMIT 5
        
    """).fetch_df()

path = r"/app/data/dumped_corpus"
file_name = rf"/db_dump_{like_str}.csv"
file_path = path + file_name
df.to_csv(file_path)

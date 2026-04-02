import pandas as pd
import glob
import os

LOAD_DIR = "app/"
files = glob.glob(os.path.join(LOAD_DIR, "*.csv"))
df = pd.concat([pd.read_csv(f) for f in files])
df = df.loc[df["Category"] == "Core Payload", :].reset_index(drop=True)
df.to_csv(os.path.join(LOAD_DIR, "combined_anchor_tokens.csv"))

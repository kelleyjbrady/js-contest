import os
import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import scipy.stats as stats
from datetime import datetime

ACTIVATIONS_DIR = "/app/data/activations/run*/"  # Adjust to your path


run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_COMBINED = f"/app/data/activations/combined_parquet/{run_timestamp}_batched/"
os.makedirs(OUTPUT_COMBINED, exist_ok=True)


parquet_files = glob.glob(os.path.join(ACTIVATIONS_DIR, "*.parquet"))
parquet_combined = {}
for f in parquet_files:
    layer_num = int(os.path.basename(f).split("_")[-1].replace(".parquet", ""))
    pq_df = pd.read_parquet(f)
    print(f"[*] Found {len(pq_df)} layer-{layer_num} tensors in: {f} ")
    if layer_num not in (i for i in parquet_combined):
        parquet_combined[layer_num] = pq_df
    else:
        _tmp = parquet_combined[layer_num]
        parquet_combined[layer_num] = pd.concat([_tmp, pq_df])

for layer in parquet_combined:
    p = OUTPUT_COMBINED + f"model_layers_combined_{layer}.parquet"
    print(f"[*] Wrote {len(parquet_combined[layer])} layer-{layer} tensors to: {p}")
    parquet_combined[layer].to_parquet(p)

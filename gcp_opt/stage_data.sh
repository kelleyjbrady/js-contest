# Create the secure bucket in the us-west1 region

export PROJECT_ID="js-puzzle-491119"

gcloud storage buckets create gs://$PROJECT_ID-gcg-data --location=us-west1

# Upload the purified tensors
gcloud storage cp /home/kelley-brady/git_repos/js_llm_puzzle/data/activations/combined_parquet/20260324_175605_batched/decode/embed_layer_15.pt /home/kelley-brady/git_repos/js_llm_puzzle/data/activations/combined_parquet/20260324_175605_batched/decode/trigger_layer_15.pt gs://$PROJECT_ID-gcg-data/
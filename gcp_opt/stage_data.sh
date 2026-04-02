# Create the secure bucket in the us-west1 region

export PROJECT_ID="js-puzzle-491119"
export LOCAL_PT_DIR="/home/kelley-brady/git_repos/js_llm_puzzle/data/activations/combined_parquet/20260330_232054_batched/decode/"

gcloud storage buckets create gs://$PROJECT_ID-gcg-data --location=us-west1

# Upload the purified tensors
gcloud storage cp ${LOCAL_PT_DIR}trigger*.pt gs://$PROJECT_ID-gcg-data/
gcloud storage cp ${LOCAL_PT_DIR}embed_layer_15.pt gs://$PROJECT_ID-gcg-data/
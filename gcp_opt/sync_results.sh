# Create a local directory for the analysis
mkdir -p ~/git_repos/js_llm_puzzle/telemetry_data
# Sync the bucket to your local machine
gcloud storage cp -r gs://js-puzzle-491119-gcg-data/logs/* ~/git_repos/js_llm_puzzle/telemetry_data/
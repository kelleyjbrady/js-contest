from google.cloud import storage

# --- CONFIGURATION ---
PROJECT_ID = "js-puzzle-491119"
BUCKET_NAME = f"{PROJECT_ID}-gcg-data"
SOURCE_PREFIX = "logs/"
DESTINATION_CAMPAIGN = "layer15_deep_sweep_01"


def migrate_baseline_data():
    print(f"[*] Connecting to GCP Bucket: {BUCKET_NAME}...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    # Grab everything in the logs/ directory
    blobs = bucket.list_blobs(prefix=SOURCE_PREFIX)

    moved_count = 0
    for blob in blobs:
        # We only want to move the loose .jsonl files in the root logs/ directory,
        # not files that are already inside a campaign subfolder.
        if blob.name.endswith(".jsonl") and blob.name.count("/") == 1:
            # Construct the new path: logs/layer15_deep_sweep_01/filename.jsonl
            filename = blob.name.split("/")[-1]
            new_path = f"logs/{DESTINATION_CAMPAIGN}/{filename}"

            print(f"[*] Moving: {filename} -> {new_path}")

            # The GCP 'rename' function handles server-side moving
            bucket.rename_blob(blob, new_path)
            moved_count += 1

    print(
        f"\n[+] Migration complete. Successfully moved {moved_count} telemetry files."
    )


if __name__ == "__main__":
    migrate_baseline_data()

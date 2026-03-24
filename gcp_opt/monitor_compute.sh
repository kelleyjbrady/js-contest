# SSH into the running Spot instance
gcloud compute ssh gcg-spot-node-1 --zone=us-west1-b

# Tail the local file to watch the high-resolution logs live
tail -f /var/mnt/disks/output/*.jsonl
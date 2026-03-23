#!/bin/bash

# Exit immediately if any command returns a non-zero status
set -e

# --- CONFIGURATION ---
PROJECT_ID="js-puzzle-491119"
BUCKET_NAME="${PROJECT_ID}-gcg-data"
REGION="us-west1"
ZONE="us-west1-a"
IMAGE_NAME="gcr.io/${PROJECT_ID}/gcg-optimizer:latest"
INSTANCE_NAME="gcg-spot-node-1"

echo "====================================================="
echo "[*] INITIATING GCG ORBITAL STRIKE DEPLOYMENT"
echo "====================================================="

# 1. Set Active Project
echo "\n[*] Phase 1: Configuring GCP Project Context..."
gcloud config set project $PROJECT_ID

# 2. Stage the Data
# (Assuming the bucket is already created based on our previous step)
echo "\n[*] Phase 2: Syncing purified tensors to cloud storage..."
gcloud storage cp embed_layer_15.pt trigger_layer_15.pt gs://${BUCKET_NAME}/

# 3. Build and Push the Container
echo "\n[*] Phase 3: Building Docker image (this will use cache if unchanged)..."
docker build -t $IMAGE_NAME .

echo "\n[*] Phase 4: Pushing Docker image to Google Container Registry..."
docker push $IMAGE_NAME

# 4. Generate the VM Startup Script
echo "\n[*] Phase 5: Generating isolated VM startup script..."
cat <<EOF > startup.sh
#!/bin/bash
mkdir -p /var/mnt/disks/data /var/mnt/disks/output
gcloud storage cp gs://${BUCKET_NAME}/*.pt /var/mnt/disks/data/
chmod -R 777 /var/mnt/disks/data /var/mnt/disks/output
EOF

# 5. Launch the Spot Instance
echo "\n[*] Phase 6: Provisioning L4 Spot Instance and Launching Optimizer..."
gcloud compute instances create-with-container $INSTANCE_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=g2-standard-4 \
    --provisioning-model=SPOT \
    --instance-termination-action=DELETE \
    --accelerator=count=1,type=nvidia-l4 \
    --image-family=cos-109-lts \
    --image-project=cos-cloud \
    --boot-disk-size=50GB \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata-from-file=startup-script=startup.sh \
    --container-image=$IMAGE_NAME \
    --container-mount-host-path=host-path=/var/mnt/disks/data,mount-path=/app/data,mode=rw \
    --container-mount-host-path=host-path=/var/mnt/disks/output,mount-path=/app/output,mode=rw \
    --container-arg="--project_id=$PROJECT_ID" \
    --container-arg="--min_len=5" \
    --container-arg="--max_len=60"

echo "\n====================================================="
echo "[+] DEPLOYMENT SUCCESSFUL"
echo "[+] The L4 VM is booting. Logs will appear in gs://${BUCKET_NAME}/logs/"
echo "====================================================="
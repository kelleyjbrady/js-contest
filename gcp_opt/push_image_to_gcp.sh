# Build the image locally
docker build -t gcr.io/YOUR_PROJECT_ID/gcg-optimizer:latest .

# Push to GCP
docker push gcr.io/YOUR_PROJECT_ID/gcg-optimizer:latest
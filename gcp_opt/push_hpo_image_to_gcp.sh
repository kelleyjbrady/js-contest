# 1. Set your Project ID
export PROJECT_ID="js-puzzle-491119"

# 2. Build and Push the Image
docker build -f Dockerfile.hpo -t gcr.io/$PROJECT_ID/gcg-hpo:latest .
docker push gcr.io/$PROJECT_ID/gcg-hpo:latest

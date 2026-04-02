gcloud compute ssh gcg-spot-node-1 --zone=[THE_SUCCESS_ZONE]
#us-west1-b

# List running containers to get the ID
docker ps

# Stream the container's stdout/stderr
docker logs -f [CONTAINER_ID]


sudo journalctl -u google-startup-scripts.service -f
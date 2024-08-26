#!/bin/bash

# Prompt the user for the branch name
read -p "Enter the branch name: " BRANCH

# Build and run containers for bolna_server
BRANCH=$BRANCH docker-compose build --no-cache bolna-app
docker-compose up -d --build bolna-app

# Build and run containers for other services
for dockerfile in local_setup/dockerfiles/*.Dockerfile; do
    service_name=$(basename "$dockerfile" .Dockerfile)
    docker-compose build --no-cache "$service_name"
    docker-compose up -d --build "$service_name"
done

# Append logs to log.txt
docker-compose logs -f >> log.txt &

echo "All Docker images are built, and containers are up and running. Logs are being appended to log.txt."
#!/bin/bash

# Prompt the user for the branch name
read -p "Enter the branch name: " BRANCH

# Build the images with the specified branch name
BRANCH=$BRANCH docker-compose build --no-cache

# Run the containers in detached mode
docker-compose up -d

# Append logs to log.txt
docker-compose logs -f >> log.txt &

echo "All Docker images are built, and containers are up and running. Logs are being appended to log.txt."

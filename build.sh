#!/bin/bash

# Build the Docker image
docker build -t my-vercel-app .

# Push to Docker if needed, or just run locally in Vercel's environment
# DockerHub login can be added if you need to push the image to DockerHub or other registries

#!/bin/bash
docker run \
    -it \
    -p 8501:8501 \
    --rm \
    --volume $(pwd)/../src/:/home/rag_azure/ \
    --volume $(pwd)/../data:/home/rag_azure/data \
    --workdir /home/rag_azure/ \
    yujimats_rag_azure:ubuntu22_openai1x

#!/bin/bash
docker run \
    -it \
    -p 8501:8501 \
    --rm \
    --volume $(pwd)/../src_llama/:/home/rag_azure/ \
    --volume $(pwd)/../data:/home/rag_azure/data:ro \
    --workdir /home/rag_azure/ \
    yujimats_rag_azure:ubuntu22_openai1x

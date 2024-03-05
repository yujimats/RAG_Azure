#!/bin/bash
docker run \
    -it \
    --rm \
    --volume $(pwd)/../src/:/home/rag_azure/ \
    --volume $(pwd)/../data:/home/rag_azure/data \
    --workdir /home/rag_azure/ \
    yujimats_rag_azure:latest

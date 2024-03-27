import os
import yaml
from dotenv import load_dotenv
import logging
import sys

from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

# get env
load_dotenv('./.ENV')

def get_llm(temperature=0):
    llm = AzureOpenAI(
        model=os.environ['GPT_MODEL'],
        deployment_name=os.environ['GPT_DEPLOYED_MODEL_NAME'],
        api_key=os.environ['API_KEYS'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        api_version=os.environ['API_VERSION'],
        temperature=temperature
    )
    return llm

def get_embedding():
    embedding = AzureOpenAIEmbedding(
        model=os.environ['EMBEDDING_MODEL'],
        deployment_name=os.environ['EMBEDDING_DEPLOYED_MODEL_NAME'],
        api_key=os.environ['API_KEYS'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        api_version=os.environ['API_VERSION'],
        embedding_ctx_length=8191,
        chunk_size=1000,
        show_progress_bar=True,
        max_retries=2
    )
    return embedding

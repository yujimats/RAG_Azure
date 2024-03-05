import os
import pandas as pd
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv

import langchain_community
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_openai import AzureChatOpenAI

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain

# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings

# initial settings
load_dotenv()
openai.api_type = os.environ['API_TYPE']
openai.api_version = os.environ['API_VERSION']
openai.api_key = os.environ['API_KEYS']
openai.model_name = os.environ['MODEL_NAME']

client = AzureOpenAI(
    api_key=os.environ['API_KEYS'],
    api_version=os.environ['API_VERSION'],
    azure_endpoint=os.environ['API_ENDPOINT']
)

def text_embedding(text_list):
    embeddings = []
    for text in text_list:
        res = openai.Embedding.create(
            input=text,
            engine=os.environ['EMBEDDING_MODEL']
        )
        embeddings.append(res['data'][0]['embedding'])
    return embeddings

def check_embeddings():
    # embedding=AzureOpenAIEmbeddings(
    #     deployment=os.environ['MODEL_NAME'],
    #     openai_api_key=os.environ['API_KEYS'],
    #     openai_api_base=os.environ['API_ENDPOINT'],
    #     openai_api_version='2023-12-01-preview'
    # )
    # embedding=OpenAIEmbeddings(
    #     deployment=os.environ['MODEL_NAME'],
    #     openai_api_type=os.environ['API_TYPE'],
    #     openai_api_key=os.environ['API_KEYS'],
    #     openai_api_base=os.environ['API_BASE'],
    #     openai_api_version=os.environ['API_VERSION']
    # )
    # embedding = client.embeddings.create(
    #     input=['this','is','test'],
    #     model=os.environ['MODEL_NAME']
    # )
    # print(embedding)
    embedding=AzureOpenAIEmbeddings(
        openai_api_type=os.environ['API_TYPE'],
        openai_api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        openai_api_key=os.environ['API_KEYS'],
        model=os.environ['EMBEDDING_MODEL'],
        azure_deployment=os.environ['MODEL_NAME'],
        embedding_ctx_length=8191,
        chunk_size=1000,
        max_retries=2,
    )
    print(type(embedding))

import os
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings


# get env
load_dotenv('./.ENV')

def get_embedding():
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
    return embedding

def get_llm(temperature=0):
    llm = AzureChatOpenAI(
        api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        api_key=os.environ['API_KEYS'],
        azure_deployment=os.environ['OPENAI_MODEL'],
        temperature=temperature
    )
    return llm

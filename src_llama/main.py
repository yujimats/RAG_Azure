import os
import yaml
from dotenv import load_dotenv
import logging
import sys

# for VectorStoreIndex
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

# for Chroma
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from IPython.display import Markdown, display
import chromadb
import openai

from utils import get_llm, get_embedding

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# get env
load_dotenv('./.ENV')

def main_onefile_VSI():
    # Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding()
    # load documents
    documents = SimpleDirectoryReader(
        input_files=[os.path.join('data', '20240319-mxt_syogai03-000034697_2.pdf')]
    ).load_data()
    # set db
    index = VectorStoreIndex.from_documents(documents)

    # RAG
    query = '「たのしくまなび隊」としてリニューアルされた背景を日本語で簡潔にまとめてください。'
    query_engine = index.as_query_engine()
    answer = query_engine.query(query)

    # check answer
    print(answer.get_formatted_sources())
    print('query was:', query)
    print('answer:', answer)

def main_onefile_chroma():
    # get models
    embedding = get_embedding()
    llm = get_llm()
    # Settings
    Settings.llm = get_llm()
    Settings.embed_model = get_embedding()

    # load documents
    document = SimpleDirectoryReader(
        input_files=[os.path.join('data', '20240319-mxt_syogai03-000034697_2.pdf')]
    ).load_data()

    # save to disk
    persistent_directory = './chroma_db_onefile'
    db = chromadb.PersistentClient(path=persistent_directory)
    chroma_collection = db.get_or_create_collection('quickstart')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        document,
        storage_context=storage_context,
        embed_model=embedding
    )

    # load from disk
    db2 = chromadb.PersistentClient(path=persistent_directory)
    chroma_collection = db2.get_or_create_collection('quickstart')
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embedding
    )

    # RAG
    query = '「たのしくまなび隊」としてリニューアルされた背景を日本語で簡潔にまとめてください。'
    query_engine = index.as_query_engine()
    answer = query_engine.query(query)

    # check answer
    print(answer.get_formatted_sources())
    print('query was:',query)
    print('answer:',answer)

if __name__=='__main__':
    # one file
    # main_onefile_VSI()
    main_onefile_chroma()

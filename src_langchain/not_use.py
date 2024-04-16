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


def main_onefile_beta():
    # using Chroma.from_documents; it works, but not useful.
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # load files
    filepath = os.path.join('data', filename)
    if filetype == 'pdf':
        loader = PyPDFLoader(filepath)
    elif filetype == 'word':
        loader = Docx2txtLoader(filepath)
    elif filetype == 'json':
        loader = JSONLoader(filepath)
    else:
        '''
        Excel, Powerpointの場合。
        langchain_communityのフレームワークを使って直接テキストを取得できない
        ファイルから一度テキストのみ抽出し、テキストファイルとして保存する
        そのテキストファイルをロードするならできそう。
        TextLoaderが良さそう。
        '''
        pass
    documents = loader.load_and_split()

    embedding=get_embedding()
    # index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])

    llm = get_llm()

    vectorstore = Chroma(
        collection_name='langchain',
        embedding_function=embedding,
        persist_directory='./chroma_db/'
    )
    # print(type(vectorstore))
    db = vectorstore.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory='./chroma_db/'
    )

    # 検索・参照先ファイルを出力するチェーンを作成
    retriever = db.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # チェーン実行
    query = '研究公募を実施する目的は何か?'
    print(chain({chain.question_key: query}))


def main_onefile_beta2():
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # get models
    embedding = get_embedding()
    llm = get_llm()

    # path to db
    persist_directory = './chroma_db_multi/'
    # chroma db settings
    client_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    # get instance of db
    db = Chroma(
        collection_name='langchain',
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory=persist_directory
    )

    # load files
    filepath = os.path.join('data', filename)
    if filetype == 'pdf':
        loader = PyPDFLoader(filepath)
    elif filetype == 'word':
        loader = Docx2txtLoader(filepath)
    elif filetype == 'json':
        loader = JSONLoader(filepath)
    else:
        '''
        Excel, Powerpointの場合。
        langchain_communityのフレームワークを使って直接テキストを取得できない
        ファイルから一度テキストのみ抽出し、テキストファイルとして保存する
        そのテキストファイルをロードするならできそう。
        TextLoaderが良さそう。
        '''
        pass
    documents = loader.load_and_split()

    db.add_documents(
        documents=documents,
        embedding=embedding
    )
    db.persist()

    # 検索・参照先ファイルを出力するチェーンを作成
    retriever = db.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # チェーン実行
    query = '研究公募を実施する目的は何か？'
    print(chain({chain.question_key: query}))

def main_multifiles_beta():
    # get models
    embedding = get_embedding()
    llm = get_llm()

    # path to db
    persist_directory = './chroma_db_multi/'
    # chroma db settings
    client_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=persist_directory,
        anonymized_telemetry=False
    )
    # get instance of db
    db = Chroma(
        collection_name='langchain',
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory=persist_directory
    )

    # get source files
    target_path = os.path.join('data')
    list_pdf = [f for f in os.listdir(target_path) if f.endswith('.pdf')]

    for pdf in list_pdf:
        # path指定
        filepath = os.path.join(target_path, pdf)
        # ファイルをロード
        loader = PyPDFLoader(filepath)
        # langchain_core.documents.base.Document生成
        documents = loader.load_and_split()
        # add db
        db.add_documents(
            documents=documents,
            embedding=embedding
        )

    # persist db
    db.persist()

    # 検索・参照先ファイルを出力するチェーンを作成
    retriever = db.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    # チェーン実行
    query = '研究公募を実施する目的は何か？'
    print(chain({chain.question_key: query}))

    # チェーン実行
    query = '日本における高精度遺伝子診断導入の早期立ち上げの目的で買収した企業はどこ？'
    print(chain({chain.question_key: query}))


def chat_from_db():
    # get models
    embedding = get_embedding()

    llm = get_llm()

    client_settings = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='./chroma_db_separate',
        anonymized_telemetry=False
    )

    db = Chroma(
        collection_name='langchain',
        embedding_function=embedding,
        client_settings=client_settings,
        persist_directory='./chroma_db_separate'
    )

    retriever = db.as_retriever()
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever
    )

    query='研究公募を実施する目的は何か?'
    print(chain({chain.question_key: query}))

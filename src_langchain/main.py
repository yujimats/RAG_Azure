import os
import yaml
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import Chroma

# from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain.text_splitter import CharacterTextSplitter

from chromadb.config import Settings
import chromadb

from utils import get_llm, get_embedding

# get env
load_dotenv('./.ENV')

def main_onefile():
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # get models
    embedding = get_embedding()
    llm = get_llm()

    # path to db
    persist_directory = './chroma_db_onefile/'
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

    # chunk setting
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        # separator = '。',
        chunk_size=1000,
        chunk_overlap=10
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
    documents = loader.load_and_split(text_splitter=text_splitter)

    db.add_documents(
        documents=documents,
        embedding=embedding
    )
    db.persist()


    # 検索・参照先ファイルを出力するチェーンを作成
    retriever = db.as_retriever()

    # get prompt
    prompt = PromptTemplate.from_template("""
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
    """)

    # get answer
    completion = PromptTemplate.from_template("""
    answer:{content}
    total token:{token}
    """)

    # get chain
    chain_rag = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    chain_answer = completion

    # get answer
    query = '研究公募を実施する目的は何か?'
    answer = chain_rag.invoke(query)
    response = chain_answer.invoke({"content": answer.content, "token": answer.response_metadata['token_usage']['total_tokens']})
    print(response.text)

def main_multifiles():
    # get models
    embedding = get_embedding()
    llm = get_llm()

    # chunk settings
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        # separator = '。',
        chunk_size=1000,
        chunk_overlap=10
    )

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
        documents = loader.load_and_split(
            text_splitter=text_splitter
        )
        # add db
        db.add_documents(
            documents=documents,
            embedding=embedding
        )

    # persist db
    db.persist()

    # get retriever
    retriever = db.as_retriever()

    # setting prompt template
    prompt = PromptTemplate.from_template("""
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
    """)

    # get chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # get answer
    query = '研究公募を実施する目的は何か？'
    answer = chain.invoke(query)
    print(answer)

def make_db():
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # chunk settings
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        # separator = '。',
        chunk_size=1000,
        chunk_overlap=10
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
    documents = loader.load_and_split(
        text_splitter=text_splitter
    )

    embedding = get_embedding()

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

    db.add_documents(
        documents=documents,
        embedding=embedding
    )
    db.persist()

def chat_from_db():
    # get models
    embedding = get_embedding()
    llm = get_llm()

    # setting prompt template
    prompt = PromptTemplate.from_template("""
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
    """)

    # get retriever
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

    # get chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # get answer
    query='研究公募を実施する目的は何か?'
    answer = chain.invoke(query)
    print(answer)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main_onefile_withSource():
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # get models
    embedding = get_embedding()
    llm = get_llm()

    # path to db
    persist_directory = './chroma_db_onefile/'
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

    # chunk setting
    text_splitter = CharacterTextSplitter(
        separator='\n\n',
        # separator = '。',
        chunk_size=1000,
        chunk_overlap=10
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
    documents = loader.load_and_split(text_splitter=text_splitter)

    db.add_documents(
        documents=documents,
        embedding=embedding
    )
    db.persist()


    # 検索・参照先ファイルを出力するチェーンを作成
    retriever = db.as_retriever()

    # get prompt
    prompt = PromptTemplate.from_template("""
    あなたはcontextを参考に、questionに回答します。
    <context>{context}</context>
    <question>{question}</question>
    """)

    # get answer
    completion = PromptTemplate.from_template("""
    question:{question}
    answer:{content}
    total_token:{token}

    source:{source}, page {page}
    """)

    # get chain
    chain_rag_from_docs = (
        RunnablePassthrough.assign(content=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
    )
    chain_rag_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=chain_rag_from_docs)
    chain_answer = completion

    # get answer
    query = '研究公募を実施する目的は何か?'
    answer = chain_rag_with_source.invoke(query)

    response = chain_answer.invoke({
        "question":answer['question'],
        "content":answer['answer'].content,
        "token":answer['answer'].response_metadata['token_usage']['total_tokens'],
        "source":answer['context'][0].metadata['source'],
        "page":answer['context'][0].metadata['page']
    })
    print(response.text)

if __name__=='__main__':
    main_onefile_withSource()
    # main_onefile()
    # main_multifiles()

    # # make database
    # make_db()
    # # use database for RAG
    # chat_from_db()

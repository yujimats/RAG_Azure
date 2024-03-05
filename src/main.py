import os
import yaml
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI

# from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain

from langchain_openai import AzureOpenAIEmbeddings

# get env
load_dotenv()

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

def main_onefile():
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

    llm = AzureChatOpenAI(
        api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        api_key=os.environ['API_KEYS'],
        azure_deployment=os.environ['OPENAI_MODEL'],
        temperature=0
    )

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

def main_multifiles():
    target_path = os.path.join('data')
    list_pdf = [f for f in os.listdir(target_path) if f.endswith('.pdf')]

    embedding = get_embedding()

    llm = AzureChatOpenAI(
        api_version=os.environ['API_VERSION'],
        azure_endpoint=os.environ['API_ENDPOINT'],
        api_key=os.environ['API_KEYS'],
        azure_deployment=os.environ['OPENAI_MODEL'],
        temperature=0
    )

    vectorstore = Chroma(
        collection_name='langchain',
        embedding_function=embedding
        # persist_directory='./chroma_db_multi/'
    )

    for pdf in list_pdf:
        # path指定
        filepath = os.path.join(target_path, pdf)
        # ファイルをロード
        loader = PyPDFLoader(filepath)
        # langchain_core.documents.base.Document生成
        documents = loader.load_and_split()

        if os.path.exists('./chroma_db_multi/'):
            vectorstore.add_documents(
                documents=documents,
                embedding=embedding,
                persist_directory='./chroma_db_multi/'
            )
        else:
            db = vectorstore.from_documents(
                documents=documents,
                embedding=embedding,
                persist_directory='./chroma_db_multi/'
            )

    # filepath = os.path.join(target_path, list_pdf[0])
    # loader = PyPDFLoader(filepath)
    # documents = loader.load_and_split()
    # db = vectorstore.from_documents(
    #     documents=documents,
    #     embedding=embedding,
    #     persist_directory='./chroma_db_multi/'
    # )

    # filepath = os.path.join(target_path, list_pdf[1])
    # loader = PyPDFLoader(filepath)
    # documents = loader.load_and_split()
    # vectorstore.add_documents(
    #     documents=documents,
    #     embedding=embedding,
    #     persist_directory='./chroma_db_multi/'
    # )

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

if __name__=='__main__':
    # main_onefile()
    main_multifiles()

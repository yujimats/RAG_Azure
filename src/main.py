import os
import yaml
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
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

def main_fromPDF():
    # get configs
    with open(os.path.join('config', 'config.yml'), 'r') as yml:
        config = yaml.safe_load(yml)
    filetype = config['data']['filetype']
    filename = config['data']['filename']

    # load files
    filepath = os.path.join('data', filename)
    loader = PyPDFLoader(filepath)
    documents = loader.load_and_split()

    # data.jsonlをDataFrameで読み込み
    # path_data = os.path.join('data', 'data.jsonl')
    # df_data = pd.read_json(path_data, orient='records', lines=True)

    # # 'copyright'が気象庁のものだけピックアップ
    # df_data = df_data[df_data['copyright']=='気象庁']

    # # QuestionとAnswerを結合
    # df_data['QnA'] = df_data['Question'] + ' ' + df_data['Answer']

    # list_qna = df_data['QnA'].tolist()

    # loader = JSONLoader(
    #     file_path=path_data,
    #     jq_schema='Answer'
    # )

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

def main_fromlist():
    pass

if __name__=='__main__':
    main_fromPDF()
    main_fromlist()

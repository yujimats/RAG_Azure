import os

from dotenv import load_dotenv
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI

def main_1():
    # 環境変数読み込み
    load_dotenv()

    # ベクターストアの設定
    vectorstore = DocArrayInMemorySearch.from_texts(
        ["私の会社は、東京都の赤坂にあります。", "私の会社の最寄り駅は、溜池山王駅です。"],
        embedding=AzureOpenAIEmbeddings(
            openai_api_type=os.getenv("API_TYPE"),
            openai_api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("API_ENDPOINT"),
            openai_api_key=os.getenv("API_KEYS"),
            model="text-embedding-ada-002",
            azure_deployment=os.getenv("MODEL_NAME"),
            embedding_ctx_length=8191,
            chunk_size=1000,
            max_retries=2,
        ),
    )

    # 検索機能の設定
    retriever = vectorstore.as_retriever()

    template = """次の文脈（context）のみに基づいて質問（question）に答えてください。:
    {context}

    質問: {question}
    """

    # プロンプトテンプレートの設定
    prompt = ChatPromptTemplate.from_template(template)

    # モデルの設定
    model = AzureChatOpenAI(
        api_version="2023-05-15",
        azure_endpoint=os.getenv("API_ENDPOINT"),
        api_key=os.getenv("API_KEYS"),
        azure_deployment=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )

    # 出力パーサの設定
    output_parser = StrOutputParser()

    # 情報の取得と質問の処理
    setup_and_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )

    # LCELによるチェーンの作成と結果の取得
    chain = setup_and_retrieval | prompt | model | output_parser

    result = chain.invoke("私の会社はどこにありますか？")
    print(result)

def main_2():
    # 環境変数読み込み
    load_dotenv()

    # ベクターストアの設定
    vectorstore = FAISS.from_texts(
        ["私の会社は、東京都の赤坂にあります。", "私の会社の最寄り駅は、溜池山王駅です。"],
        embedding=AzureOpenAIEmbeddings(
            openai_api_type=os.getenv("API_TYPE"),
            openai_api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("API_ENDPOINT"),
            openai_api_key=os.getenv("API_KEYS"),
            model="text-embedding-ada-002",
            azure_deployment=os.getenv("MODEL_NAME"),
            embedding_ctx_length=8191,
            chunk_size=1000,
            max_retries=2,
        ),
    )

    # 検索機能の設定
    retriever = vectorstore.as_retriever()

    template = """次の文脈（context）のみに基づいて質問（question）に答えてください。:
    {context}

    質問: {question}
    """

    # プロンプトテンプレートの設定
    prompt = ChatPromptTemplate.from_template(template)

    # モデルの設定
    model = AzureChatOpenAI(
        api_version="2023-05-15",
        azure_endpoint=os.getenv("API_ENDPOINT"),
        api_key=os.getenv("API_KEYS"),
        azure_deployment=os.getenv("OPENAI_MODEL"),
        temperature=0,
    )

    # 出力パーサの設定
    output_parser = StrOutputParser()

    # LCELによるチェーンの作成と結果の取得
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | output_parser
    )

    result = chain.invoke("私の会社の最寄り駅は何駅ですか？")
    print(result)

if __name__=='__main__':
    # main_1()
    main_2()

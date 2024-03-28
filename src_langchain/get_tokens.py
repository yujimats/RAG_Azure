import os
import pandas as pd
import matplotlib.pyplot as plt
import tiktoken

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, JSONLoader

GPT_MODEL = "gpt-3.5-turbo"

def num_tokens(text, model=GPT_MODEL):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def save_fig(df, colum_name, path_save, title='token', xlabel='index', ylabel='token'):
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[colum_name], marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(path_save)

def get_token_fromDocuments(documents, model='text-embedding-ada-002'):
    token = 0
    encoding = tiktoken.encoding_for_model(model)
    for document in documents:
        token += len(encoding.encode(document.page_content))
    return token

def get_token_cost_fromDocuments(documents, cost_per_ktoken=0.0001, rate=150, model='text-embedding-ada-002', flag_yen=True):
    token = get_token_fromDocuments(documents, model)
    if flag_yen:
        # default: yen
        cost = token / 1000 * cost_per_ktoken * rate
    else:
        # dallers
        cost = token / 1000 * cost_per_ktoken
    return cost

if __name__ == '__main__':
    # dataフォルダ内のファイルを読み取り
    # pdfのみ対応、その他ファイルは後ほど対応
    target_dir = os.path.join('data')
    list_pdf = [f for f in os.listdir(target_dir) if f.endswith('.pdf')]

    token = 0; cost = 0
    for pdf in list_pdf:
        # path指定
        filepath = os.path.join(target_dir, pdf)
        # ファイルにロード
        loader = PyPDFLoader(filepath)
        # langchain_core.documents.base.Document生成
        documents = loader.load_and_split()
        token += get_token_fromDocuments(documents)
        cost += get_token_cost_fromDocuments(documents)

    print(f'total token: {token}')
    print(f'total cost: {cost}')

    # word; 動作確認無し
    target_dir = os.path.join('data')
    list_word = [f for f in os.listdir(target_dir) if f.endswith('.docx')]

    token = 0; cost = 0
    for word in list_word:
        # path指定
        filepath = os.path.join(target_dir, word)
        # ファイルにロード
        loader = Docx2txtLoader(filepath)
        # langchain_core.documents.base.Document生成
        documents = loader.load_and_split()
        token += get_token_fromDocuments(documents)
        cost += get_token_cost_fromDocuments(documents)

    # json: 動作確認なし
    target_dir = os.path.join('data')
    list_json = [f for f in os.listdir(target_dir) if f.endswith('.json')]

    token = 0; cost = 0
    for json in list_json:
        # path指定
        filepath = os.path.join(target_dir, json)
        # ファイルにロード
        loader = JSONLoader(filepath)
        # langchain_core.documents.base.Document生成
        documents = loader.load_and_split()
        token += get_token_fromDocuments(documents)
        cost += get_token_cost_fromDocuments(documents)


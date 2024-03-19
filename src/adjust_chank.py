import os
import yaml
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI

# from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain

# from langchain_text_splitter import TextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter

from langchain_openai import AzureOpenAIEmbeddings

from chromadb.config import Settings

# get configs
with open(os.path.join('config', 'config.yml'), 'r') as yml:
    config = yaml.safe_load(yml)

# filename = config['data']['filename']
# filename = '140120230522578536.pdf' # 48
filename = '140120230706518488.pdf'

filepath = os.path.join('data', filename)
loader = PyPDFLoader(filepath)

documents = loader.load_and_split()

# print(type(loader))
print(len(documents))

# セパレータ関係なしに分割
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

documents = loader.load_and_split(text_splitter=text_splitter)

print(len(documents))

# セパレータで分割して文字数をマージするTextSplitter
# セパレータには句読点など指定可能。
# 関連性の高い隣同士の文字がセットになりやすい。
text_splitter = CharacterTextSplitter(
    separator='\n\n',
    # separator='。',
    chunk_size=1000,
    chunk_overlap=100
)
documents = loader.load_and_split(text_splitter=text_splitter)

print(len(documents))

'''
分割したchunkサイズが小さくなったとき、
検索後にllmに投入するchunk数は増やしたほうが、
より多くの情報を参照することができるため良い。
方法はおそらくdb.as_retriver()の引数で設定する方法

使用方法の例は以下の通り。
実際の応答でどう変化するかを見る

```python
# Retrieve more documents with higher diversity
# Useful if your dataset has many similar documents
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 6, 'lambda_mult': 0.25}
)

# Fetch more documents for the MMR algorithm to consider
# But only return the top 5
docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 5, 'fetch_k': 50}
)

# Only retrieve documents that have a relevance score
# Above a certain threshold
docsearch.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={'score_threshold': 0.8}
)

# Only get the single most similar document from the dataset
docsearch.as_retriever(search_kwargs={'k': 1})

# Use a filter to only retrieve documents from a specific paper
docsearch.as_retriever(
    search_kwargs={'filter': {'paper_title':'GPT-4 Technical Report'}}
)
```
'''


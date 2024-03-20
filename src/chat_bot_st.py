import os
import streamlit as st
from streamlit_chat import message
import yaml
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, JSONLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureChatOpenAI

# from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQAWithSourcesChain

from langchain_openai import AzureOpenAIEmbeddings

from chromadb.config import Settings

from main import get_embedding, get_llm

# get env
load_dotenv('./.ENV')

# get configs
with open(os.path.join('config', 'config.yml'), 'r') as yml:
    config = yaml.safe_load(yml)
filetype = config['data']['filetype']
filename = config['data']['filename']

# get models
embedding = get_embedding()
llm = get_llm()

# path to db
persist_directory = './chroma_db_streamllit/'
#chroma db settings
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
loader = PyPDFLoader(filepath)
documents = loader.load_and_split()

# add db
db.add_documents(
    documents=documents,
    embedding=embedding
)
db.persist()

# 検索参照先ファイルを出力するインスタンス
retriever = db.as_retriever()
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

def conversational_chat(query):
    # get result
    result = chain({'question': query, 'chat_history': st.session_state['history']})
    st.session_state['history'].append({query, result['answer']})

    return result['answer']

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ['welcome to chat bot!']

if 'past' not in st.session_state:
    st.session_state['past'] = ['hi!']

# make container to display the chat history
response_container = st.container()
# make container to display the user's input and the response from the chat
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input('Input:', placeholder='Please enter your message.', key='input')
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output = conversational_chat(user_input)

        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style='big-smile')
                message(st.session_state['generated'][i], key=str(i), avatar_style='thumbs')

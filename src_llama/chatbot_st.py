import os
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv

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

# get env
load_dotenv('./.ENV')

# get models
llm = get_llm()
embedding = get_embedding()

# Settings
Settings.llm = llm
Settings.embed_model = embedding

# get files
target_path = os.path.join('data')
list_pdffiles = [f for f in os.listdir(target_path) if f.endswith('.pdf')]

# save to disk
persistent_directory = './chroma_db_streamlit'
db = chromadb.PersistentClient(path=persistent_directory)
chroma_collection = db.get_or_create_collection('quickstart')
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# add data
for pdffile in list_pdffiles:
    document = SimpleDirectoryReader(
        input_files=[os.path.join(target_path, pdffile)]
    ).load_data()

    index = VectorStoreIndex.from_documents(
        document,
        storage_context=storage_context,
        embed_model=embedding
    )

# get index
query_engine = index.as_query_engine()

def conversational_chat(query):
    # get result
    answer = query_engine.query(query)
    answer = str(answer) # change to str
    st.session_state['history'].append({query, answer})
    return answer

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

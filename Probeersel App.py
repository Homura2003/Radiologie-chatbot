from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st

from langchain.chat_models import ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

st.title('Radiologie chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})

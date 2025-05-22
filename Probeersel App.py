# Importing langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain.embeddings import OpenAIEmbeddings  # <-- vervangen
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Streamlit voor UI
import streamlit as st

# OpenAI LLM toevoegen
from langchain.chat_models import ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

st.title('Radiologie chatbot')

prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
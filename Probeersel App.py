from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import Ollama  

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

llm = Ollama(
    model='mistral:instruct',
    base_url='http://localhost:11434',  # Of jouw externe serveradres
    temperature=0.5,
    num_ctx=2048
)

st.title('Radiologie chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    response = llm(prompt)
    st.chat_message('assistant').markdown(response)
    st.session_state.message.append(
        {'role':'assistant', 'content':response})
    

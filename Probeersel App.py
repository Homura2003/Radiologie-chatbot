from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

import streamlit as st
from langchain.chat_models import ChatOpenAI

import os
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_KEY"]

llm = ChatOpenAI(temperature=0.5)

st.title('Radiologie chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Pass your prompt here')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    messages = [HumanMessage(content=prompt)]
    response = llm(messages)
    
    st.chat_message('assistant').markdown(response.content)
    st.session_state.messages.append(
        {'role':'assistant', 'content':response.content})

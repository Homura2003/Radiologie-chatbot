from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

import os
os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/google/gemma-2b-it",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    do_sample=True,
    max_new_tokens=256
)

st.title('Radiologie chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Stel hier je vraag')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})
    
    try:
        response = llm.invoke(prompt)
        
        st.chat_message('assistant').markdown(response)
        st.session_state.messages.append(
            {'role':'assistant', 'content':response})
    except Exception as e:
        error_message = f"Er is een fout opgetreden: {str(e)}"
        st.chat_message('assistant').markdown(error_message)
        st.session_state.messages.append(
            {'role':'assistant', 'content':error_message})
    



    


    

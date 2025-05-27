from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from transformers import pipeline, Conversation

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

import os
os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

chatbot = pipeline(
    "conversational", 
    model="BramVanroy/GEITje-7B-ultra", 
    model_kwargs={
        "load_in_8bit": True, 
        "attn_implementation": "flash_attention_2"
    }, 
    device_map="auto"
)

st.title('Radiologie chatbot')

if 'conversation' not in st.session_state:
    st.session_state.conversation = Conversation([
        {"role": "system", "content": "Je bent een behulpzame assistent gespecialiseerd in radiologie."}
    ])

for message in st.session_state.conversation.messages:
    if message["role"] != "system":  
        st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input('Stel hier je vraag')

if prompt:
    st.chat_message('user').markdown(prompt)
    
    try:
        st.session_state.conversation.add_user_input(prompt)
        
        st.session_state.conversation = chatbot(st.session_state.conversation)
        
        response = st.session_state.conversation.messages[-1]["content"]
        
        st.chat_message('assistant').markdown(response)
        
    except Exception as e:
        error_message = f"Er is een fout opgetreden: {str(e)}"
        st.chat_message('assistant').markdown(error_message)


    




    


    

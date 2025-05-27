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

MODEL_DIR='yhavinga/gpt2-large-dutch'
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
generator = pipeline('text-generation', model, tokenizer=tokenizer)

generated_text = generator('Het eiland West-', max_length=100, do_sample=True, top_k=40, top_p=0.95, repetition_penalty=2.0))
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

    




    


    

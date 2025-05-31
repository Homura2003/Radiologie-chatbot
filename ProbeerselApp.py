from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

import os
import requests

os.environ["HUGGINGFACE_API_KEY"] = "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"

API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}"}

def query(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Controleer op HTTP-fouten
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"HTTP-fout: {e}")
        return {"error": str(e)}
    except ValueError as e:
        print(f"JSON-fout: {e}")
        return {"error": "Ongeldige JSON-respons"}

llm = HuggingFaceEndpoint(
    repo_id="TheBloke/Llama-2-13B-Chat-Dutch-GPTQ",
    task="text-generation",
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    max_new_tokens=512
)

prompt = "Wat is radiologie?"
response = llm.invoke(prompt)
print(response)

st.title('Radiologie chatbot')

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
prompt = st.chat_input('Stel hier je vraag')

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    try:
        response = query({"inputs": prompt})
        assistant_response = response.get("generated_text", "Geen antwoord ontvangen.")
        
        st.chat_message('assistant').markdown(assistant_response)
        st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
    except Exception as e:
        error_message = f"Er is een fout opgetreden: {str(e)}"
        print(f"Foutdetails: {error_message}")  # Log foutdetails in de terminal
        st.chat_message('assistant').markdown(error_message)
        st.session_state.messages.append({'role': 'assistant', 'content': error_message})










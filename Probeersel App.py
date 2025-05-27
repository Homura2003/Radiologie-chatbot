from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from transformers import pipeline, TFAutoModel, AutoTokenizer

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

import os
os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

try:
    model_name = "GroNLP/gpt2-small-dutch"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModel.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        framework="tf"
    )

    st.title('Radiologie chatbot')

    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "Je bent een behulpzame assistent gespecialiseerd in radiologie."}
        ]

    for message in st.session_state.messages:
        if message["role"] != "system":
            st.chat_message(message["role"]).markdown(message["content"])

    prompt = st.chat_input('Stel hier je vraag')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            conversation_text = "\n".join([
                f"{'Assistant' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
                for msg in st.session_state.messages
            ])
            
            response = pipe(
                conversation_text,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95
            )[0]['generated_text']
            
            response = response.split("Assistant: ")[-1].strip()
            
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Er is een fout opgetreden: {str(e)}"
            st.chat_message('assistant').markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

except Exception as e:
    st.error(f"Er is een fout opgetreden bij het initialiseren van de applicatie: {str(e)}")



    




    


    

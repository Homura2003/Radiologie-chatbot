from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import retrieval_qa
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, AIMessage
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint

import os
os.environ["HUGGINGFACE_API_KEY"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

try:
    torch.random.manual_seed(0)
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3.5-mini-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True, 
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
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
            generation_args = {
                "max_new_tokens": 500,
                "return_full_text": False,
                "temperature": 0.0,
                "do_sample": False,
            }
            
            response = pipe(st.session_state.messages, **generation_args)[0]['generated_text']
            
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            error_message = f"Er is een fout opgetreden: {str(e)}"
            st.chat_message('assistant').markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

except Exception as e:
    st.error(f"Er is een fout opgetreden bij het initialiseren van de applicatie: {str(e)}")

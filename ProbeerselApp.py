from optimum.transformers import GPTQConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
import streamlit as st

# Installeer optimum als het nog niet is geïnstalleerd:
pip install optimum

# Laad het model en de tokenizer
MODEL_NAME = "TheBloke/Llama-2-13B-Chat-Dutch-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=GPTQConfig()  # Specificeer GPTQ-configuratie
)

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
        # Tokenize de input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Genereer een antwoord
        outputs = model.generate(inputs["input_ids"], max_length=512, temperature=0.7, top_p=0.95)
        assistant_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.chat_message('assistant').markdown(assistant_response)
        st.session_state.messages.append({'role': 'assistant', 'content': assistant_response})
    except Exception as e:
        error_message = f"Er is een fout opgetreden: {str(e)}"
        print(f"Foutdetails: {error_message}")  # Log foutdetails in de terminal
        st.chat_message('assistant').markdown(error_message)
        st.session_state.messages.append({'role': 'assistant', 'content': error_message})










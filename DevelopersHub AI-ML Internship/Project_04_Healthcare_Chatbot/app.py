from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

# Load API keys
load_dotenv()

azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# 🛡️ Safety Filter
def is_safe_query(text):
    unsafe_keywords = ["prescribe", "diagnose", "dosage", "dose", "treat", "medicine for", "can I take", "should I take"]
    return not any(word in text.lower() for word in unsafe_keywords)

# 🧠 Prompt Template with safety tone
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly and helpful medical assistant. You can answer general health-related questions, but you do not provide medical advice, prescriptions, or diagnoses."),
    ("user", "Question: {question}")
])

# GPT setup
llm = AzureChatOpenAI(
    openai_api_version="2024-02-15-preview",
    azure_endpoint=azure_endpoint,
    azure_deployment=azure_deployment,
    api_key=azure_api_key
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit App
st.title("🏥 General Health Query Chatbot")
st.write("Ask me general health-related questions! (No medical advice or prescriptions)")

user_input = st.text_input("Enter your health question:")

if user_input:
    if not is_safe_query(user_input):
        st.warning("⚠️ Sorry, I cannot provide advice on medication, dosage, or treatment. Please consult a doctor.")
    else:
        with st.spinner("Thinking..."):
            response = chain.invoke({"question": user_input})
            st.success("Answer:")
            st.write(response)



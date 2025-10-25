from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()


# Initialize the LLM with caching for performance
@st.cache_resource
def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in environment variables!")
        st.error("Please make sure you have a .env file with your Groq API key.")
        st.stop()
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=api_key
    )


llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

def get_ai_response(user_message: str) -> str:
    messages = [
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessage(content=user_message)
    ]
    response = llm.invoke(messages)
    return response.content

def get_ai_responseSLIT(user_message: str, llm) -> str:
    messages = [
        SystemMessage(content="You are a helpful AI assistant. Answer the user's questions clearly and concisely."),
        HumanMessage(content=user_message),
    ]
    response = llm.invoke(messages)
    return response.content
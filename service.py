from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()

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
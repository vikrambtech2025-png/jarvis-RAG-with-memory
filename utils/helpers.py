import os
from langchain_core.messages import HumanMessage, AIMessage

def format_chat_history(messages):
    """Convert streamlit messages to LangChain message objects."""
    history = []
    for msg in messages:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    return history

def check_env_vars():
    """Check if all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    return missing_vars

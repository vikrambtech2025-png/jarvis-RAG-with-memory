# Jarvis Lite — AI-Powered RAG Chatbot with Memory

A Streamlit **RAG chatbot with conversation memory** — upload documents and chat with them while the assistant remembers the ongoing conversation. Built with LangChain, FAISS, and OpenAI.

## How it works

```
Upload docs → chunk & embed → store in FAISS vector index
    ↓
User query → retrieve relevant chunks (+ memory context) → LLM → answer
```

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Set your `OPENAI_API_KEY` in a `.env` file (see `requirements.txt` / `config.py`), then ask questions about your documents.

## Features

- **Document Q&A** — chat over your uploaded PDFs/docs with grounded answers
- **Conversation memory** — the assistant recalls earlier turns
- **Streamlit UI** — simple chat interface

## Stack

Streamlit · LangChain (openai, community, text-splitters) · FAISS · pypdf · tiktoken

## License

No license specified — for learning/reference use.
# 🤖 Jarvis Lite: AI-powered RAG Chatbot with Memory

## Project Title & Description

Jarvis Lite is an AI-powered Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate answers to user queries based on uploaded documents (PDF/Text). It leverages LangChain for orchestration, FAISS for vector storage, and OpenAI embeddings for document understanding. The chatbot maintains conversation memory, allowing for a more natural and continuous interaction. The user interface is built with Streamlit, offering a clean and intuitive experience for document upload and chat.

## ✨ Features

*   **Document Upload**: Easily upload PDF and plain text files for the chatbot to ingest.
*   **Intelligent RAG Pipeline**: Utilizes LangChain to process documents, generate embeddings, and retrieve relevant information.
*   **Conversation Memory**: Maintains chat history to provide context-aware responses.
*   **Streamlit UI**: A clean, interactive web interface for seamless user interaction.
*   **Modular Codebase**: Organized into `app.py`, `rag_pipeline.py`, `config.py`, and `utils/` for maintainability and scalability.
*   **Error Handling**: Robust error handling for file processing and API interactions.
*   **Real-Time Interaction**: Provides instant responses with loading indicators for a smooth user experience.

## 🚀 Tech Stack

*   **Python**: Main programming language.
*   **LangChain**: For orchestrating the RAG pipeline and managing conversation memory.
*   **FAISS**: Efficient similarity search library for vector database storage.
*   **OpenAI**: For generating embeddings (`text-embedding-3-small`) and powering the Large Language Model (`gpt-4.1-mini`).
*   **Streamlit**: For building the interactive web user interface.
*   **python-dotenv**: For managing environment variables.
*   **pypdf**: For loading PDF documents.
*   **tiktoken**: For tokenization.

## ⚙️ Installation Steps

1.  **Clone the repository (or create the project structure manually):**

    ```bash
    git clone https://github.com/your_username/jarvis_lite.git
    cd jarvis_lite
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**

    Create a `.env` file in the root directory of the project based on `.env.example` and add your OpenAI API key:

    ```ini
    # .env
    OPENAI_API_KEY=your_openai_api_key_here
    ```

    Replace `your_openai_api_key_here` with your actual OpenAI API key.

## 💡 Usage Instructions

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**

    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Upload Documents:**

    *   Use the sidebar to upload PDF or Text files.
    *   Click the "🚀 Process Documents" button to ingest the files. This will create embeddings and store them in a FAISS vector database.

4.  **Start Chatting:**

    *   Once documents are processed, type your questions in the chat input box at the bottom of the page.
    *   The chatbot will provide answers based on the content of your uploaded documents, maintaining conversation memory.

5.  **Clear Chat History:**

    *   Use the "🗑️ Clear Chat History" button in the sidebar to reset the conversation.

## 🏛️ Architecture Diagram

```text
User → Streamlit UI → Backend (LangChain) → Embeddings → FAISS → LLM → Response
```

## 🌐 Deployment

### Render

1.  **`requirements.txt`**: The provided `requirements.txt` file is ready for use.

2.  **Start Command**: 
    ```bash
    streamlit run app.py --server.port $PORT
    ```

### Hugging Face Spaces

1.  **`app.py` compatibility**: The `app.py` is fully compatible with Hugging Face Spaces.

2.  **Environment Setup**: Add your `OPENAI_API_KEY` as a Space Secret (named `OPENAI_API_KEY`) in the "Settings" tab.

## 📸 Screenshots

*(Placeholder for future screenshots of the application UI)*

## 🔮 Future Improvements

*   **Support for more document types**: Extend document loaders to include DOCX, CSV, etc.
*   **Persistent Vector Store**: Implement saving and loading of the FAISS vector store.
*   **User Authentication**: Add user login to manage personal document collections.

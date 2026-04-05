import os
import tempfile
from typing import List, Any

# Updated LangChain 0.3.x imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from config import CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, EMBEDDING_MODEL

class RAGPipeline:
    def __init__(self):
        # Initialize components with updated models
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.vector_db = None
        self.retriever = None
        self.rag_chain = None

    def process_documents(self, uploaded_files):
        """Process uploaded files, create embeddings, and store in FAISS."""
        all_docs = []
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                if uploaded_file.name.lower().endswith('.pdf'):
                    loader = PyPDFLoader(tmp_path)
                elif uploaded_file.name.lower().endswith('.txt'):
                    loader = TextLoader(tmp_path)
                else:
                    continue
                
                all_docs.extend(loader.load())
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        if not all_docs:
            return 0

        # Modern text splitter from langchain-text-splitters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(all_docs)

        # Vector store creation remains standard but using community FAISS
        self.vector_db = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_db.as_retriever()
        
        # Setup modern RAG Chain with Memory
        self._setup_rag_chain()
        
        return len(splits)

    def _setup_rag_chain(self):
        """Set up the history-aware retrieval chain using latest LangChain patterns."""
        
        # 1. Contextualize Question Prompt (Updated for latest message placeholders)
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Modern chain creation: history_aware_retriever
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # 2. Answer Question Prompt
        system_prompt = (
            "You are Jarvis Lite, an AI assistant. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer, say that you don't know. "
            "Keep the answer concise and professional."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # Modern chain creation: stuff_documents_chain and retrieval_chain
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_response(self, user_input: str, chat_history: List[Any]):
        """Generate response using the modern RAG chain."""
        if not self.rag_chain:
            return "Please upload documents first to initialize the RAG system."
        
        # invoke is the standard method in LangChain 0.3.x
        response = self.rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        return response["answer"]
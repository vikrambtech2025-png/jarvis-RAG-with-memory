import streamlit as st
import os
from rag_pipeline import RAGPipeline
from utils.helpers import format_chat_history, check_env_vars

# Page Configuration
st.set_page_config(
    page_title="Jarvis Lite - AI RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .main {
        background-color: #f5f7f9;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "processed" not in st.session_state:
    st.session_state.processed = False

# Sidebar for Configuration and File Upload
with st.sidebar:
    st.title("⚙️ Jarvis Lite Settings")
    
    # Check Environment Variables
    missing_vars = check_env_vars()
    if missing_vars:
        st.error(f"Missing API Key: {', '.join(missing_vars)}")
        st.info("Please set the OPENAI_API_KEY in your environment.")
        # Optional: Allow user to input key manually if not in env
        user_key = st.text_input("Enter OpenAI API Key", type="password")
        if user_key:
            os.environ["OPENAI_API_KEY"] = user_key
            st.rerun()
        else:
            st.stop()
    
    st.divider()
    
    st.subheader("📄 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF or Text files", 
        type=["pdf", "txt"], 
        accept_multiple_files=True
    )
    
    if st.button("🚀 Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents and creating vector store..."):
                try:
                    num_chunks = st.session_state.rag_pipeline.process_documents(uploaded_files)
                    if num_chunks > 0:
                        st.session_state.processed = True
                        st.success(f"Success! Processed {len(uploaded_files)} files into {num_chunks} chunks.")
                    else:
                        st.error("No valid text found in uploaded files.")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
        else:
            st.warning("Please upload at least one file first.")

    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Chat Interface
st.title("🤖 Jarvis Lite: AI RAG Chatbot")
st.caption("Ask questions based on your uploaded documents. Powered by LangChain & FAISS.")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        if not st.session_state.processed:
            response = "I'm ready, but I need some documents to learn from! Please upload and process them in the sidebar."
            st.markdown(response)
        else:
            with st.spinner("Thinking..."):
                try:
                    # Format history for LangChain
                    history = format_chat_history(st.session_state.messages[:-1])
                    
                    # Get response from RAG Pipeline
                    response = st.session_state.rag_pipeline.get_response(prompt, history)
                    st.markdown(response)
                except Exception as e:
                    response = f"Oops! I encountered an error: {str(e)}"
                    st.error(response)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_DB_PATH = "vector_db"

# Model Configuration
LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

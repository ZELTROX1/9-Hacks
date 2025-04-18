from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import faiss
import pickle
import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatGroq
from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
GROQ_MODEL = "llama3-8b-8192"  # or "mixtral-8x7b-32768"
FAISS_INDEX_PATH = "./vectorstore/faiss_index"
DOCUMENTS_PATH = "./vectorstore/documents.txt"
EMBEDDING_DIM = 512  # Depends on how you built your index
DATABASE_URL = "sqlite:///./chat_history.db"

# === FastAPI app ===
app = FastAPI()

# === SQLAlchemy Database Setup ===
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === Load FAISS Vector Store ===
def load_faiss():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(DOCUMENTS_PATH, "rb") as f:
        documents = pickle.load(f)

    # Create FAISS Langchain wrapper
    vectorstore = FAISS(embedding_function=dummy_embed, index=index, docstore=documents, index_to_docstore_id=None)
    return vectorstore

# === Dummy Embedder (replace with real one if needed) ===
def dummy_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    embeddings = []
    for text in texts:
        vector = np.array([hash(text) % 1000 / 1000.0] * EMBEDDING_DIM, dtype=np.float32)
        embeddings.append(vector)
    return embeddings

vectorstore = load_faiss()

# === Groq LLM Setup ===
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_MODEL)

# === Pydantic Request Schema ===
class ChatRequest(BaseModel):
    message: str

# === Chat Endpoint ===
@app.post("/chat")
def chat(req: ChatRequest, db=Depends(get_db)):
    user_message = req.message

    # 1. Search Top K documents
    docs = vectorstore.similarity_search(user_message, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # 2. Create Prompt
    template = ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant. Use the following context:\n{context}"),
        ("user", "{user_input}")
    ])
    prompt = template.format_messages(user_input=user_message)

    # 3. Get LLM response
    bot_response = llm.invoke(prompt).content

    # 4. Save to SQL
    chat_entry = ChatHistory(user_message=user_message, bot_response=bot_response)
    db.add(chat_entry)
    db.commit()

    return {"response": bot_response}

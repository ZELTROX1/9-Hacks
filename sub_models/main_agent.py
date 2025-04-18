from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from dotenv import load_dotenv
from medical_agent.ocr import process_document_with_ocr

# Import the MedicalAgent class
from medical_agent.med import MedicalAgent

load_dotenv()

app = FastAPI(title="Medical AI Assistant API", description="API for medical consultation and analysis")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
medical_agent = None
general_llm = None
embeddings = None
vector_stores = {}  # Dictionary to store multiple vector databases

# Response models
class ChatResponse(BaseModel):
    response: str

class MedicalResponse(BaseModel):
    analysis: str

class VectorDBResponse(BaseModel):
    db_id: str
    message: str

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]

# Initialize the medical agent and general chatbot
@app.on_event("startup")
async def startup_event():
    global medical_agent, general_llm, embeddings
    
    # Initialize Medical Agent
    medical_agent = MedicalAgent()
    
    # Initialize general Groq LLM for chatbot
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    general_llm = ChatGroq(
        api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.3,
        max_tokens=512
    )
    
    # Initialize embeddings for vector database
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

# General chatbot endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(query: str = Form(...)):
    """General chatbot endpoint for non-medical queries"""
    try:
        if not general_llm:
            raise HTTPException(status_code=500, detail="Chatbot not initialized")
            
        response = general_llm.invoke(query)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Medical analysis endpoint
@app.post("/api/medical", response_model=MedicalResponse)
async def medical_analysis(
    query: str = Form(...),
    documents: Optional[List[UploadFile]] = File(None),
    xray_images: Optional[List[UploadFile]] = File(None)
):
    """Medical analysis endpoint for medical queries with optional documents and X-ray images"""
    try:
        if not medical_agent:
            raise HTTPException(status_code=500, detail="Medical agent not initialized")
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            doc_paths = []
            xray_paths = []
            
            # Process documents
            if documents:
                for doc in documents:
                    if doc.content_type not in ["application/pdf", "text/plain", "image/jpeg", "image/png"]:
                        raise HTTPException(status_code=400, detail=f"Unsupported document type: {doc.content_type}")
                    
                    file_path = Path(temp_dir) / doc.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(doc.file, buffer)
                    doc_paths.append(str(file_path))
            
            # Process X-ray images
            if xray_images:
                for img in xray_images:
                    if img.content_type not in ["image/jpeg", "image/png"]:
                        raise HTTPException(status_code=400, detail=f"Unsupported image type: {img.content_type}")
                    
                    file_path = Path(temp_dir) / img.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(img.file, buffer)
                    xray_paths.append(str(file_path))
            
            # Run medical analysis
            analysis = medical_agent.run(
                query=query,
                document_paths=doc_paths if documents else None,
                xray_images=xray_paths if xray_images else None
            )
            
            return {"analysis": analysis}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Create vector database endpoint
@app.post("/api/create-vectordb", response_model=VectorDBResponse)
async def create_vector_db(
    documents: List[UploadFile] = File(...),
    db_name: Optional[str] = Form(None)
):
    """Create a vector database from uploaded documents"""
    try:
        # Generate a unique ID for the database if name not provided
        db_id = db_name if db_name else str(uuid.uuid4())[:8]
        
        if db_id in vector_stores:
            raise HTTPException(status_code=400, detail=f"Database with ID {db_id} already exists")
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            all_documents = []
            
            # Process each uploaded document
            for doc in documents:
                file_path = Path(temp_dir) / doc.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(doc.file, buffer)
                
                # Load document based on file type
                if doc.content_type == "application/pdf":
                    loader = process_document_with_ocr(str(file_path))
                elif doc.content_type == "text/plain":
                    loader = TextLoader(str(file_path))
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported document type: {doc.content_type}")
                
                docs = loader.load()
                all_documents.extend(docs)
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(all_documents)
            
            # Create vector store
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            vector_stores[db_id] = vectorstore
            
            # Save the vector store to disk (optional)
            db_path = f"vectorstores/{db_id}"
            os.makedirs(db_path, exist_ok=True)
            vectorstore.save_local(db_path)
            
            return {
                "db_id": db_id,
                "message": f"Vector database created successfully with {len(split_docs)} chunks"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Query vector database endpoint
@app.post("/api/query-vectordb/{db_id}", response_model=QueryResponse)
async def query_vector_db(
    db_id: str,
    query: str = Form(...)
):
    """Query a specific vector database"""
    try:
        if db_id not in vector_stores:
            # Try to load from disk if not in memory
            db_path = f"vectorstores/{db_id}"
            if os.path.exists(db_path):
                vectorstore = FAISS.load_local(db_path, embeddings)
                vector_stores[db_id] = vectorstore
            else:
                raise HTTPException(status_code=404, detail=f"Vector database with ID {db_id} not found")
        
        vectorstore = vector_stores[db_id]
        
        # Create a retrieval chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=general_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        # Query the vector store
        result = qa_chain({"query": query})
        
        # Format source documents
        source_docs = []
        for doc in result.get("source_documents", []):
            source_docs.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "answer": result["result"],
            "source_documents": source_docs
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# List available vector databases
@app.get("/api/list-vectordbs")
async def list_vector_dbs():
    """List all available vector databases"""
    try:
        # Check both in-memory and on-disk databases
        memory_dbs = list(vector_stores.keys())
        
        # Check disk storage
        disk_dbs = []
        if os.path.exists("vectorstores"):
            disk_dbs = [d for d in os.listdir("vectorstores") if os.path.isdir(os.path.join("vectorstores", d))]
        
        all_dbs = list(set(memory_dbs + disk_dbs))
        
        return {"databases": all_dbs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Delete vector database endpoint
@app.delete("/api/delete-vectordb/{db_id}")
async def delete_vector_db(db_id: str):
    """Delete a specific vector database"""
    try:
        # Remove from memory
        if db_id in vector_stores:
            del vector_stores[db_id]
        
        # Remove from disk
        db_path = f"vectorstores/{db_id}"
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        
        return {"message": f"Vector database {db_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and services are initialized"""
    health_status = {
        "status": "healthy",
        "medical_agent": bool(medical_agent),
        "general_llm": bool(general_llm),
        "embeddings": bool(embeddings),
        "vector_dbs": len(vector_stores)
    }
    
    if not health_status["medical_agent"] or not health_status["general_llm"] or not health_status["embeddings"]:
        health_status["status"] = "unhealthy"
        
    return health_status

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
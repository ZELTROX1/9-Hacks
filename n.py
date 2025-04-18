from fastapi import FastAPI, File, UploadFile, HTTPException, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import shutil
import tempfile
import uuid
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document

# Import the MedicalAgent class
from medical_agent.med import MedicalAgent
from medical_agent.ocr import process_document_with_ocr 

load_dotenv()

app = FastAPI(title="Multi-Agent AI System", description="API for intelligent agent-based analysis and consultation")

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
db_metadata = {}  # Dictionary to store metadata about each vector database

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
    db_used: Optional[str] = None
    
class ManagerResponse(BaseModel):
    answer: str
    source_documents: Optional[List[dict]] = None
    db_used: Optional[str] = None
    reasoning: str

# Custom manager agent class
class ManagerAgent:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        
    def get_embedding(self, text):
        """Get embedding for a piece of text"""
        return self.embeddings.embed_query(text)
    
    def calculate_similarity(self, query_embedding, db_id, sample_size=5):
        """Calculate similarity between query and database documents"""
        if db_id not in vector_stores:
            return 0
        
        vectorstore = vector_stores[db_id]
        # Get a sample of documents from the database to compare with
        docs_with_scores = vectorstore.similarity_search_with_score(
            query="", k=sample_size  # Empty query to get random samples
        )
        
        if not docs_with_scores:
            return 0
            
        # Average similarity score
        avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
        return avg_score
    
    def select_best_db(self, query: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Select best vector database for the query"""
        query_embedding = self.get_embedding(query)
        
        if not vector_stores:
            return None
            
        # Calculate similarity scores for each database
        scores = {}
        for db_id in vector_stores:
            # Basic similarity comparison
            score = self.calculate_similarity(query_embedding, db_id)
            
            # Add extra weight if database name matches keywords in query
            if db_id.lower() in query.lower():
                score += 0.2
                
            # Add extra weight based on database metadata if available
            if db_id in metadata and "description" in metadata[db_id]:
                desc = metadata[db_id]["description"]
                for keyword in query.lower().split():
                    if keyword in desc.lower():
                        score += 0.1
            
            scores[db_id] = score
            
        # Return the database with highest score if it's above threshold
        if scores:
            best_db = max(scores.items(), key=lambda x: x[1])
            if best_db[1] > 0.5:  # Threshold to determine if relevant
                return best_db[0]
                
        return None
    
    async def process_query(self, query: str) -> dict:
        """Process user query and select best agent/database"""
        reasoning = []
        
        # Get available databases
        available_dbs = list(vector_stores.keys())
        reasoning.append(f"Found {len(available_dbs)} vector databases: {', '.join(available_dbs) if available_dbs else 'none'}")
        
        # Check if query is medical-related
        is_medical = False
        medical_keywords = ["symptom", "disease", "treatment", "diagnosis", "patient", "doctor", "medicine", 
                          "hospital", "prescription", "pain", "health", "medical", "clinical", "surgery"]
        
        for keyword in medical_keywords:
            if keyword in query.lower():
                is_medical = True
                break
                
        if is_medical:
            reasoning.append("Query appears to be medical-related; considering medical agent.")
        
        # Select best database if any
        best_db = self.select_best_db(query, db_metadata)
        
        if best_db:
            reasoning.append(f"Selected vector database '{best_db}' as most relevant for this query.")
            
            # Query the vector database
            vectorstore = vector_stores[best_db]
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
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
                "source_documents": source_docs,
                "db_used": best_db,
                "reasoning": "\n".join(reasoning)
            }
            
        elif is_medical and medical_agent:
            reasoning.append("No relevant vector database found. Using medical agent for specialized analysis.")
            
            # Use medical agent without documents
            analysis = medical_agent.run(query=query)
            
            return {
                "answer": analysis,
                "source_documents": None,
                "db_used": None,
                "reasoning": "\n".join(reasoning)
            }
            
        else:
            reasoning.append("No relevant vector database found. Using general LLM.")
            
            # Fallback to general LLM
            response = self.llm.invoke(query)
            
            return {
                "answer": response.content,
                "source_documents": None,
                "db_used": None,
                "reasoning": "\n".join(reasoning)
            }

# Initialize the medical agent and general chatbot
@app.on_event("startup")
async def startup_event():
    global medical_agent, general_llm, embeddings, manager_agent
    
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
    
    # Initialize manager agent
    manager_agent = ManagerAgent(general_llm, embeddings)

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
    db_name: Optional[str] = Form(None),
    description: Optional[str] = Form(None)
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
            docs = []
            # Process each uploaded document
            for doc in documents:
                file_path = Path(temp_dir) / doc.filename
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(doc.file, buffer)
                
                # Load document based on file type
                if doc.content_type == "application/pdf":
                    try:
                        loader = process_document_with_ocr(file_path)
                        docs = loader
                    except Exception as e:
                        raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
                elif doc.content_type == "text/plain":
                    loader = TextLoader(str(file_path))
                    docs = loader.load()
                else:
                    raise HTTPException(status_code=400, detail=f"Unsupported document type: {doc.content_type}")
                
                
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
            
            # Save metadata about this database
            db_metadata[db_id] = {
                "description": description or f"Vector database created from {len(documents)} documents",
                "document_count": len(documents),
                "chunk_count": len(split_docs),
                "document_names": [doc.filename for doc in documents]
            }
            
            # Save the vector store to disk (optional)
            db_path = f"vectorstores/{db_id}"
            os.makedirs(db_path, exist_ok=True)
            vectorstore.save_local(db_path)
            
            # Save metadata to disk
            import json
            with open(f"{db_path}/metadata.json", "w") as f:
                json.dump(db_metadata[db_id], f)
            
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
                
                # Load metadata if available
                import json
                metadata_path = f"{db_path}/metadata.json"
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        db_metadata[db_id] = json.load(f)
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
            "source_documents": source_docs,
            "db_used": db_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NEW ENDPOINT: Manager agent that intelligently routes queries
@app.post("/api/manager", response_model=ManagerResponse)
async def manager_route(
    query: str = Form(...),
):
    """Manager agent that routes queries to the appropriate system"""
    try:
        if not manager_agent:
            raise HTTPException(status_code=500, detail="Manager agent not initialized")
        
        # Load any vector databases from disk that are not already in memory
        if os.path.exists("vectorstores"):
            disk_dbs = [d for d in os.listdir("vectorstores") if os.path.isdir(os.path.join("vectorstores", d))]
            for db_id in disk_dbs:
                if db_id not in vector_stores:
                    db_path = f"vectorstores/{db_id}"
                    try:
                        vectorstore = FAISS.load_local(db_path, embeddings)
                        vector_stores[db_id] = vectorstore
                        
                        # Load metadata if available
                        import json
                        metadata_path = f"{db_path}/metadata.json"
                        if os.path.exists(metadata_path):
                            with open(metadata_path, "r") as f:
                                db_metadata[db_id] = json.load(f)
                    except Exception:
                        # Skip if can't load this database
                        continue
        
        # Process the query with the manager agent
        result = await manager_agent.process_query(query)
        
        return result
    
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
        db_info = {}
        
        # Add metadata for each database
        for db_id in all_dbs:
            # If database has metadata, use it
            if db_id in db_metadata:
                db_info[db_id] = db_metadata[db_id]
            else:
                # Try to load metadata from disk
                metadata_path = f"vectorstores/{db_id}/metadata.json"
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        db_metadata[db_id] = metadata
                        db_info[db_id] = metadata
                else:
                    db_info[db_id] = {"description": "Vector database"}
        
        return {"databases": all_dbs, "db_info": db_info}
    
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
            
        # Remove metadata
        if db_id in db_metadata:
            del db_metadata[db_id]
        
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
        "manager_agent": bool(manager_agent),
        "vector_dbs": len(vector_stores)
    }
    
    if not health_status["medical_agent"] or not health_status["general_llm"] or not health_status["embeddings"] or not health_status["manager_agent"]:
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
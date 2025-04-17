from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_groq import ChatGroq
from pydantic import BaseModel
from typing import Optional, List
import os
import shutil
import tempfile
from pathlib import Path
from dotenv import load_dotenv

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

# Response models
class ChatResponse(BaseModel):
    response: str

class MedicalResponse(BaseModel):
    analysis: str

# Initialize the medical agent and general chatbot
@app.on_event("startup")
async def startup_event():
    global medical_agent, general_llm
    
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

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check if the API is running and services are initialized"""
    health_status = {
        "status": "healthy",
        "medical_agent": bool(medical_agent),
        "general_llm": bool(general_llm)
    }
    
    if not health_status["medical_agent"] or not health_status["general_llm"]:
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
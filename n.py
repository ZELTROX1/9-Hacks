from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import os
import shutil
from uuid import uuid4
import uvicorn

# Import your MedicalAgent
from medical_agent.med import MedicalAgent

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Medical Assistant")

# Initialize the Medical Agent
medical_agent = MedicalAgent()

# Directory setup
os.makedirs("uploads/documents", exist_ok=True)
os.makedirs("uploads/images", exist_ok=True)

# Model for chat request
class MedicalQueryRequest(BaseModel):
    query: str
    document_paths: Optional[List[str]] = []
    xray_images: Optional[List[str]] = []

# Upload document
@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid4()}{ext}"
        path = f"uploads/documents/{filename}"
        
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"stored_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Upload image
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, PNG allowed.")
        
        filename = f"{uuid4()}{ext}"
        path = f"uploads/images/{filename}"
        
        with open(path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"stored_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat endpoint
@app.post("/medical/chat")
async def medical_chat(request: MedicalQueryRequest):
    try:
        result = medical_agent.run(
            query=request.query,
            document_paths=request.document_paths,
            xray_images=request.xray_images
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

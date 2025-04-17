from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain.agents import Tool, AgentExecutor, create_json_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
import os
import uvicorn
from dotenv import load_dotenv
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
import shutil
import os
from uuid import uuid4
from typing import Optional, List, Dict, Any

# Import the MedicalAgent
from medical_agent.med import MedicalAgent

load_dotenv()

# FastAPI app
app = FastAPI(title="Multi-Domain Assistant Chatbot")

# Initialize the main LLM
main_llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama3-70b-8192",
    temperature=0.3,
    max_tokens=2048
)

# Initialize the medical agent
medical_agent = MedicalAgent()

# Create specialized tool functions
def medical_expert_function(input_data: str) -> str:
    """Process medical queries with optional documents and X-ray images."""
    try:
        # Parse the input data
        parsed_data = json.loads(input_data)
        query = parsed_data.get("query", "")
        document_paths = parsed_data.get("document_paths", [])
        xray_images = parsed_data.get("xray_images", [])
        
        # Call the medical agent's run method
        return medical_agent.run(
            query=query,
            document_paths=document_paths,
            xray_images=xray_images
        )
    except json.JSONDecodeError:
        # If it's not JSON, treat it as a simple query
        return medical_agent.run(query=input_data)
    except Exception as e:
        return f"Error in medical expert: {str(e)}"

def legal_expert_function(query: str) -> str:
    """Process legal queries."""
    # Placeholder for legal agent logic
    legal_prompt = f"""As a legal expert, provide information about: {query}
    
    Remember to:
    1. Explain legal concepts clearly
    2. Cite relevant laws or precedents when possible
    3. Include appropriate disclaimers"""
    
    response = main_llm.invoke(legal_prompt)
    return response.content



def general_assistant_function(query: str) -> str:
    """Handle general queries."""
    response = main_llm.invoke(query)
    return response.content

# Create tools
tools = [
    Tool(
        name="medical_expert",
        func=medical_expert_function,
        description="Expert medical system that analyzes medical queries with optional document paths and X-ray images. Use JSON format for input with documents/images.",
    ),
    Tool(
        name="legal_expert",
        func=legal_expert_function,
        description="Expert legal system that analyzes legal queries and provides legal information.",
    ),
    Tool(
        name="general_assistant",
        func=general_assistant_function,
        description="General assistant for queries that don't require specialized medical or legal expertise.",
    )
]

# Create the agent prompt
# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="""You are a multi-domain expert assistant with access to medical, legal, and general knowledge tools.

Your task is to:
1. Analyze the user's query to determine which domain it belongs to
2. Use the appropriate tool(s) to get expert information
3. Synthesize the information into a coherent, helpful response

For medical queries with documents/images, format your tool input as JSON:
{
    "query": "medical question",
    "document_paths": ["path1", "path2"],
    "xray_images": ["image1.jpg", "image2.jpg"]
}

Always provide accurate, comprehensive responses based on the tool outputs."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    # Add these lines to include the required variables
    ("human", "Available tools: {tool_names}\n\n{tools}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent
agent = create_json_chat_agent(llm=main_llm, tools=tools, prompt=prompt)

# Create memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    max_iterations=3,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    document_paths: Optional[List[str]] = None
    xray_images: Optional[List[str]] = None

class QueryResponse(BaseModel):
    response: str
    intermediate_steps: Optional[List[Dict[str, Any]]] = None

@app.post("/upload/document")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document file."""
    try:
        # Generate a unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid4()}{file_extension}"
        file_path = f"uploads/documents/{unique_filename}"
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"filename": file.filename, "stored_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file."""
    try:
        # Validate file type
        valid_extensions = [".jpg", ".jpeg", ".png"]
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in valid_extensions:
            raise HTTPException(status_code=400, detail="Only JPG, JPEG, and PNG files are allowed")
        
        # Generate a unique filename
        unique_filename = f"{uuid4()}{file_extension}"
        file_path = f"uploads/images/{unique_filename}"
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"filename": file.filename, "stored_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    try:
        # Prepare the input with context data if documents/images are provided
        if request.document_paths or request.xray_images:
            context_data = {
                "query": request.query,
                "document_paths": request.document_paths if request.document_paths else [],
                "xray_images": request.xray_images if request.xray_images else []
            }
            input_str = request.query + f"\n\nContext data for medical analysis: {json.dumps(context_data)}"
        else:
            input_str = request.query
        
        # Run the agent
        result = agent_executor.invoke({"input": input_str})
        
        # Format intermediate steps for easier viewing
        intermediate_steps = None
        if "intermediate_steps" in result:
            intermediate_steps = [
                {
                    "action": str(step[0]),
                    "observation": str(step[1])
                } for step in result["intermediate_steps"]
            ]
        
        return QueryResponse(
            response=result["output"],
            intermediate_steps=intermediate_steps
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory")
async def get_conversation_memory():
    """Return the conversation history."""
    return {"memory": memory.load_memory_variables({})}

@app.delete("/memory")
async def clear_memory():
    """Clear the conversation memory."""
    memory.clear()
    return {"message": "Memory cleared successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
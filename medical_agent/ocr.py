from dotenv import load_dotenv
import os
import json
from mistralai import Mistral
from typing import Optional, Dict, Any, Union

def process_document_with_ocr(file_path: str, output_json_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a document using Mistral's OCR capabilities.
    
    Args:
        file_path: Path to the PDF or image file to process
        output_json_path: Optional path to save the OCR results as JSON
        
    Returns:
        Dictionary containing the OCR results
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable is not set")
    
    # Initialize Mistral client
    client = Mistral(api_key=api_key)
    
    # Get file name from path
    file_name = os.path.basename(file_path)
    
    # Upload the file
    uploaded_file = client.files.upload(
        file={
            "file_name": file_name,
            "content": open(file_path, "rb"),
        },
        purpose="ocr"
    )
    
    # Retrieve file info and get signed URL
    client.files.retrieve(file_id=uploaded_file.id)
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id)
    
    # Process with OCR
    ocr_response = client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": signed_url.url,
        }
    )
    ocr_result = ocr_response.dict()
    
    return ocr_result

# Example usage
if __name__ == "__main__":
    result = process_document_with_ocr(
        file_path="adts.202401460_reviewer.pdf", 
        output_json_path="response.json"
    )
    print(f"Document processed successfully. Results saved to response.json this is the resut:-{result}")
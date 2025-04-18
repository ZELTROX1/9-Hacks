from dotenv import load_dotenv
import os
import json
from mistralai import Mistral
from typing import Optional, Dict, Any, Union
from langchain.schema import Document
from typing import List



def process_document_with_ocr(file_path: str) -> List[Document]:
    """
    Process a document (PDF or image) using Mistral's OCR capabilities and convert to 
    LangChain Document objects with extracted text.
    
    Args:
        file_path: Path to the PDF or image file to process
        
    Returns:
        List of Document objects with extracted text
    """
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")
        
        # Initialize Mistral client
        client = Mistral(api_key=api_key)
        
        # Get file name from path
        file_name = os.path.basename(str(file_path))
        
        # Check if file exists and is readable
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
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
        
        # Convert OCR results to Document objects
        documents = []
        all_text = ""  # Collect all text from the document
        
        # Extract text from each page in the OCR result
        if "pages" in ocr_result and ocr_result["pages"]:
            for idx, page in enumerate(ocr_result["pages"]):
                # Initialize text for this page
                page_text = ""
                
                # Try different ways to extract text depending on OCR result structure
                if "text" in page and page["text"]:
                    page_text = page["text"]
                elif "blocks" in page and page["blocks"]:
                    for block in page["blocks"]:
                        if "text" in block:
                            page_text += block["text"] + " "
                
                # Add to the all_text collector
                if page_text.strip():
                    all_text += page_text.strip() + "\n\n"
                    
                    # Also create individual page documents
                    documents.append(Document(
                        page_content=page_text,
                        metadata={
                            "source": str(file_path),
                            "page": idx,
                            "file_name": file_name
                        }
                    ))
        
        # If no pages were found but there's text at the top level
        if not documents and "text" in ocr_result and ocr_result["text"]:
            all_text = ocr_result["text"]
            documents.append(Document(
                page_content=all_text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_name
                }
            ))
        
        # If we still don't have any documents, create one with a placeholder message
        if not documents:
            documents.append(Document(
                page_content="No text content could be extracted from the document.",
                metadata={
                    "source": str(file_path),
                    "file_name": file_name,
                    "extraction_error": True
                }
            ))
        
        # Add an additional document that contains all the text from all pages
        # This is useful for cases where you want the entire document content
        if all_text and len(documents) > 1:
            documents.append(Document(
                page_content=all_text,
                metadata={
                    "source": str(file_path),
                    "file_name": file_name,
                    "type": "full_document"
                }
            ))
            
        return documents
        
    except Exception as e:
        # Return a document with the error information
        print(f"Error processing document {file_path}: {str(e)}")
        return [Document(
            page_content=f"Error processing document: {str(e)}",
            metadata={
                "source": str(file_path),
                "error": str(e),
                "extraction_error": True
            }
        )]

# Example usage
if __name__ == "__main__":
    result = process_document_with_ocr(
        file_path="adts.202401460_reviewer.pdf", 
        output_json_path="response.json"
    )
    print(f"Document processed successfully. Results saved to response.json this is the resut:-{result}")
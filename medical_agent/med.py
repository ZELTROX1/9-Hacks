from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import BaseTool
from langchain.tools import Tool
from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import Optional, List
from dotenv import load_dotenv

from medical_agent.x_ray_scan import predict_xray # Import the X-ray prediction function
from medical_agent.ocr import process_document_with_ocr  # Your Mistral OCR system

load_dotenv()

class MedicalAgent:
    """
    A specialized medical agent using Groq LLM and PubMedBERT for enhanced understanding of medical terminology.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "llama3-70b-8192"):
        """Initialize the MedicalAgent with Groq LLM and PubMedBERT."""
        # Setup Groq LLM
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key is None:
                raise ValueError("Groq API key must be provided")
        
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model_name,
            temperature=0.1,
            max_tokens=2048
        )
        
        # Setup PubMedBERT
        print("Loading PubMedBERT model...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        self.bert_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        print("PubMedBERT model loaded successfully.")
        
        # Create prompt template
        self.medical_prompt = PromptTemplate(
            input_variables=["query", "medical_context", "bert_insights", "xray_analysis"],
            template="""You are a specialized medical informative agent.

MEDICAL QUERY: {query}

DOCUMENT CONTEXT: {medical_context}

BIOMEDICAL MODEL INSIGHTS: {bert_insights}

X-RAY ANALYSIS: {xray_analysis}

Provide an evidence-based, clear, and professional medical response. Include appropriate medical disclaimers if necessary.
"""
        )
        
        # Create LangChain chain
        self.chain = LLMChain(llm=self.llm, prompt=self.medical_prompt)
    
    def extract_biomedical_insights(self, text: str) -> str:
        """Use PubMedBERT to extract biomedical insights from text."""
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        
        embeddings = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # Extract medical terms (simple)
        medical_terms = self._extract_medical_terms(text)
        
        if medical_terms:
            return f"PubMedBERT analysis identified key medical concepts: {', '.join(medical_terms)}"
        else:
            return "PubMedBERT analysis did not identify specific medical terminology in the text."
    
    def _extract_medical_terms(self, text: str) -> List[str]:
        """Extract simple medical terms from text."""
        medical_terminology = {
            "hypertension", "blood pressure", "systolic", "diastolic", 
            "antihypertensive", "diuretic", "ace inhibitor", "arb",
            "sodium", "potassium", "cardiovascular", "comorbidity",
            "diabetes", "cholesterol", "exercise", "dietary", "dash diet"
        }
        
        found_terms = []
        for term in medical_terminology:
            if term.lower() in text.lower():
                found_terms.append(term)
        
        return found_terms
    
    def process_documents(self, document_paths: List[str]) -> tuple:
        """OCR process documents and extract biomedical insights."""
        if not document_paths:
            return "No medical context available.", "No biomedical analysis available."
        
        extracted_texts = []
        for path in document_paths:
            try:
                text = process_document_with_ocr(path)
                extracted_texts.append(text)
            except Exception as e:
                print(f"Warning: Failed to OCR document {path}: {e}")
        
        combined_text = "\n\n".join(extracted_texts)
        bert_insights = self.extract_biomedical_insights(combined_text)
        
        return combined_text, bert_insights
    
    def analyze_xray(self, xray_images: Optional[List[str]] = None) -> str:
        """Analyze X-ray images using the `predict_xray` function."""
        if not xray_images:
            return "No X-ray images provided for analysis."
        
        xray_analysis = ""
        for image in xray_images:
            try:
                with open(image, "rb") as f:
                    image_bytes = f.read()
                analysis = predict_xray(image_bytes)
                xray_analysis += f"Analysis of {image}: {analysis}\n"
            except Exception as e:
                print(f"Warning: Failed to analyze X-ray image {image}: {e}")
                xray_analysis += f"Failed to analyze {image}: {str(e)}\n"
        
        return xray_analysis
    
    def run(self, query: str, document_paths: Optional[List[str]] = None, xray_images: Optional[List[str]] = None) -> str:
        """Run a medical query through PubMedBERT and Groq LLM, including X-ray analysis."""
        query_insights = self.extract_biomedical_insights(query)
        
        if document_paths:
            doc_context, doc_insights = self.process_documents(document_paths)
            combined_insights = f"QUERY ANALYSIS: {query_insights}\nDOCUMENT ANALYSIS: {doc_insights}"
        else:
            doc_context = "No additional medical context available."
            combined_insights = f"QUERY ANALYSIS: {query_insights}"
        
        # Analyze X-ray images if provided
        xray_analysis = self.analyze_xray(xray_images)
        
        response = self.chain.invoke({
            "query": query,
            "medical_context": doc_context,
            "bert_insights": combined_insights,
            "xray_analysis": xray_analysis
        })
        
        return response["text"]
    
    def as_tool(self) -> BaseTool:
        """Create a LangChain Tool wrapping this MedicalAgent."""
        return Tool(
            name="MedicalExpert",
            func=self.run,
            description="Medical expert system enhanced with biomedical language understanding (PubMedBERT + Groq LLM) and X-ray analysis.",
            return_direct=True
        )


if __name__ == "__main__":
    med = MedicalAgent()
    # Example usage
    
    print(med.run(
        "Can you diagnose the potential chest condition from this X-ray?", 
        document_paths=['med.txt'],
        xray_images=["images.jpeg"]
    ))
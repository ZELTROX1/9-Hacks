# RAG System using FAISS with Text Files and Groq LLM
# ---------------------------------------------------

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TextFileProcessor:
    """Process text files for RAG system"""
    
    def __init__(self, text_directory: Optional[str] = None):
        """Initialize with directory containing text files or individual file"""
        self.text_directory = text_directory
        self.documents = []
        self.document_sources = []
        self.chunk_size = 500  # Default chunk size in characters
        self.chunk_overlap = 100  # Default overlap between chunks
    
    def load_single_file(self, file_path: str) -> List[str]:
        """Load and chunk a single text file"""
        chunks = []
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Create chunks with overlap
            for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
                chunk = content[i:i + self.chunk_size]
                if len(chunk) >= 50:  # Only include chunks with reasonable content
                    chunks.append(chunk)
                    self.document_sources.append(f"{file_path}:{i}-{i + len(chunk)}")
        
        return chunks
    
    def load_directory(self, directory_path: Optional[str] = None) -> List[str]:
        """Load all text files from a directory"""
        if directory_path:
            self.text_directory = directory_path
            
        if not self.text_directory:
            raise ValueError("No text directory specified")
            
        all_chunks = []
        text_files = glob.glob(os.path.join(self.text_directory, "*.txt"))
        
        print(f"Found {len(text_files)} text files in {self.text_directory}")
        
        for file_path in text_files:
            file_chunks = self.load_single_file(file_path)
            all_chunks.extend(file_chunks)
            
        return all_chunks
    
    def set_chunk_params(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """Set chunking parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def get_documents(self) -> List[str]:
        """Get all processed document chunks"""
        if not self.documents:
            if self.text_directory:
                self.documents = self.load_directory()
        return self.documents
    
    def add_file(self, file_path: str):
        """Add a single file to the document collection"""
        new_chunks = self.load_single_file(file_path)
        self.documents.extend(new_chunks)
        print(f"Added {len(new_chunks)} chunks from {file_path}")
        return new_chunks
    
    def analyze_text_content(self) -> Dict[str, Any]:
        """Analyze text content for common terms and statistics"""
        if not self.documents:
            self.get_documents()
            
        if not self.documents:
            return {"error": "No documents loaded"}
            
        # Basic statistics
        total_words = sum(len(doc.split()) for doc in self.documents)
        avg_words_per_chunk = total_words / len(self.documents)
        
        # Count term frequency (simple approach)
        word_count = {}
        for doc in self.documents:
            for word in doc.lower().split():
                # Clean the word (remove punctuation)
                word = ''.join(c for c in word if c.isalnum())
                if word and len(word) > 3:  # Skip short words
                    word_count[word] = word_count.get(word, 0) + 1
        
        # Get top terms
        top_terms = sorted(word_count.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            "document_count": len(self.documents),
            "total_words": total_words,
            "average_words_per_chunk": avg_words_per_chunk,
            "top_terms": top_terms
        }
    
    def visualize_top_terms(self, top_n: int = 10):
        """Visualize the top terms in the corpus"""
        stats = self.analyze_text_content()
        top_terms = stats["top_terms"][:top_n]
        
        terms = [term for term, _ in top_terms]
        counts = [count for _, count in top_terms]
        
        plt.figure(figsize=(12, 6))
        plt.bar(terms, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Terms in Corpus')
        plt.xlabel('Terms')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Save the plot
        output_file = "top_terms.png"
        plt.savefig(output_file)
        plt.close()
        return f"Chart saved as {output_file}"
    
    def visualize_document_lengths(self):
        """Visualize the distribution of document lengths"""
        if not self.documents:
            self.get_documents()
            
        doc_lengths = [len(doc.split()) for doc in self.documents]
        
        plt.figure(figsize=(12, 6))
        plt.hist(doc_lengths, bins=20)
        plt.title('Distribution of Document Chunk Lengths')
        plt.xlabel('Number of Words')
        plt.ylabel('Frequency')
        plt.tight_layout()
        
        # Save the plot
        output_file = "doc_lengths.png"
        plt.savefig(output_file)
        plt.close()
        return f"Chart saved as {output_file}"


class FAISSVectorStore:
    """Vector store using FAISS"""
    
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize with embedding model"""
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        
    def add_documents(self, documents: List[str], source_info: Optional[List[str]] = None):
        """Add documents to the vector store"""
        self.documents = documents
        self.source_info = source_info if source_info else ['unknown'] * len(documents)
        
        # Create embeddings
        embeddings = self.embedding_model.encode(documents)
        
        # Convert to float32 for FAISS
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        print(f"Added {len(documents)} documents to FAISS index")
        
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):  # Ensure index is valid
                results.append({
                    'document': self.documents[idx],
                    'distance': float(distances[0][i]),
                    'index': int(idx),
                    'source': self.source_info[idx] if hasattr(self, 'source_info') else 'unknown'
                })
                
        return results
    
    def save_index(self, index_path: str, documents_path: str):
        """Save FAISS index and documents to disk"""
        if self.index is None:
            raise ValueError("No index to save")
            
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents and sources
        with open(documents_path, 'w', encoding='utf-8') as f:
            for i, doc in enumerate(self.documents):
                source = self.source_info[i] if i < len(self.source_info) else 'unknown'
                f.write(f"{source}|||{doc}\n")
                
        print(f"Index saved to {index_path} and documents to {documents_path}")
    
    def load_index(self, index_path: str, documents_path: str):
        """Load FAISS index and documents from disk"""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents and sources
        self.documents = []
        self.source_info = []
        
        with open(documents_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|||', 1)
                if len(parts) == 2:
                    source, doc = parts
                    self.source_info.append(source)
                    self.documents.append(doc)
                else:
                    self.documents.append(line.strip())
                    self.source_info.append('unknown')
                    
        print(f"Loaded index with {len(self.documents)} documents")


class GroqLLM:
    """Interface to Groq LLM API"""
    
    def __init__(self):
        """Initialize Groq client"""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"  # Default model
        
    def set_model(self, model_name: str):
        """Set the model to use"""
        self.model = model_name
        
    def generate_response(self, prompt: str, system_prompt: str = None, temperature: float = 0.7) -> str:
        """Generate response from Groq LLM"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            messages.append({"role": "system", "content": "You are a helpful assistant analyzing text data and providing insights."})
            
        messages.append({"role": "user", "content": prompt})
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature
        )
        
        return completion.choices[0].message.content
    
    def available_models(self) -> List[str]:
        """Return list of available models from Groq"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error fetching models: {e}")
            return ["llama3-70b-8192", "mixtral-8x7b", "gemma-7b"]


class RAGSystem:
    """Main RAG system combining all components"""
    
    def __init__(self, text_directory: Optional[str] = None):
        """Initialize the RAG system"""
        self.text_processor = TextFileProcessor(text_directory)
        self.vector_store = FAISSVectorStore()
        self.llm = GroqLLM()
        
    def build_index(self, chunk_size: int = 500, chunk_overlap: int = 100, save_path: Optional[str] = None):
        """Build the vector index from text files"""
        # Set chunking parameters
        self.text_processor.set_chunk_params(chunk_size, chunk_overlap)
        
        # Get documents
        documents = self.text_processor.get_documents()
        source_info = self.text_processor.document_sources
        
        # Add to vector store
        self.vector_store.add_documents(documents, source_info)
        
        # Save if requested
        if save_path:
            index_path = os.path.join(save_path, "faiss_index")
            documents_path = os.path.join(save_path, "documents.txt")
            self.vector_store.save_index(index_path, documents_path)
        
        return len(documents)
    
    def load_index(self, index_path: str, documents_path: str):
        """Load an existing index"""
        self.vector_store.load_index(index_path, documents_path)
        return len(self.vector_store.documents)
    
    def add_file(self, file_path: str, rebuild_index: bool = True):
        """Add a new file to the system"""
        # Add file to text processor
        new_chunks = self.text_processor.add_file(file_path)
        
        if rebuild_index:
            # Get all documents and rebuild index
            documents = self.text_processor.get_documents()
            source_info = self.text_processor.document_sources
            self.vector_store.add_documents(documents, source_info)
        
        return len(new_chunks)
    
    def query(self, question: str, k: int = 5, temperature: float = 0.7):
        """Process a user query through the RAG system"""
        # Step 1: Retrieve relevant documents
        results = self.vector_store.search(question, k)
        
        # Step 2: Format context for the LLM
        context = "Here are some relevant text passages:\n\n"
        
        for i, result in enumerate(results):
            context += f"Passage {i+1} [Source: {result['source']}]:\n{result['document']}\n\n"
            
        # Step 3: Create prompt for the LLM
        prompt = f"""
        I need you to answer a question based on the context provided from various text documents.
        
        {context}
        
        Based on the above context only, please answer the following question:
        {question}
        
        If the context doesn't contain sufficient information to answer the question, 
        please indicate that and provide what you can based on the available information.
        If you need to reason through calculations or analysis, please show your work step by step.
        """
        
        system_prompt = """You are a helpful text analysis assistant. 
        Your task is to provide accurate answers based solely on the provided context. 
        If the context doesn't contain enough information to answer a question confidently, 
        acknowledge the limitations rather than making up information."""
        
        # Step 4: Generate response
        response = self.llm.generate_response(prompt, system_prompt, temperature)
        
        return {
            "question": question,
            "retrieved_documents": results,
            "answer": response
        }
    
    def analyze_corpus(self):
        """Analyze the text corpus"""
        return self.text_processor.analyze_text_content()
    
    def visualize_terms(self, top_n: int = 10):
        """Visualize top terms in the corpus"""
        return self.text_processor.visualize_top_terms(top_n)
    
    def visualize_document_lengths(self):
        """Visualize document length distribution"""
        return self.text_processor.visualize_document_lengths()
    
    def generate_text_summary(self):
        """Generate a summary of the text corpus using the LLM"""
        # Sample a few documents for the LLM to analyze
        documents = self.text_processor.get_documents()
        sample_size = min(5, len(documents))
        sample_docs = documents[:sample_size]
        
        prompt = f"""
        I need you to generate a concise summary of a text corpus.
        Here are {sample_size} sample passages from the corpus:
        
        {''.join([f'Sample {i+1}:{doc}' for i, doc in enumerate(sample_docs)])}
        
        Based on these samples, please:
        1. Generate a concise summary of what this corpus appears to be about
        2. Identify the main themes or topics
        3. Note any interesting patterns you observe
        
        Keep your response under 300 words.
        """
        
        system_prompt = "You are a text analysis expert who excels at summarizing and identifying patterns in text."
        
        return self.llm.generate_response(prompt, system_prompt)


# Example usage
if __name__ == "__main__":
    # Setup RAG system with text directory
    rag = RAGSystem()
    rag.add_file("requirements.txt")
    
    # Build index with custom chunking parameters
    num_chunks = rag.build_index(chunk_size=300, chunk_overlap=50)
    print(f"Indexed {num_chunks} text chunks")
    
    # Save the index
    os.makedirs("./index", exist_ok=True)
    rag.vector_store.save_index("./index/faiss_index", "./index/documents.txt")
    
    # Query the system
    result = rag.query("What are the main themes in these documents?")
    print(result["answer"])
    
    # Visualize data
    rag.visualize_terms(top_n=15)
    rag.visualize_document_lengths()
    
    # Get a summary of the corpus
    summary = rag.generate_text_summary()
    print("\nCorpus Summary:")
    print(summary)
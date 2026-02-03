"""
FastAPI backend for Second Brain Lite.

Exposes HTTP endpoints for the RAG pipeline, enabling web UI integration
while maintaining flexibility for different model providers.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from loader import DocumentLoader
from chunker import DocumentChunker
from embeddings import OpenAIEmbeddings, EmbeddingManager
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Second Brain Lite API",
    description="Local RAG system for querying personal notes",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in-memory for simplicity)
vector_store_instance: Optional[VectorStore] = None
embedding_manager_instance: Optional[EmbeddingManager] = None
rag_pipeline_instance: Optional[RAGPipeline] = None


# Request/Response models
class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    file: str
    chunk_index: int
    heading: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class IndexResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


class StatusResponse(BaseModel):
    indexed: bool
    total_chunks: int


def load_config():
    """Load configuration from environment variables."""
    return {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'notes_dir': os.getenv('NOTES_DIR', 'notes'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        'top_k': int(os.getenv('TOP_K', '5')),
        'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Second Brain Lite API",
        "version": "1.0.0",
        "endpoints": {
            "POST /index": "Index notes from the notes directory",
            "POST /query": "Query the RAG system",
            "GET /status": "Check indexing status"
        }
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Check if the vector store is indexed.
    
    Returns:
        StatusResponse with indexed status and total chunks
    """
    global vector_store_instance
    
    # Check if index exists on disk
    index_path = Path(".faiss_index/faiss.index")
    
    if index_path.exists() and vector_store_instance is None:
        # Load existing index
        try:
            vector_store_instance = VectorStore()
            vector_store_instance.load(".faiss_index")
        except Exception as e:
            return StatusResponse(indexed=False, total_chunks=0)
    
    if vector_store_instance:
        return StatusResponse(
            indexed=True,
            total_chunks=vector_store_instance.size
        )
    
    return StatusResponse(indexed=False, total_chunks=0)


@app.post("/index", response_model=IndexResponse)
async def index_notes():
    """
    Index all notes from the notes directory.
    
    This endpoint:
    1. Loads documents from the notes directory
    2. Chunks them intelligently
    3. Generates embeddings
    4. Stores in FAISS vector store
    
    Returns:
        IndexResponse with status and chunk count
    """
    global vector_store_instance, embedding_manager_instance, rag_pipeline_instance
    
    config = load_config()
    
    # Validate API key if using OpenAI
    if not config['openai_api_key']:
        raise HTTPException(
            status_code=400,
            detail="OPENAI_API_KEY not configured. Set it in .env file."
        )
    
    # Check if notes directory exists
    notes_path = Path(config['notes_dir'])
    if not notes_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Notes directory not found: {config['notes_dir']}"
        )
    
    try:
        # Step 1: Load documents
        loader = DocumentLoader(config['notes_dir'])
        documents = loader.load_documents()
        
        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No documents found in notes directory"
            )
        
        # Step 2: Chunk documents
        chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        # Step 3: Generate embeddings
        embedding_backend = OpenAIEmbeddings(
            api_key=config['openai_api_key'],
            model=config['embedding_model']
        )
        embedding_manager_instance = EmbeddingManager(embedding_backend)
        embeddings = embedding_manager_instance.embed_chunks(all_chunks)
        
        # Step 4: Build vector store
        vector_store_instance = VectorStore(dimension=len(embeddings[0]))
        vector_store_instance.add_documents(all_chunks, embeddings)
        
        # Step 5: Save to disk
        vector_store_instance.save(".faiss_index")
        
        # Step 6: Initialize RAG pipeline
        rag_pipeline_instance = RAGPipeline(
            vector_store=vector_store_instance,
            embedding_manager=embedding_manager_instance,
            llm_api_key=config['openai_api_key'],
            llm_model=config['llm_model'],
            top_k=config['top_k'],
            similarity_threshold=config['similarity_threshold']
        )
        
        return IndexResponse(
            status="success",
            chunks_indexed=vector_store_instance.size,
            message=f"Successfully indexed {len(documents)} documents into {vector_store_instance.size} chunks"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_notes(request: QueryRequest):
    """
    Query the RAG system.
    
    Args:
        request: QueryRequest with question
        
    Returns:
        QueryResponse with answer and sources
    """
    global vector_store_instance, embedding_manager_instance, rag_pipeline_instance
    
    config = load_config()
    
    # Check if index exists
    if vector_store_instance is None:
        # Try to load from disk
        index_path = Path(".faiss_index/faiss.index")
        if not index_path.exists():
            raise HTTPException(
                status_code=400,
                detail="No index found. Please run /index first."
            )
        
        try:
            # Load vector store
            vector_store_instance = VectorStore()
            vector_store_instance.load(".faiss_index")
            
            # Initialize embedding manager
            embedding_backend = OpenAIEmbeddings(
                api_key=config['openai_api_key'],
                model=config['embedding_model']
            )
            embedding_manager_instance = EmbeddingManager(embedding_backend)
            
            # Initialize RAG pipeline
            rag_pipeline_instance = RAGPipeline(
                vector_store=vector_store_instance,
                embedding_manager=embedding_manager_instance,
                llm_api_key=config['openai_api_key'],
                llm_model=config['llm_model'],
                top_k=config['top_k'],
                similarity_threshold=config['similarity_threshold']
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load index: {str(e)}"
            )
    
    try:
        # Run query through RAG pipeline
        result = rag_pipeline_instance.query(request.question)
        
        # Convert sources to response format
        sources = [
            Source(
                file=source['file'],
                chunk_index=source['chunk_index'],
                heading=source.get('heading')
            )
            for source in result['sources']
        ]
        
        return QueryResponse(
            answer=result['answer'],
            sources=sources
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Second Brain Lite API server...")
    print("API docs available at: http://localhost:8000/docs")
    print("Frontend should connect to: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

"""
Main CLI interface for Second Brain Lite.

Provides commands to index notes and query the RAG system.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from loader import DocumentLoader
from chunker import DocumentChunker
from embeddings import OpenAIEmbeddings, EmbeddingManager
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


# Load environment variables
load_dotenv()


def load_config():
    """Load configuration from environment variables."""
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'notes_dir': os.getenv('NOTES_DIR', 'notes'),
        'embedding_model': os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small'),
        'llm_model': os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        'top_k': int(os.getenv('TOP_K', '5')),
        'similarity_threshold': float(os.getenv('SIMILARITY_THRESHOLD', '0.3'))
    }
    
    # Validate required config
    if not config['openai_api_key']:
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file based on .env.example")
        sys.exit(1)
    
    return config


def index_notes(config):
    """
    Index all notes from the notes directory.
    
    Args:
        config: Configuration dictionary
    """
    print("🔍 Starting indexing process...")
    
    # Check if notes directory exists
    notes_path = Path(config['notes_dir'])
    if not notes_path.exists():
        print(f"Error: Notes directory not found: {config['notes_dir']}")
        print("Please create the directory and add some .md or .txt files")
        sys.exit(1)
    
    # Step 1: Load documents
    print(f"\n📂 Loading documents from {config['notes_dir']}...")
    loader = DocumentLoader(config['notes_dir'])
    documents = loader.load_documents()
    print(f"   Loaded {len(documents)} documents")
    
    if not documents:
        print("   No documents found. Add some .md or .txt files to the notes directory.")
        sys.exit(1)
    
    # Step 2: Chunk documents
    print("\n✂️  Chunking documents...")
    chunker = DocumentChunker(chunk_size=800, chunk_overlap=100)
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc.content, doc.metadata)
        all_chunks.extend(chunks)
    print(f"   Created {len(all_chunks)} chunks")
    
    # Step 3: Generate embeddings
    print("\n🧮 Generating embeddings...")
    embedding_backend = OpenAIEmbeddings(
        api_key=config['openai_api_key'],
        model=config['embedding_model']
    )
    embedding_manager = EmbeddingManager(embedding_backend)
    embeddings = embedding_manager.embed_chunks(all_chunks)
    print(f"   Generated {len(embeddings)} embeddings")
    
    # Step 4: Build vector store
    print("\n💾 Building vector store...")
    vector_store = VectorStore(dimension=len(embeddings[0]))
    vector_store.add_documents(all_chunks, embeddings)
    
    # Step 5: Save to disk
    vector_store.save(".faiss_index")
    print("\n✅ Indexing complete!")
    print(f"   Total chunks indexed: {vector_store.size}")
    print(f"   Vector store saved to .faiss_index/")


def query_notes(question: str, config):
    """
    Query the indexed notes.
    
    Args:
        question: User's question
        config: Configuration dictionary
    """
    # Check if index exists
    index_path = Path(".faiss_index/faiss.index")
    if not index_path.exists():
        print("Error: No index found. Please run 'python src/main.py index' first.")
        sys.exit(1)
    
    # Load vector store
    print("📚 Loading vector store...")
    vector_store = VectorStore()
    vector_store.load(".faiss_index")
    
    # Initialize embedding manager
    embedding_backend = OpenAIEmbeddings(
        api_key=config['openai_api_key'],
        model=config['embedding_model']
    )
    embedding_manager = EmbeddingManager(embedding_backend)
    
    # Initialize RAG pipeline
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        llm_api_key=config['openai_api_key'],
        llm_model=config['llm_model'],
        top_k=config['top_k'],
        similarity_threshold=config['similarity_threshold']
    )
    
    # Run query
    print(f"\n🔎 Searching for: {question}\n")
    result = pipeline.query(question)
    
    # Display results
    print("=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result['answer'])
    print()
    
    if result['sources']:
        print("=" * 80)
        print("SOURCES")
        print("=" * 80)
        for i, source in enumerate(result['sources'], 1):
            heading_info = f" - {source['heading']}" if source['heading'] else ""
            print(f"{i}. {source['file']}{heading_info} (chunk {source['chunk_index']})")
        print()


def print_usage():
    """Print usage information."""
    print("""
Second Brain Lite - Local RAG System for Your Notes

Usage:
    python src/main.py index              Index all notes from the notes directory
    python src/main.py query "<question>" Query your indexed notes

Examples:
    python src/main.py index
    python src/main.py query "What are my thoughts on machine learning?"
    python src/main.py query "Tell me about productivity tips"

Before running:
    1. Create a .env file based on .env.example
    2. Add your OPENAI_API_KEY to the .env file
    3. Add .md or .txt files to the notes/ directory
    """)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    config = load_config()
    
    if command == "index":
        index_notes(config)
    
    elif command == "query":
        if len(sys.argv) < 3:
            print("Error: Please provide a question")
            print('Usage: python src/main.py query "your question here"')
            sys.exit(1)
        
        question = " ".join(sys.argv[2:])
        query_notes(question, config)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

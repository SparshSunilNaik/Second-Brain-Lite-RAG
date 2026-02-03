"""
Tests for Second Brain Lite RAG system.

Validates document loading, chunking, retrieval, and end-to-end query pipeline.
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
from loader import DocumentLoader, Document
from chunker import DocumentChunker, Chunk
from embeddings import OpenAIEmbeddings, EmbeddingManager
from vector_store import VectorStore
from rag_pipeline import RAGPipeline


class TestDocumentLoader:
    """Test document loading functionality."""
    
    def test_load_documents(self):
        """Test loading documents from notes directory."""
        # Assuming notes directory exists with sample files
        notes_dir = Path(__file__).parent.parent / 'notes'
        
        if not notes_dir.exists():
            pytest.skip("Notes directory not found")
        
        loader = DocumentLoader(str(notes_dir))
        documents = loader.load_documents()
        
        assert len(documents) > 0, "Should load at least one document"
        assert all(isinstance(doc, Document) for doc in documents)
        assert all(doc.content for doc in documents), "All documents should have content"
        assert all('source' in doc.metadata for doc in documents)
    
    def test_supported_extensions(self):
        """Test that only .md and .txt files are loaded."""
        notes_dir = Path(__file__).parent.parent / 'notes'
        
        if not notes_dir.exists():
            pytest.skip("Notes directory not found")
        
        loader = DocumentLoader(str(notes_dir))
        documents = loader.load_documents()
        
        for doc in documents:
            file_type = doc.metadata.get('file_type')
            assert file_type in ['.md', '.txt'], f"Unexpected file type: {file_type}"


class TestDocumentChunker:
    """Test document chunking functionality."""
    
    def test_chunk_markdown(self):
        """Test markdown chunking preserves headings."""
        content = """# Main Title

This is the introduction.

## Section 1

Content for section 1.

## Section 2

Content for section 2."""
        
        metadata = {'source': 'test.md', 'file_type': '.md'}
        chunker = DocumentChunker(chunk_size=100)
        chunks = chunker.chunk_document(content, metadata)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all(isinstance(chunk, Chunk) for chunk in chunks)
        assert all(chunk.text for chunk in chunks), "All chunks should have text"
        
        # Check that headings are preserved in metadata
        headings = [c.metadata.get('heading') for c in chunks if c.metadata.get('heading')]
        assert len(headings) > 0, "Should preserve at least one heading"
    
    def test_chunk_text(self):
        """Test plain text chunking."""
        content = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with even more content."""
        
        metadata = {'source': 'test.txt', 'file_type': '.txt'}
        chunker = DocumentChunker(chunk_size=50)
        chunks = chunker.chunk_document(content, metadata)
        
        assert len(chunks) > 0, "Should create at least one chunk"
        assert all('chunk_index' in chunk.metadata for chunk in chunks)
    
    def test_chunk_metadata(self):
        """Test that chunk metadata includes required fields."""
        content = "Test content for chunking."
        metadata = {'source': 'test.md', 'file_type': '.md'}
        
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(content, metadata)
        
        for chunk in chunks:
            assert 'source' in chunk.metadata
            assert 'chunk_index' in chunk.metadata
            assert chunk.metadata['source'] == 'test.md'


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_add_and_search(self):
        """Test adding documents and searching."""
        # Create mock chunks
        chunks = [
            Chunk(text="Machine learning is awesome", metadata={'source': 'test1.md', 'chunk_index': 0}),
            Chunk(text="Deep learning uses neural networks", metadata={'source': 'test2.md', 'chunk_index': 0}),
            Chunk(text="Python is a programming language", metadata={'source': 'test3.md', 'chunk_index': 0})
        ]
        
        # Create mock embeddings (random vectors for testing)
        import numpy as np
        embeddings = [np.random.rand(1536).tolist() for _ in chunks]
        
        # Create and populate vector store
        store = VectorStore(dimension=1536)
        store.add_documents(chunks, embeddings)
        
        assert store.size == 3, "Should have 3 documents"
        
        # Search with a query embedding
        query_embedding = np.random.rand(1536).tolist()
        results = store.search(query_embedding, top_k=2)
        
        assert len(results) == 2, "Should return top 2 results"
        assert all(len(r) == 3 for r in results), "Each result should have (text, metadata, score)"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OPENAI_API_KEY not set"
    )
    def test_full_pipeline(self):
        """Test the complete RAG pipeline from indexing to querying."""
        # Setup
        notes_dir = Path(__file__).parent.parent / 'notes'
        if not notes_dir.exists():
            pytest.skip("Notes directory not found")
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Load and chunk documents
        loader = DocumentLoader(str(notes_dir))
        documents = loader.load_documents()
        assert len(documents) > 0, "Should load documents"
        
        chunker = DocumentChunker()
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        assert len(all_chunks) > 0, "Should create chunks"
        
        # Generate embeddings
        embedding_backend = OpenAIEmbeddings(api_key=api_key)
        embedding_manager = EmbeddingManager(embedding_backend)
        embeddings = embedding_manager.embed_chunks(all_chunks)
        
        assert len(embeddings) == len(all_chunks), "Should have embedding for each chunk"
        assert all(len(emb) > 0 for emb in embeddings), "Embeddings should not be empty"
        
        # Build vector store
        vector_store = VectorStore(dimension=len(embeddings[0]))
        vector_store.add_documents(all_chunks, embeddings)
        
        # Create RAG pipeline
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            llm_api_key=api_key,
            top_k=3
        )
        
        # Test query
        result = pipeline.query("What are my thoughts on machine learning?")
        
        # Assertions
        assert 'answer' in result, "Result should contain answer"
        assert 'sources' in result, "Result should contain sources"
        assert 'retrieved_chunks' in result, "Result should contain retrieved chunks"
        
        assert len(result['retrieved_chunks']) > 0, "Should retrieve at least one chunk"
        assert len(result['sources']) > 0, "Should return at least one source"
        
        # Verify source structure
        for source in result['sources']:
            assert 'file' in source, "Source should have file"
            assert 'chunk_index' in source, "Source should have chunk_index"
        
        print(f"\nQuery: What are my thoughts on machine learning?")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {[s['file'] for s in result['sources']]}")
    
    @pytest.mark.skipif(
        not os.getenv('OPENAI_API_KEY'),
        reason="OPENAI_API_KEY not set"
    )
    def test_irrelevant_query(self):
        """Test that irrelevant queries return appropriate fallback."""
        notes_dir = Path(__file__).parent.parent / 'notes'
        if not notes_dir.exists():
            pytest.skip("Notes directory not found")
        
        api_key = os.getenv('OPENAI_API_KEY')
        
        # Quick setup (reuse from previous test in practice)
        loader = DocumentLoader(str(notes_dir))
        documents = loader.load_documents()
        
        chunker = DocumentChunker()
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc.content, doc.metadata)
            all_chunks.extend(chunks)
        
        embedding_backend = OpenAIEmbeddings(api_key=api_key)
        embedding_manager = EmbeddingManager(embedding_backend)
        embeddings = embedding_manager.embed_chunks(all_chunks)
        
        vector_store = VectorStore(dimension=len(embeddings[0]))
        vector_store.add_documents(all_chunks, embeddings)
        
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedding_manager=embedding_manager,
            llm_api_key=api_key,
            similarity_threshold=0.5  # Stricter threshold
        )
        
        # Query about something not in notes
        result = pipeline.query("What is the capital of France?")
        
        # Should indicate lack of information
        answer_lower = result['answer'].lower()
        assert any(phrase in answer_lower for phrase in [
            "don't have", "not found", "no information", "couldn't find"
        ]), "Should indicate lack of relevant information"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

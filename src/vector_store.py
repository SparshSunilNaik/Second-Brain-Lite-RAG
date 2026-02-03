"""
Vector store for Second Brain Lite.

FAISS-based vector storage with metadata management for efficient
similarity search and retrieval.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "FAISS is not installed. Install with: pip install faiss-cpu"
    )


class VectorStore:
    """FAISS-based vector store with metadata."""
    
    def __init__(self, dimension: int = 1536):
        """
        Initialize vector store.
        
        Args:
            dimension: Dimension of embedding vectors (1536 for text-embedding-3-small)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity
        self.metadata: List[Dict[str, Any]] = []
        self.chunks: List[str] = []  # Store chunk texts for retrieval
    
    def add_documents(
        self, 
        chunks: List[Any], 
        embeddings: List[List[float]]
    ):
        """
        Add documents to the vector store.
        
        Args:
            chunks: List of Chunk objects with .text and .metadata
            embeddings: List of embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embeddings_array)
        
        # Store metadata and chunk texts
        for chunk in chunks:
            self.metadata.append(chunk.metadata)
            self.chunks.append(chunk.text)
    
    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of tuples: (chunk_text, metadata, similarity_score)
            Lower scores = more similar (L2 distance)
        """
        if self.index.ntotal == 0:
            return []
        
        # Ensure we don't request more results than we have
        k = min(top_k, self.index.ntotal)
        
        # Convert query to numpy array
        query_array = np.array([query_embedding], dtype=np.float32)
        
        # Search
        distances, indices = self.index.search(query_array, k)
        
        # Prepare results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Safety check
                results.append((
                    self.chunks[idx],
                    self.metadata[idx],
                    float(distances[0][i])
                ))
        
        return results
    
    def save(self, directory: str = "."):
        """
        Save vector store to disk.
        
        Args:
            directory: Directory to save files
        """
        dir_path = Path(directory)
        dir_path.mkdir(exist_ok=True)
        
        # Save FAISS index
        index_path = dir_path / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata and chunks
        metadata_path = dir_path / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'chunks': self.chunks,
                'dimension': self.dimension
            }, f)
        
        print(f"Vector store saved to {directory}")
    
    def load(self, directory: str = "."):
        """
        Load vector store from disk.
        
        Args:
            directory: Directory containing saved files
        """
        dir_path = Path(directory)
        
        # Load FAISS index
        index_path = dir_path / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata and chunks
        metadata_path = dir_path / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.chunks = data['chunks']
            self.dimension = data['dimension']
        
        print(f"Vector store loaded from {directory}")
        print(f"Total documents: {self.index.ntotal}")
    
    def clear(self):
        """Clear all data from the vector store."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        self.chunks = []
    
    @property
    def size(self) -> int:
        """Get number of documents in the store."""
        return self.index.ntotal

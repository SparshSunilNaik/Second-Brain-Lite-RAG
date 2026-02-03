"""
Embedding backend for Second Brain Lite.

Provides abstraction for embedding models with caching to avoid
redundant API calls and costs.
"""

import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any
from abc import ABC, abstractmethod

from openai import OpenAI


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass


class OpenAIEmbeddings(EmbeddingBackend):
    """OpenAI embedding backend with caching."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        """
        Initialize OpenAI embeddings.
        
        Args:
            api_key: OpenAI API key
            model: Embedding model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.cache_file = Path("embeddings_cache.pkl")
        self.cache = self._load_cache()
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        cache_key = self._get_cache_key(text)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate new embedding
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        embedding = response.data[0].embedding
        
        # Cache the result
        self.cache[cache_key] = embedding
        self._save_cache()
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts efficiently.
        
        Uses batch API when possible, falls back to cached results.
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self.cache:
                embeddings.append(self.cache[cache_key])
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Batch embed uncached texts
        if uncached_texts:
            response = self.client.embeddings.create(
                input=uncached_texts,
                model=self.model
            )
            
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                original_index = uncached_indices[i]
                text = uncached_texts[i]
                
                # Update cache and results
                cache_key = self._get_cache_key(text)
                self.cache[cache_key] = embedding
                embeddings[original_index] = embedding
            
            self._save_cache()
        
        return embeddings
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text content."""
        # Use hash of text + model name for cache key
        content = f"{self.model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        """Load embedding cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save embedding cache to disk."""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")


class EmbeddingManager:
    """Manages embeddings for chunks with persistence."""
    
    def __init__(self, backend: EmbeddingBackend):
        """
        Initialize embedding manager.
        
        Args:
            backend: Embedding backend to use
        """
        self.backend = backend
    
    def embed_chunks(self, chunks: List[Any]) -> List[List[float]]:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of Chunk objects with .text attribute
            
        Returns:
            List of embedding vectors
        """
        texts = [chunk.text for chunk in chunks]
        return self.backend.embed_batch(texts)
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        return self.backend.embed_text(query)

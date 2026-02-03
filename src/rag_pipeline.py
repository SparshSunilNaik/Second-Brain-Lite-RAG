"""
RAG Pipeline for Second Brain Lite.

Orchestrates retrieval and generation with strict grounding to prevent
hallucination and ensure source attribution.
"""

from typing import List, Dict, Any, Tuple
from openai import OpenAI

from embeddings import EmbeddingManager
from vector_store import VectorStore


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    # System prompt enforcing strict grounding
    SYSTEM_PROMPT = """You are a helpful assistant that answers questions based STRICTLY on the provided context from the user's notes.

CRITICAL RULES:
1. Answer ONLY using information from the provided context
2. If the context doesn't contain enough information to answer the question, respond with: "I don't have enough information in your notes to answer this question."
3. Do not use external knowledge or make assumptions
4. When referencing information, be specific about which note it comes from
5. If multiple notes contain relevant information, synthesize them clearly

Your goal is to help the user understand what's in their notes, not to provide general knowledge."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        llm_api_key: str,
        llm_model: str = "gpt-4o-mini",
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            vector_store: Vector store for retrieval
            embedding_manager: Embedding manager for queries
            llm_api_key: OpenAI API key for LLM
            llm_model: LLM model to use
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval (lower = more strict)
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm_client = OpenAI(api_key=llm_api_key)
        self.llm_model = llm_model
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with 'answer', 'sources', and 'retrieved_chunks'
        """
        # Step 1: Embed the query
        query_embedding = self.embedding_manager.embed_query(question)
        
        # Step 2: Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, self.top_k)
        
        # Step 3: Filter by similarity threshold
        # Note: FAISS returns L2 distance, lower is better
        # We use a simple threshold; you may want to normalize this
        filtered_results = [
            (text, metadata, score) 
            for text, metadata, score in results 
            if score < self.similarity_threshold or self.similarity_threshold == 0
        ]
        
        # Step 4: Check if we have relevant results
        if not filtered_results:
            return {
                'answer': "I couldn't find any relevant information in your notes to answer this question.",
                'sources': [],
                'retrieved_chunks': []
            }
        
        # Step 5: Build context from retrieved chunks
        context = self._build_context(filtered_results)
        
        # Step 6: Generate answer with LLM
        answer = self._generate_answer(question, context)
        
        # Step 7: Extract sources
        sources = self._extract_sources(filtered_results)
        
        return {
            'answer': answer,
            'sources': sources,
            'retrieved_chunks': [
                {
                    'text': text,
                    'source': metadata.get('source', 'unknown'),
                    'score': score
                }
                for text, metadata, score in filtered_results
            ]
        }
    
    def _build_context(self, results: List[Tuple[str, Dict, float]]) -> str:
        """
        Build context string from retrieved chunks.
        
        Args:
            results: List of (text, metadata, score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (text, metadata, score) in enumerate(results, 1):
            source = metadata.get('source', 'unknown')
            chunk_idx = metadata.get('chunk_index', 0)
            heading = metadata.get('heading', '')
            
            # Format chunk with source attribution
            chunk_header = f"[Source {i}: {source}"
            if heading:
                chunk_header += f" - {heading}"
            chunk_header += f" (chunk {chunk_idx})]"
            
            context_parts.append(f"{chunk_header}\n{text}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with strict grounding.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        user_prompt = f"""Context from notes:

{context}

---

Question: {question}

Answer based ONLY on the context above:"""
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused answers
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def _extract_sources(self, results: List[Tuple[str, Dict, float]]) -> List[Dict[str, Any]]:
        """
        Extract unique sources from results.
        
        Args:
            results: List of (text, metadata, score) tuples
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_sources = set()
        
        for text, metadata, score in results:
            source = metadata.get('source', 'unknown')
            chunk_idx = metadata.get('chunk_index', 0)
            
            # Create unique identifier for this chunk
            source_id = f"{source}::{chunk_idx}"
            
            if source_id not in seen_sources:
                seen_sources.add(source_id)
                sources.append({
                    'file': source,
                    'chunk_index': chunk_idx,
                    'heading': metadata.get('heading'),
                    'similarity_score': score
                })
        
        return sources

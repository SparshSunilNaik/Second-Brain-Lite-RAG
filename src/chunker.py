"""
Intelligent document chunker for Second Brain Lite.

Splits documents into semantically coherent chunks while preserving
context and metadata for accurate retrieval.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    text: str
    metadata: Dict[str, Any]


class DocumentChunker:
    """Chunks documents intelligently based on file type."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Overlap between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a document based on its file type.
        
        Args:
            content: Document content
            metadata: Document metadata (must include 'file_type')
            
        Returns:
            List of Chunk objects
        """
        file_type = metadata.get('file_type', '.txt')
        
        if file_type == '.md':
            return self._chunk_markdown(content, metadata)
        else:
            return self._chunk_text(content, metadata)
    
    def _chunk_markdown(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk markdown by headings and paragraphs.
        
        Preserves heading hierarchy and creates semantically coherent chunks.
        """
        chunks = []
        
        # Split by headings (# to ######)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_chunk = []
        current_heading = None
        chunk_index = 0
        
        for line in lines:
            heading_match = re.match(heading_pattern, line)
            
            if heading_match:
                # Save previous chunk if it exists
                if current_chunk:
                    chunk_text = '\n'.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(self._create_chunk(
                            chunk_text, metadata, chunk_index, current_heading
                        ))
                        chunk_index += 1
                
                # Start new chunk with heading
                current_heading = heading_match.group(2)
                current_chunk = [line]
            else:
                current_chunk.append(line)
                
                # Check if chunk is getting too large
                chunk_text = '\n'.join(current_chunk)
                if len(chunk_text) > self.chunk_size:
                    # Split by paragraphs within this section
                    paragraphs = self._split_by_paragraphs(chunk_text)
                    for para in paragraphs:
                        if para.strip():
                            chunks.append(self._create_chunk(
                                para, metadata, chunk_index, current_heading
                            ))
                            chunk_index += 1
                    current_chunk = []
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                chunks.append(self._create_chunk(
                    chunk_text, metadata, chunk_index, current_heading
                ))
        
        return chunks if chunks else [self._create_chunk(content, metadata, 0, None)]
    
    def _chunk_text(self, content: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk plain text by paragraphs.
        
        Uses double newlines as paragraph boundaries.
        """
        paragraphs = self._split_by_paragraphs(content)
        chunks = []
        
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_size = len(para)
            
            # If single paragraph exceeds chunk size, split it
            if para_size > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(self._create_chunk(
                        chunk_text, metadata, chunk_index, None
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = []
                temp_size = 0
                
                for sentence in sentences:
                    if temp_size + len(sentence) > self.chunk_size and temp_chunk:
                        chunks.append(self._create_chunk(
                            ' '.join(temp_chunk), metadata, chunk_index, None
                        ))
                        chunk_index += 1
                        temp_chunk = []
                        temp_size = 0
                    
                    temp_chunk.append(sentence)
                    temp_size += len(sentence)
                
                if temp_chunk:
                    chunks.append(self._create_chunk(
                        ' '.join(temp_chunk), metadata, chunk_index, None
                    ))
                    chunk_index += 1
            
            # Add paragraph to current chunk
            elif current_size + para_size > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(self._create_chunk(
                    chunk_text, metadata, chunk_index, None
                ))
                chunk_index += 1
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(self._create_chunk(
                chunk_text, metadata, chunk_index, None
            ))
        
        return chunks if chunks else [self._create_chunk(content, metadata, 0, None)]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines (paragraph boundaries)."""
        return re.split(r'\n\s*\n', text)
    
    def _create_chunk(
        self, 
        text: str, 
        doc_metadata: Dict[str, Any], 
        chunk_index: int,
        heading: str = None
    ) -> Chunk:
        """Create a chunk with combined metadata."""
        chunk_metadata = {
            **doc_metadata,
            'chunk_index': chunk_index,
        }
        
        if heading:
            chunk_metadata['heading'] = heading
        
        return Chunk(text=text.strip(), metadata=chunk_metadata)

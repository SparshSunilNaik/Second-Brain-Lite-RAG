"""
Document loader for Second Brain Lite.

Recursively loads .md and .txt files from a directory,
preserving metadata for downstream processing.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Document:
    """Represents a loaded document with metadata."""
    content: str
    metadata: Dict[str, Any]


class DocumentLoader:
    """Loads text documents from a directory."""
    
    SUPPORTED_EXTENSIONS = {'.md', '.txt'}
    
    def __init__(self, notes_dir: str):
        """
        Initialize the document loader.
        
        Args:
            notes_dir: Path to the directory containing notes
        """
        self.notes_dir = Path(notes_dir)
        if not self.notes_dir.exists():
            raise ValueError(f"Notes directory does not exist: {notes_dir}")
    
    def load_documents(self) -> List[Document]:
        """
        Recursively load all supported documents from the notes directory.
        
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        
        for file_path in self._get_text_files():
            try:
                content = self._read_file(file_path)
                relative_path = file_path.relative_to(self.notes_dir)
                
                doc = Document(
                    content=content,
                    metadata={
                        'source': str(relative_path),
                        'file_type': file_path.suffix,
                        'absolute_path': str(file_path)
                    }
                )
                documents.append(doc)
                
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
                continue
        
        return documents
    
    def _get_text_files(self) -> List[Path]:
        """
        Get all text files from the notes directory.
        
        Returns:
            List of Path objects for supported files
        """
        text_files = []
        
        for root, dirs, files in os.walk(self.notes_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            root_path = Path(root)
            for file in files:
                file_path = root_path / file
                
                # Skip hidden files and unsupported extensions
                if file.startswith('.'):
                    continue
                if file_path.suffix not in self.SUPPORTED_EXTENSIONS:
                    continue
                
                text_files.append(file_path)
        
        return text_files
    
    def _read_file(self, file_path: Path) -> str:
        """
        Read file content with proper encoding.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        # Try UTF-8 first, fall back to latin-1 if needed
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()

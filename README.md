# Second Brain Lite

A local, production-quality Retrieval-Augmented Generation (RAG) system for querying your personal notes with strict grounding and source attribution.

## What is RAG?

**Retrieval-Augmented Generation (RAG)** is a technique that combines information retrieval with large language model (LLM) generation. Instead of relying solely on an LLM's training data (which can lead to hallucinations), RAG:

1. **Retrieves** relevant documents from your personal knowledge base
2. **Augments** the LLM's context with this retrieved information
3. **Generates** answers strictly based on the retrieved content

Think of it as giving the AI a "cheat sheet" of your notes before answering questions.

## Why This Project?

**Problem**: LLMs are powerful but can hallucinate facts, especially about your personal notes and knowledge.

**Solution**: Second Brain Lite ensures answers are:
- ✅ **Grounded** in your actual notes (no hallucinations)
- ✅ **Attributed** to specific sources (you know where info comes from)
- ✅ **Local-first** (your notes stay on your machine)
- ✅ **Transparent** (you can verify the retrieved chunks)

## How Retrieval Prevents Hallucination

Traditional LLM query:
```
You → LLM → Answer (may hallucinate)
```

RAG pipeline:
```
You → Embed Query → Search Vector Store → Retrieve Relevant Chunks → LLM + Context → Grounded Answer + Sources
```

**Key mechanisms**:
1. **Semantic search**: Finds chunks semantically similar to your query
2. **Context injection**: LLM only sees retrieved chunks, not its full training data
3. **Strict prompting**: System prompt enforces "answer only from context" rule
4. **Source tracking**: Every answer includes which files/chunks were used

If no relevant chunks are found, the system explicitly says "I don't have information on this" instead of making something up.

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Setup

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   cd second-brain-lite
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   ```bash
   # Copy the example env file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=sk-your-key-here
   ```

4. **Add your notes**:
   - Place `.md` and `.txt` files in the `notes/` directory
   - Organize in subdirectories if desired (e.g., `notes/ai/`, `notes/productivity/`)

## Architecture

Second Brain Lite offers **three ways to use it** - choose what fits your needs:

```
┌─────────────────────────────────────────────────────────────┐
│                    USAGE MODES                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. CLI Only (Standalone)                                   │
│     python src/main.py index                                │
│     python src/main.py query "question"                     │
│                                                             │
│  2. Web UI (Recommended)                                    │
│     Backend API + React Frontend                            │
│                                                             │
│  3. API Only                                                │
│     Use FastAPI backend with your own client                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

         Web UI Architecture (Mode 2)
         ────────────────────────────

    ┌─────────────────┐
    │  React Frontend │  ← Optional web interface
    │   (Port 5173)   │    Clean, minimal UI
    └────────┬────────┘
             │ HTTP/JSON
             ▼
    ┌─────────────────┐
    │  FastAPI Server │  ← Optional REST API
    │   (Port 8000)   │    /index, /query, /status
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  RAG Pipeline   │  ← Core Python modules
    │  + Vector Store │    Works standalone
    └─────────────────┘
```

**Key Principles**:
- ✅ **Frontend is optional** - CLI works standalone
- ✅ **Backend API is optional** - Use CLI if you prefer
- ✅ **No vendor lock-in** - Bring your own model provider
- ✅ **No forced API keys** - Use local models if desired
- ✅ **No authentication** - Local-first tool

## Usage

### Option 1: CLI (Standalone)

Perfect for quick queries and automation.

### 1. Index Your Notes

Before querying, you need to build the vector index:

```bash
python src/main.py index
```

This will:
- Load all `.md` and `.txt` files from `notes/`
- Chunk them intelligently (by headings for Markdown, by paragraphs for text)
- Generate embeddings using OpenAI's `text-embedding-3-small`
- Store vectors in a local FAISS index (`.faiss_index/`)

**Note**: Embeddings are cached in `embeddings_cache.pkl` to avoid re-embedding unchanged content.

### 2. Query Your Notes

```bash
python src/main.py query "What are my thoughts on machine learning?"
```

**Example output**:
```
================================================================================
ANSWER
================================================================================
Based on your notes, you view machine learning as transformative across 
industries, with a key focus on interpretability, fairness, and robustness. 
You emphasize the importance of understanding mathematical foundations rather 
than just using libraries. You're particularly interested in supervised 
learning for NLP tasks and find reinforcement learning fascinating for game 
AI and robotics.

================================================================================
SOURCES
================================================================================
1. ai/machine_learning.md - My Thoughts (chunk 4)
2. ai/machine_learning.md - Types of Machine Learning (chunk 2)
3. ai/machine_learning.md - What is Machine Learning? (chunk 1)
```

### More Example Queries

```bash
# Ask about specific topics
python src/main.py query "Tell me about the Zettelkasten method"

# Ask about connections between topics
python src/main.py query "How do deep learning and neural networks relate?"

# Ask about something NOT in your notes (tests grounding)
python src/main.py query "What is quantum computing?"
# Expected: "I don't have enough information in your notes to answer this."
```

### Option 2: Web UI (Recommended)

Use the clean, modern web interface for a better experience.

#### 1. Start the Backend API

```bash
# From the project root
python src/api.py

# Or using uvicorn directly
uvicorn src.api:app --reload
```

The API will start on `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

#### 2. Start the Frontend

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies (first time only)
npm install

# Start development server
npm run dev
```

The frontend will start on `http://localhost:5173`

#### 3. Use the Web Interface

1. **Index Notes**: Click "Index Notes" button
2. **Ask Questions**: Type your question and click "Submit Query"
3. **View Results**: See answer with source attribution

**Features**:
- Clean, minimal interface
- Real-time status updates
- Source file attribution
- Error handling with clear messages

### Option 3: API Only

Use the FastAPI backend with your own client (curl, Postman, custom app).

```bash
# Start the API
python src/api.py

# Check status
curl http://localhost:8000/status

# Index notes
curl -X POST http://localhost:8000/index

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are my thoughts on machine learning?"}'
```

**API Endpoints**:
- `GET /` - API information
- `GET /status` - Check if index exists
- `POST /index` - Index notes from notes/ directory
- `POST /query` - Query the RAG system

Full API documentation at: `http://localhost:8000/docs`

## Project Structure

```
second-brain-lite/
├── notes/                      # Your personal notes (.md, .txt)
│   ├── ai/
│   │   ├── machine_learning.md
│   │   └── deep_learning.md
│   └── productivity/
│       └── note_taking.md
├── src/
│   ├── loader.py              # Document loading
│   ├── chunker.py             # Intelligent chunking
│   ├── embeddings.py          # Embedding generation with caching
│   ├── vector_store.py        # FAISS vector storage
│   ├── rag_pipeline.py        # RAG orchestration
│   ├── main.py                # CLI interface
│   └── api.py                 # FastAPI backend (optional)
├── frontend/                   # React web UI (optional)
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Header.jsx
│   │   │   ├── IndexPanel.jsx
│   │   │   ├── QueryPanel.jsx
│   │   │   └── AnswerPanel.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── tests/
│   └── test_retrieval.py      # Automated tests
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Configuration

Edit `.env` to customize behavior:

```bash
# Embedding model (default: text-embedding-3-small)
EMBEDDING_MODEL=text-embedding-3-small

# LLM for answer generation (default: gpt-4o-mini)
LLM_MODEL=gpt-4o-mini

# Number of chunks to retrieve (default: 5)
TOP_K=5

# Similarity threshold - lower = stricter (default: 0.3)
SIMILARITY_THRESHOLD=0.3
```

## Testing

Run the automated test suite:

```bash
# Make sure OPENAI_API_KEY is set in .env
pytest tests/test_retrieval.py -v
```

Tests cover:
- Document loading and chunking
- Vector store operations
- End-to-end query pipeline
- Grounding behavior (irrelevant queries)

## Limitations

**Current constraints**:
- **Requires API key**: Uses OpenAI for embeddings and generation (costs ~$0.01-0.10 per session)
- **English-optimized**: Works best with English text
- **No OCR**: Can't read images or PDFs (text only)
- **Static index**: Need to re-run `index` after adding/modifying notes
- **No conversation history**: Each query is independent

**Quality depends on**:
- How well your notes are written
- Chunk size and retrieval settings
- Quality of your questions

## Future Extensions (TODO)

Potential improvements:

- [ ] **Local embeddings**: Swap OpenAI embeddings for local models (sentence-transformers)
- [ ] **Local LLM**: Use Ollama or llama.cpp for fully offline operation
- [ ] **Incremental indexing**: Only re-index changed files
- [ ] **PDF support**: Extract text from PDFs
- [ ] **Web UI**: Build a simple web interface (Streamlit/Gradio)
- [ ] **Conversation mode**: Multi-turn conversations with context
- [ ] **Hybrid search**: Combine semantic search with keyword search (BM25)
- [ ] **Metadata filtering**: Filter by date, tags, or folders
- [ ] **Citation extraction**: Show exact sentences used from sources

## How It Works (Technical)

### 1. Document Processing
- **Loader**: Recursively scans `notes/` for `.md` and `.txt` files
- **Chunker**: 
  - Markdown: Splits by headings, preserves hierarchy
  - Text: Splits by paragraphs
  - Target: 800 chars/chunk with 100 char overlap

### 2. Embedding Generation
- Uses OpenAI's `text-embedding-3-small` (1536 dimensions)
- Caches embeddings by content hash to minimize API calls
- Batch processing for efficiency

### 3. Vector Storage
- FAISS IndexFlatL2 for exact L2 distance search
- Metadata stored separately (source file, chunk index, headings)
- Persisted to disk for reuse

### 4. Query Pipeline
```python
query → embed → search_top_k → filter_by_similarity → build_context → LLM(context + query) → answer + sources
```

### 5. Grounding Mechanism
- **System prompt**: Explicitly forbids using external knowledge
- **Context-only**: LLM only sees retrieved chunks
- **Fallback**: Returns "no information" if similarity too low
- **Temperature**: Set to 0.3 for focused, deterministic answers

## Contributing

This is a learning/portfolio project. Feel free to fork and extend!

## Acknowledgments

Built as a demonstration of production-quality RAG system design, emphasizing:
- Clean architecture
- Honest limitations
- Practical grounding techniques
- Developer-friendly code

---

**Built with**: Python, OpenAI API, FAISS, pytest

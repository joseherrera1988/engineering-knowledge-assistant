"""
Document Ingestion Pipeline
Handles PDF, Markdown, and text files with intelligent chunking
"""

import os
import re
from typing import List, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Represents a document chunk with metadata"""
    content: str
    source: str
    chunk_id: int
    start_line: int = 0
    end_line: int = 0


class DocumentIngester:
    """Ingests and chunks documents intelligently"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def ingest_directory(self, directory: str) -> List[Document]:
        """Ingest all documents from a directory"""
        documents = []
        doc_id = 0
        
        for file_path in Path(directory).rglob("*"):
            if file_path.suffix.lower() in [".txt", ".md", ".py", ".js", ".sql"]:
                chunks = self.ingest_file(str(file_path))
                for chunk in chunks:
                    chunk.chunk_id = doc_id
                    documents.append(chunk)
                    doc_id += 1
        
        return documents
    
    def ingest_file(self, file_path: str) -> List[Document]:
        """Ingest a single file and return chunks"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        file_type = Path(file_path).suffix.lower()
        
        if file_type == ".md":
            return self._chunk_markdown(content, file_path)
        elif file_type == ".py" or file_type == ".js":
            return self._chunk_code(content, file_path)
        else:
            return self._chunk_text(content, file_path)
    
    def _chunk_markdown(self, content: str, source: str) -> List[Document]:
        """Chunk markdown by headers and paragraphs"""
        documents = []
        lines = content.split('\n')
        
        current_chunk = []
        chunk_id = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Split on headers or when chunk size exceeded
            is_header = line.startswith('#')
            chunk_text = '\n'.join(current_chunk).strip()
            
            if (is_header and current_chunk != [line]) or len(chunk_text) > self.chunk_size:
                if current_chunk and current_chunk != [line]:
                    chunk_text = '\n'.join(current_chunk[:-1] if is_header else current_chunk).strip()
                    if chunk_text:
                        documents.append(Document(
                            content=chunk_text,
                            source=source,
                            chunk_id=chunk_id,
                            start_line=start_line,
                            end_line=i
                        ))
                        chunk_id += 1
                
                current_chunk = [line] if is_header else []
                start_line = i
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text:
                documents.append(Document(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    start_line=start_line,
                    end_line=len(lines)
                ))
        
        return documents
    
    def _chunk_code(self, content: str, source: str) -> List[Document]:
        """Chunk code by functions/classes"""
        documents = []
        lines = content.split('\n')
        
        current_chunk = []
        chunk_id = 0
        start_line = 0
        
        for i, line in enumerate(lines):
            current_chunk.append(line)
            
            # Split on function/class definitions or size limit
            is_def = line.strip().startswith(('def ', 'class ', 'async def ', 'function ', 'const ', 'let '))
            chunk_text = '\n'.join(current_chunk).strip()
            
            if (is_def and len(current_chunk) > 1) or len(chunk_text) > self.chunk_size:
                if current_chunk and current_chunk != [line]:
                    chunk_text = '\n'.join(current_chunk[:-1] if is_def else current_chunk).strip()
                    if chunk_text and len(chunk_text) > 20:
                        documents.append(Document(
                            content=chunk_text,
                            source=source,
                            chunk_id=chunk_id,
                            start_line=start_line,
                            end_line=i
                        ))
                        chunk_id += 1
                
                current_chunk = [line] if is_def else []
                start_line = i
        
        if current_chunk:
            chunk_text = '\n'.join(current_chunk).strip()
            if chunk_text and len(chunk_text) > 20:
                documents.append(Document(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    start_line=start_line,
                    end_line=len(lines)
                ))
        
        return documents
    
    def _chunk_text(self, content: str, source: str) -> List[Document]:
        """Chunk plain text with overlap"""
        documents = []
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = []
        chunk_id = 0
        
        for sentence in sentences:
            current_chunk.append(sentence)
            chunk_text = ' '.join(current_chunk)
            
            if len(chunk_text) > self.chunk_size:
                documents.append(Document(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id
                ))
                
                # Keep overlap
                overlap_sentences = max(1, len(current_chunk) // 3)
                current_chunk = current_chunk[-overlap_sentences:]
                chunk_id += 1
        
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if chunk_text:
                documents.append(Document(
                    content=chunk_text,
                    source=source,
                    chunk_id=chunk_id
                ))
        
        return documents


def create_sample_documents(directory: str = "/home/claude/sample_docs") -> None:
    """Create sample engineering documentation"""
    os.makedirs(directory, exist_ok=True)
    
    # API Documentation
    api_doc = """# API Documentation

## Overview
This is the REST API for the DataPipeline system. All endpoints require authentication via Bearer token.

## Authentication
Include your API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### GET /api/documents
Retrieve a list of documents.

**Parameters:**
- `limit` (int): Maximum number of results (default: 10)
- `offset` (int): Number of results to skip (default: 0)
- `search` (string): Optional search query

**Response:**
```json
{
  "documents": [
    {
      "id": "doc-123",
      "title": "System Architecture",
      "created_at": "2024-01-15T10:30:00Z",
      "updated_at": "2024-01-20T15:45:00Z"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

### POST /api/documents
Create a new document.

**Request Body:**
```json
{
  "title": "New Document",
  "content": "Document content here",
  "tags": ["engineering", "api"]
}
```

**Response:**
```json
{
  "id": "doc-456",
  "title": "New Document",
  "created_at": "2024-01-22T10:00:00Z"
}
```

### GET /api/documents/{id}
Retrieve a specific document.

**Response:**
Same as POST response body with full content.

### PUT /api/documents/{id}
Update a document.

### DELETE /api/documents/{id}
Delete a document.

## Error Handling
All errors return a standard error response:
```json
{
  "error": "Error message",
  "status": 400,
  "timestamp": "2024-01-22T10:00:00Z"
}
```

Common status codes:
- 200: Success
- 400: Bad request
- 401: Unauthorized
- 404: Not found
- 500: Internal server error

## Rate Limiting
API requests are limited to 100 per minute per API key.
"""
    
    with open(f"{directory}/api_docs.md", "w") as f:
        f.write(api_doc)
    
    # System Architecture
    arch_doc = """# System Architecture

## Overview
The system is built on a microservices architecture with the following components:

### Core Services

#### 1. Document Service
Responsible for document management, versioning, and storage.
- Language: Python/FastAPI
- Database: PostgreSQL
- Cache: Redis
- Responsibilities:
  - CRUD operations on documents
  - Document versioning and history
  - Full-text search indexing
  - Access control and permissions

#### 2. Search Service
Provides semantic and full-text search capabilities.
- Language: Python
- Search Engine: Elasticsearch + FAISS
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Features:
  - Semantic similarity search
  - Full-text search with BM25
  - Vector-based retrieval (FAISS)
  - Faceted search and filtering

#### 3. Analytics Service
Tracks usage, performance metrics, and user analytics.
- Language: Go
- Message Queue: Apache Kafka
- Time Series DB: InfluxDB
- Monitoring: Prometheus + Grafana

### Data Flow

1. User uploads document
   → Document Service validates and stores
   → Triggers indexing pipeline
   → Search Service ingests and chunks
   → Embeddings generated
   → FAISS index updated
   → Cache invalidated in Redis

2. User performs search
   → Query Service receives request
   → Generates embedding for query
   → FAISS semantic search
   → Elasticsearch full-text search
   → Results ranked and merged
   → Response cached
   → Analytics logged

### Deployment
- Container: Docker
- Orchestration: Kubernetes
- Cloud Provider: AWS (EKS)
- CI/CD: GitHub Actions
- Infrastructure as Code: Terraform
"""
    
    with open(f"{directory}/architecture.md", "w") as f:
        f.write(arch_doc)
    
    # Database Schema
    db_schema = """# Database Schema

## Tables

### documents
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    file_type VARCHAR(50),
    source_path VARCHAR(512),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by UUID NOT NULL REFERENCES users(id),
    is_deleted BOOLEAN DEFAULT FALSE,
    INDEX idx_created_at (created_at),
    INDEX idx_title (title),
    FULLTEXT INDEX ft_content (content)
);
```

### document_chunks
```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_number INT NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(384),
    embedding_model VARCHAR(255),
    tokens INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_document_id (document_id),
    INDEX idx_embedding (embedding) USING hnsw
);
```

### users
```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    role VARCHAR(50) DEFAULT 'user'
);
```

### queries
```sql
CREATE TABLE queries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id),
    query_text TEXT NOT NULL,
    query_embedding VECTOR(384),
    results_count INT,
    execution_time_ms INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_user_id (created_at),
    INDEX idx_created_at (created_at)
);
```

## Indexes
- document_chunks uses HNSW index for vector similarity
- Full-text search index on documents.content
- Timestamp indexes for fast querying of recent documents
"""
    
    with open(f"{directory}/database_schema.md", "w") as f:
        f.write(db_schema)
    
    # Python code sample
    python_code = '''"""
Vector Search Module
Implements efficient similarity search using FAISS
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class VectorSearch:
    """FAISS-based vector search with sentence transformers"""
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with embedding model"""
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def build_index(self, documents):
        """Build FAISS index from documents"""
        embeddings = self.model.encode(
            [doc.content for doc in documents],
            convert_to_numpy=True
        )
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
        self.documents = documents
        print(f"Built index with {len(documents)} documents")
    
    def search(self, query, k=5):
        """Search for top-k similar documents"""
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True
        ).astype(np.float32).reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append({
                    "document": self.documents[idx],
                    "distance": float(distances[0][i]),
                    "similarity": 1 / (1 + distances[0][i])
                })
        
        return results
    
    def save(self, path):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}/index.faiss")
        # Save documents metadata
        import pickle
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
    
    def load(self, path):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}/index.faiss")
        import pickle
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
'''
    
    with open(f"{directory}/vector_search.py", "w") as f:
        f.write(python_code)
    
    print(f"✓ Sample documents created in {directory}")

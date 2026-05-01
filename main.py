#!/usr/bin/env python3
"""
RAG System Quick Start
Initialize the vector store and launch the Streamlit UI
"""

import os
import subprocess
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import Document, DocumentIngester, create_sample_documents
from paths import APP_PY, INDEX_DIR, SAMPLE_DOCS_DIR
from rag_agent import RAGAgent
from vector_store import FAISSVectorStore


def initialize_system() -> tuple[FAISSVectorStore, list[Document]]:
    """Initialize the RAG system"""
    print("=" * 70)
    print("🚀 RAG System Initialization")
    print("=" * 70)

    # Step 1: Create sample documents
    print("\n[1/4] Creating sample documents...")
    docs_dir = str(SAMPLE_DOCS_DIR)
    create_sample_documents(docs_dir)

    # Step 2: Ingest documents
    print("\n[2/4] Ingesting documents...")
    ingester = DocumentIngester(chunk_size=512, overlap=100)
    documents = ingester.ingest_directory(docs_dir)
    print(f"  ✓ Ingested {len(documents)} document chunks")

    # Step 3: Build vector store
    print("\n[3/4] Building vector store...")
    vector_store = FAISSVectorStore()
    vector_store.add_documents(documents)

    # Step 4: Save index
    print("\n[4/4] Saving index...")
    index_dir = str(INDEX_DIR)
    vector_store.save(index_dir)
    print(f"  ✓ Index saved to {index_dir}")

    # Display stats
    stats = vector_store.get_document_stats()
    print("\n" + "=" * 70)
    print("📊 System Statistics")
    print("=" * 70)
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total indexed: {stats['total_indexed']}")
    print(f"Unique sources: {stats['unique_sources']}")
    print(f"Embedding dimension: {stats['embedding_dimension']}")
    print("\nSources:")
    for source in stats["sources"]:
        print(f"  - {source}")

    print("\n" + "=" * 70)
    print("✓ System initialized successfully!")
    print("=" * 70)

    return vector_store, documents


def run_demo_queries(agent: RAGAgent, queries: list[str]) -> None:
    """Run demo queries to test the system"""
    print("\n" + "=" * 70)
    print("🎯 Demo Queries")
    print("=" * 70)

    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 70)

        response = agent.query(query, k=5)

        print(
            f"\n📝 Answer:\n{response.answer[:500]}..."
            if len(response.answer) > 500
            else f"\n📝 Answer:\n{response.answer}"
        )
        print(f"\n✓ Tool: {response.tool_used}")
        print(f"✓ Confidence: {response.confidence:.1%}")
        print(f"✓ Sources: {len(response.sources)}")


def main() -> None:
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument(
        "--mode", choices=["init", "demo", "ui", "full"], default="full", help="Operation mode"
    )

    args = parser.parse_args()

    if args.mode in ["init", "full"]:
        vector_store, documents = initialize_system()

    if args.mode in ["demo", "full"]:
        if args.mode == "full":
            agent = RAGAgent(vector_store)
        else:
            # Load from saved index
            vector_store = FAISSVectorStore()
            vector_store.load(str(INDEX_DIR))
            agent = RAGAgent(vector_store)

        demo_queries = [
            "What authentication methods are supported by the API?",
            "Summarize the system architecture",
            "Generate a SQL query to get recent documents",
            "How is the FAISS vector store implemented?",
        ]

        run_demo_queries(agent, demo_queries)

    if args.mode in ["ui", "full"]:
        print("\n" + "=" * 70)
        print("🌐 Launching Streamlit UI...")
        print("=" * 70)
        print("\nThe app will open at: http://localhost:8501")
        print("Press Ctrl+C to stop the server\n")

        # Launch Streamlit
        subprocess.run(["streamlit", "run", str(APP_PY), "--logger.level=error"])


if __name__ == "__main__":
    main()

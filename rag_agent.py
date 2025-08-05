"""Pydantic AI agent that leverages RAG with a local ChromaDB for Pydantic documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import asyncio
import chromadb
import re
from pathlib import Path

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI

from utils import (
    get_chroma_client,
    get_or_create_collection,
    query_collection,
    format_results_as_context,
    add_documents_to_collection
)

# Load environment variables from .env file
dotenv.load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please create a .env file with your OpenAI API key or set it in your environment.")
    sys.exit(1)


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    chroma_client: chromadb.PersistentClient
    collection_name: str
    embedding_model: str


class RagAgent:
    """RAG Agent for handling document insertion from URLs and local files."""
    
    def __init__(self, collection_name: str = "docs", db_directory: str = "./chroma_db", 
                 embedding_model: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.db_directory = db_directory
        self.embedding_model = embedding_model
        self.client = get_chroma_client(db_directory)
        self.collection = get_or_create_collection(
            self.client, 
            collection_name, 
            embedding_model_name=embedding_model
        )
    
    def smart_chunk_markdown(self, markdown: str, max_len: int = 1000) -> List[str]:
        """Hierarchically splits markdown by #, ##, ### headers, then by characters, to ensure all chunks < max_len."""
        def split_by_header(md, header_pattern):
            indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
            indices.append(len(md))
            return [md[indices[i]:indices[i+1]].strip() for i in range(len(indices)-1) if md[indices[i]:indices[i+1]].strip()]

        chunks = []

        for h1 in split_by_header(markdown, r'^# .+$'):
            if len(h1) > max_len:
                for h2 in split_by_header(h1, r'^## .+$'):
                    if len(h2) > max_len:
                        for h3 in split_by_header(h2, r'^### .+$'):
                            if len(h3) > max_len:
                                for i in range(0, len(h3), max_len):
                                    chunks.append(h3[i:i+max_len].strip())
                            else:
                                chunks.append(h3)
                    else:
                        chunks.append(h2)
            else:
                chunks.append(h1)

        final_chunks = []

        for c in chunks:
            if len(c) > max_len:
                final_chunks.extend([c[i:i+max_len].strip() for i in range(0, len(c), max_len)])
            else:
                final_chunks.append(c)

        return [c for c in final_chunks if c]
    
    def extract_section_info(self, chunk: str) -> Dict[str, Any]:
        """Extracts headers and stats from a chunk."""
        headers = re.findall(r'^(#+)\s+(.+)$', chunk, re.MULTILINE)
        header_str = '; '.join([f'{h[0]} {h[1]}' for h in headers]) if headers else ''

        return {
            "headers": header_str,
            "char_count": len(chunk),
            "word_count": len(chunk.split())
        }
    
    async def insert_from_url(self, url: str, chunk_size: int = 1000, max_depth: int = 3, 
                             max_concurrent: int = 10, batch_size: int = 100) -> None:
        """Insert documents from a URL into the collection."""
        from insert_docs import (
            is_sitemap, is_txt, crawl_recursive_internal_links, 
            crawl_markdown_file, parse_sitemap, crawl_batch
        )
        
        # Detect URL type and crawl
        if is_txt(url):
            print(f"Detected .txt/markdown file: {url}")
            crawl_results = await crawl_markdown_file(url)
        elif is_sitemap(url):
            print(f"Detected sitemap: {url}")
            sitemap_urls = parse_sitemap(url)
            if not sitemap_urls:
                print("No URLs found in sitemap.")
                return
            crawl_results = await crawl_batch(sitemap_urls, max_concurrent=max_concurrent)
        else:
            print(f"Detected regular URL: {url}")
            crawl_results = await crawl_recursive_internal_links([url], max_depth=max_depth, max_concurrent=max_concurrent)
        
        # Process and insert chunks
        self._process_and_insert_chunks(crawl_results, chunk_size, batch_size)
    
    async def insert_from_file(self, file_path: str, chunk_size: int = 1000, batch_size: int = 100) -> None:
        """Insert documents from a local file into the collection."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File {file_path} does not exist.")
            return
        
        print(f"Processing local file: {file_path}")
        
        # Read the file content based on file type
        try:
            if file_path.suffix.lower() == '.pdf':
                content = self._read_pdf(file_path)
            else:
                # For other file types, use the existing logic
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return
        
        # Create a single document result
        crawl_results = [{'url': str(file_path), 'markdown': content}]
        
        # Process and insert chunks
        self._process_and_insert_chunks(crawl_results, chunk_size, batch_size)
    
    def _read_pdf(self, file_path: Path) -> str:
        """Extract text from a PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("Error: PyMuPDF (fitz) is required for PDF processing.")
            print("Please install it with: pip install PyMuPDF")
            raise ImportError("PyMuPDF not installed")
        
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def _process_and_insert_chunks(self, crawl_results: List[Dict[str, Any]], chunk_size: int, batch_size: int) -> None:
        """Process crawl results and insert chunks into the collection."""
        # Chunk and collect metadata
        ids, documents, metadatas = [], [], []
        chunk_idx = 0
        
        for doc in crawl_results:
            url = doc['url']
            md = doc['markdown']
            chunks = self.smart_chunk_markdown(md, max_len=chunk_size)
            
            for chunk in chunks:
                ids.append(f"chunk-{chunk_idx}")
                documents.append(chunk)
                meta = self.extract_section_info(chunk)
                meta["chunk_index"] = chunk_idx
                meta["source"] = url
                metadatas.append(meta)
                chunk_idx += 1

        if not documents:
            print("No documents found to insert.")
            return

        print(f"Inserting {len(documents)} chunks into ChromaDB collection '{self.collection_name}'...")
        
        add_documents_to_collection(
            self.collection, 
            ids, 
            documents, 
            metadatas, 
            batch_size=batch_size
        )
        
        print(f"Successfully added {len(documents)} chunks to ChromaDB collection '{self.collection_name}'.")


# Create the RAG agent
agent = Agent(
    os.getenv("MODEL_CHOICE", "gpt-4.1-mini"),
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response."
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str, n_results: int = 5) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        n_results: Number of results to return (default: 5).
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    # Get ChromaDB client and collection
    collection = get_or_create_collection(
        context.deps.chroma_client,
        context.deps.collection_name,
        embedding_model_name=context.deps.embedding_model
    )
    
    # Query the collection
    query_results = query_collection(
        collection,
        search_query,
        n_results=n_results
    )
    
    # Format the results as context
    return format_results_as_context(query_results)


async def run_rag_agent(
    question: str,
    collection_name: str = "docs",
    db_directory: str = "./chroma_db",
    embedding_model: str = "all-MiniLM-L6-v2",
    n_results: int = 5
) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        collection_name: Name of the ChromaDB collection to use.
        db_directory: Directory where ChromaDB data is stored.
        embedding_model: Name of the embedding model to use.
        n_results: Number of results to return from the retrieval.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    deps = RAGDeps(
        chroma_client=get_chroma_client(db_directory),
        collection_name=collection_name,
        embedding_model=embedding_model
    )
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    parser.add_argument("--collection", default="docs", help="Name of the ChromaDB collection")
    parser.add_argument("--db-dir", default="./chroma_db", help="Directory where ChromaDB data is stored")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Name of the embedding model to use")
    parser.add_argument("--n-results", type=int, default=5, help="Number of results to return from the retrieval")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(
        args.question,
        collection_name=args.collection,
        db_directory=args.db_dir,
        embedding_model=args.embedding_model,
        n_results=args.n_results
    ))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()

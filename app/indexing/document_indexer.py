from typing import List
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import faiss
import os
from dotenv import load_dotenv
from pathlib import Path
import json

load_dotenv()  # Load .env before initializing DocumentIndexer

class DocumentIndexer:
    def __init__(self, docs_dir: str, embed_dir: str):
        # Configure embeddings
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        # Configure LLM
        Settings.llm = OpenAI(
            model="gpt-4o",
            temperature=0.3,
            max_tokens=512
        )
        
        self.docs_dir = docs_dir
        self.embed_dir = embed_dir
        self.index = None
        self.initialize_index()
    
    def initialize_index(self):
        """Initialize index with all required JSON files"""
        index_path = Path(self.embed_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Add index_store.json to required files
        required_files = {
            "docstore.json": {"docs": {}, "metadata": {}},
            "vector_store.json": {"index": None, "config": {}},
            "graph_store.json": {"nodes": [], "edges": []},
            "index_store.json": {}  # New required file
        }
        
        for filename, default_content in required_files.items():
            file_path = index_path / filename
            if not file_path.exists() or file_path.stat().st_size == 0:
                with open(file_path, "w") as f:
                    json.dump(default_content, f)
        
        # 2. Initialize FAISS components
        faiss_index = faiss.IndexFlatL2(1536)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        
        # 3. Load storage context with validated files
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(index_path)
        )
        
        # 4. Create/load index
        if not SimpleDirectoryReader(self.docs_dir).load_data():
            # Empty index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
        else:
            # Create index with documents
            self.index = VectorStoreIndex.from_documents(
                SimpleDirectoryReader(self.docs_dir).load_data(),
                storage_context=storage_context,
                show_progress=True
            )
        
        storage_context.persist()
    
    async def retrieve_relevant_context(self, query: str) -> List[str]:
        """Retrieve context with query optimization"""
        if not self.index:
            raise ValueError("Index not initialized")
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
        response = await query_engine.aquery(query)
        return [node.text for node in response.source_nodes]

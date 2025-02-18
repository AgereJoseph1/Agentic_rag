from typing import List
from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext

class DocumentIndexer:
    def __init__(self):
        self.index = None
        self.initialize_index()
    
    def initialize_index(self):
        """
        Initialize the document index using LlamaIndex
        """
        # Load documents from the data directory
        documents = SimpleDirectoryReader(
            input_dir="data/documents"
        ).load_data()
        
        # Create vector store
        vector_store = ChromaVectorStore(persist_dir="data/embeddings")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context
        )
    
    async def retrieve_relevant_context(self, query: str) -> List[str]:
        """
        Retrieve relevant context based on the query
        """
        if not self.index:
            raise ValueError("Index not initialized")
        
        # Query the index
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        
        return response.source_nodes

import os
import json

def create_directory_structure():
    # Define the base directory name
    base_dir = "portfolio_rag_agent"
    
    # Define the directory structure
    directories = [
        "",  # Base directory
        "app",
        "app/agent",
        "app/indexing",
        "app/utils",
        "data",
        "data/documents",
        "data/embeddings",
        "config",
        "tests",
    ]
    
    # Create directories
    for dir_path in directories:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)
        print(f"Created directory: {full_path}")

def create_main_app():
    content = '''from fastapi import FastAPI
from app.agent.portfolio_agent import PortfolioAgent
from app.indexing.document_indexer import DocumentIndexer

app = FastAPI(title="Portfolio RAG Agent")
agent = None

@app.on_event("startup")
async def startup_event():
    global agent
    indexer = DocumentIndexer()
    agent = PortfolioAgent(indexer)

@app.post("/query")
async def query_agent(query: str):
    """
    Endpoint to query the portfolio agent
    """
    if not agent:
        return {"error": "Agent not initialized"}
    
    response = await agent.process_query(query)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    with open("portfolio_rag_agent/app/main.py", "w") as f:
        f.write(content)

def create_portfolio_agent():
    content = '''from typing import List, Dict
from app.indexing.document_indexer import DocumentIndexer

class PortfolioAgent:
    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer
        self.conversation_history = []
        
    async def process_query(self, query: str) -> str:
        """
        Process incoming queries about the portfolio
        """
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Retrieve relevant context from the index
        context = await self.indexer.retrieve_relevant_context(query)
        
        # Generate response using LlamaIndex
        response = await self.generate_response(query, context)
        
        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    async def generate_response(self, query: str, context: List[str]) -> str:
        """
        Generate a response using LlamaIndex and the retrieved context
        """
        # TODO: Implement response generation using LlamaIndex
        # This will involve creating a custom response synthesis module
        pass
'''
    
    with open("portfolio_rag_agent/app/agent/portfolio_agent.py", "w") as f:
        f.write(content)

def create_document_indexer():
    content = '''from typing import List
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
'''
    
    with open("portfolio_rag_agent/app/indexing/document_indexer.py", "w") as f:
        f.write(content)

def create_config():
    config = {
        "model": {
            "name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "indexing": {
            "chunk_size": 512,
            "chunk_overlap": 50
        },
        "retrieval": {
            "top_k": 3
        }
    }
    
    with open("portfolio_rag_agent/config/config.json", "w") as f:
        json.dump(config, f, indent=4)

def create_requirements():
    content = '''fastapi==0.68.0
uvicorn==0.15.0
llama-index==0.8.8
chromadb==0.3.0
pydantic==1.8.2
python-dotenv==0.19.0
'''
    
    with open("portfolio_rag_agent/requirements.txt", "w") as f:
        f.write(content)

def create_readme():
    content = '''# Portfolio RAG Agent

An intelligent agent built with LlamaIndex that showcases professional expertise through interactive conversations about your portfolio.

## Features

- Document indexing and retrieval using LlamaIndex
- Conversational AI agent for portfolio interaction
- FastAPI backend for easy deployment
- Customizable response generation
- Persistent vector storage using Chroma

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your portfolio documents to `data/documents/`

4. Run the application:
   ```bash
   python app/main.py
   ```

## Project Structure

```
portfolio_rag_agent/
├── app/
│   ├── agent/
│   │   └── portfolio_agent.py
│   ├── indexing/
│   │   └── document_indexer.py
│   ├── utils/
│   └── main.py
├── data/
│   ├── documents/
│   └── embeddings/
├── config/
│   └── config.json
├── tests/
└── requirements.txt
```

## Configuration

Adjust settings in `config/config.json` to customize:
- Model parameters
- Indexing settings
- Retrieval parameters

## License

MIT
'''
    
    with open("portfolio_rag_agent/README.md", "w", encoding="utf-8") as f:
        f.write(content)

def main():
    print("Creating project directory structure...")
    create_directory_structure()
    
    print("\nCreating main application file...")
    create_main_app()
    
    print("\nCreating portfolio agent...")
    create_portfolio_agent()
    
    print("\nCreating document indexer...")
    create_document_indexer()
    
    print("\nCreating configuration file...")
    create_config()
    
    print("\nCreating requirements file...")
    create_requirements()
    
    print("\nCreating README...")
    create_readme()
    
    print("\nProject setup complete! You can find your project in the 'portfolio_rag_agent' directory.")

if __name__ == "__main__":
    main()
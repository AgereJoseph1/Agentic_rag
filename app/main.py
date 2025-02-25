import sys
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Configure paths
BASE_DIR = PROJECT_ROOT
DOCS_DIR = BASE_DIR / "data" / "documents"
EMBED_DIR = BASE_DIR / "data" / "embeddings"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create directories first
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    from app.indexing.document_indexer import DocumentIndexer
    from app.agent.portfolio_agent import PortfolioAgent
    
    try:
        indexer = DocumentIndexer(str(DOCS_DIR), str(EMBED_DIR))
        app.state.agent = PortfolioAgent(indexer)  # Store in app.state
        yield
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

app = FastAPI(
    title="Portfolio AI Agent",
    description="RAG-powered portfolio analysis system",
    lifespan=lifespan
)

@app.post("/chat")
async def chat_endpoint(request: Request, user_input: str):
    """Handle chat interactions with the AI agent"""
    agent = request.app.state.agent
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        response = await agent.process_query(user_input)
        return {"response": response}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

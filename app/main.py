from fastapi import FastAPI
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

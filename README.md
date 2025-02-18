# Portfolio RAG Agent

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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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


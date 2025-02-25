from typing import List
from app.indexing.document_indexer import DocumentIndexer
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings

class PortfolioAgent:
    def __init__(self, indexer: DocumentIndexer):
        self.indexer = indexer
        self.chat_history = []
        
    async def process_query(self, query: str) -> str:
        """Full RAG pipeline with conversation history"""
        # Retrieve relevant context
        context = await self.indexer.retrieve_relevant_context(query)
        
        # Format conversation history
        messages = self._format_messages(query, context)
        
        # Generate response
        response = await Settings.llm.achat(messages)
        
        # Update history
        self._update_history(query, response.message.content)
        
        return response.message.content

    def _format_messages(self, query: str, context: List[str]) -> List[ChatMessage]:
        """Strict context-enforced system prompt"""
        system_prompt = f"""
        ROLE: Portfolio Analysis Specialist
        DOMAIN: Financial documents provided by user
        SAFETY PROTOCOLS:
        1. RESPONSE VALIDATION:
        - Answer ONLY using verified context below
        - If context doesn't contain answer, respond: "I don't have information about that in your portfolio documents"
        
        2. CONTEXT ANALYSIS:
        Relevant Context:
        {''.join(context[:3]) or "No relevant documents found"}
        
        3. QUERY CLASSIFICATION:
        - Financial/Portfolio Related: Provide detailed analysis
        - Other Topics: "I specialize only in portfolio analysis based on your documents"
        
        4. RESPONSE RULES:
        - Never speculate beyond documents
        - Cite exact document excerpts used
        - Reject requests for predictions/advice
        
        Current Query: {query}
        """
        
        return [
            ChatMessage(role="system", content=system_prompt.strip()),
            *self.chat_history[-6:],  # Only last 3 exchanges
            ChatMessage(role="user", content=query)
        ]

    def _update_history(self, query: str, response: str):
        """Only keep valid responses in history"""
        if "I don't have information" not in response and "I specialize only" not in response:
            self.chat_history.extend([
                ChatMessage(role="user", content=query),
                ChatMessage(role="assistant", content=response)
            ])
            self.chat_history = self.chat_history[-8:]  # Last 4 exchanges

    async def generate_response(self, query: str, context: List[str]) -> str:
        """To be implemented with proper LLM integration"""
        raise NotImplementedError("Response generation not implemented yet")

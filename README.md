# HPE SANTA
LLM-as-a-Judge: RAG-Powered Document Query System
Overview
This project implements Retrieval-Augmented Generation (RAG) to enhance document querying with local LLMs, built using:
- LangGraph → Manages LLM workflows and decision-making.
- Qdrant → A high-performance vector database for fast retrieval.
- Ollama → Runs LLMs locally for inference.
- Streamlit → Provides an interactive UI for querying and visualization.


LLM-as-Agents: Multi-Agent Orchestrated RAG System
Overview
This project extends Retrieval-Augmented Generation (RAG) with an agentic architecture, where specialized agents collaborate to process queries in a modular and efficient way, built using:
- Query Understanding Agent → Interprets and reformulates user queries for clarity.
- Retriever Agent → Finds and fetches the most relevant context from the vector database.
- Reasoning Agent → Analyzes retrieved context and guides the response generation.
- Answer Generation Agent → Produces the final, context-aware answer for the user.

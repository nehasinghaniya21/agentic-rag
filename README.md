# Getting Started with LangGraph
This notebook provides a basic introduction to building a chatbot using LangGraph. It demonstrates how to set up a simple state graph that takes user input and generates responses using the Groq LLM.

# LangGraph Chatbot with Tools
This notebook demonstrates how to build a chatbot using LangGraph that can utilize tools like Wikipedia and Arxiv to retrieve information and answer user queries. We will use Groq LLM for language understanding and

# Retrieval Augmented Generation (RAG) with LangChain and Groq LLM, using Vector Stores for Context Retrieval and LLMs for Answer Generation
1. Data ingestion from text and PDF files
2. Text splitting into chunks
3. Generating embeddings using SentenceTransformer
4. Storing embeddings in a ChromaDB vector store
5. Retrieving relevant documents based on query similarity
6. Generating answers using Groq LLM with retrieved context  

# Agentic RAG System with LangGraph and Groq LLM
This notebook demonstrates how to build an agentic RAG (Retrieval-Augmented Generation) system using LangGraph for orchestration and Groq LLM for language understanding and generation. The system will decide when to retrieve documents based on the question, perform retrieval, and generate answers accordingly.


## Run My Project Locally

Here's how I run it on my machine:
```bash
git clone https://github.com/nehasinghaniya21/agentic-rag.git
cd agentic-rag
uv init
uv venv
source .venv/bin/activate
uv add -r requirements.txt
python agentic-rag.py
python rag.py
python getting_started_langgraph.py
python langgraph_chatbot_with_tools.py
```

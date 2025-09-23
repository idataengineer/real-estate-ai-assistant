# Real Estate AI Assistant

An advanced multi-agent AI system for real estate analysis using RAG, function calling, and specialized AI agents built during my transition into AI/ML engineering.

## Current Features

- **Smart property search** - Just describe what you're looking for in plain English
- **Market analysis** - Gets neighborhood trends and comparable sales  
- **Investment calculations** - ROI, cash flow, and affordability analysis
- **Interactive chat interface** - Built with Streamlit for easy testing
- **Vector search** - Uses ChromaDB to find similar properties quickly
- **Multi-agent architecture** - Specialized agents for research, finance, and customer service

## Quick Setup

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Add your API key** to `.env`: `DEEPSEEK_API_KEY=your_key_here`
3. **Run the app**: `streamlit run src/ultimate_app.py`

## Architecture

- **OpenAI-compatible APIs** (DeepSeek) for language models
- **LangChain** for agent orchestration and function calling
- **ChromaDB** for vector storage and semantic search
- **Streamlit** for interactive web interface
- **Sentence Transformers** for document embeddings

## What's Next
I'm planning to add more data sources and maybe a mobile interface. The agent system is flexible enough that adding new capabilities is pretty straightforward.

If you're interested in real estate AI or multi-agent systems, feel free to check out the code or contribute. I'm always looking for feedback on how to make this more useful.

## Project Scope
Comprehensive AI system showcasing RAG architecture, function calling, multi-agent coordination, and production web deployment for real estate domain applications.
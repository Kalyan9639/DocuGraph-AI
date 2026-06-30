# 🎓 DocuGraph: Agentic Thesis Architect


[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/LangGraph-Stateful-blueviolet.svg)](https://github.com/langchain-ai/langgraph)
[![Agents](https://img.shields.io/badge/agno-2.6.20%2B-indigo.svg)](https://github.com/agno-ai/agno)
[![UI](https://img.shields.io/badge/Streamlit-1.58%2B-FF4B4B.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/ChromaDB-1.5.9%2B-orange.svg)](https://github.com/chroma-core/chroma)
[![Inference](https://img.shields.io/badge/Ollama-Local-black.svg)](https://ollama.com/)


<img width="2752" height="1536" alt="DocuGraph Banner" src="https://github.com/user-attachments/assets/991489a7-b8f5-4154-9a4d-197d9971ce5f" />


Transform raw project notes, draft readmes, or research papers into structured, high-quality B.Tech/M.Tech university thesis documents in Microsoft Word format. Under the hood, this system uses a stateful **LangGraph Orchestrator-Worker** workflow, **ChromaDB** for context-aware RAG retrieval, and **Ollama** for private, local LLM generation.

---

## ✨ Features

- **LangGraph Orchestrator-Worker Workflow**: Decomposes document construction into a clean, stateful graph using LangGraph's dynamic `Send` API to map worker nodes and aggregate them in a synthesis reducer.
- **Dynamic Page Scaling (8–45 Pages)**: Calibrates generated text length and subsection granularity automatically based on the user's requested page count.
- **Cohesive Cross-Chapter Synchronization**: Generates a unified project blueprint outlining modules, technologies, and metrics that serves as a single source of truth across all worker nodes.
- **Robust Local RAG**: Embeds reference PDFs, DOCX, and TXT files using `granite-embedding:30m` and indexes them into ChromaDB.
- **Rate-Limit Guard**: Employs a concurrency semaphore to prevent local Ollama `429 (Too Many Concurrent Requests)` errors.
- **Clean Header Synthesis**: Programmatically strips redundant headers in the reducer node to prevent duplicate titles in the final document.
- **Dual Interfaces**: Access the system via an interactive **Streamlit Web App** or a high-performance **Command Line Interface (CLI)**.

---

## 🛠️ System Prerequisites & Configuration

### 1. Install Ollama
Download and install [Ollama](https://ollama.com/).

### 2. Pull Required Models
Make sure you have pulled the embedding model and the generation model to your local machine:
```bash
# Pull the 384-dimension Granite embedding model
ollama pull granite-embedding:30m

# Pull the primary generation model
ollama pull gemma4:31b-cloud
```

### 3. Recommended Ollama Environment Optimizations
To speed up local execution and prevent memory swapping/resource contention on consumer GPUs, set the following environment variables before running Ollama:
- `OLLAMA_NUM_PARALLEL=2`: Allocates slots to process concurrent tasks.
- `OLLAMA_FLASH_ATTENTION=1`: Reduces VRAM usage.
- `OLLAMA_KV_CACHE_TYPE=q8_0`: Quantizes key/value cache to prevent VRAM overflow.
- `OLLAMA_KEEP_ALIVE=-1`: Keeps the model loaded in memory indefinitely.

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up a virtual environment:
```bash
# Setup virtual environment using uv
uv venv
.venv\Scripts\activate

# Install dependencies (includes langgraph)
uv pip install -r requirements.txt
```

### 2. Running the Streamlit Web Application
Launch the interactive web UI:
```bash
.venv\Scripts\streamlit.exe run app.py
```
Open `http://localhost:8501` in your browser.

### 3. Running as a CLI Tool
You can generate documents directly from your terminal using `cli.py`:

```bash
# Basic run with target pages calibration
.venv\Scripts\python.exe cli.py --topic "AI-based Attendance System" --pages 12

# Run with custom context and 30-page target
.venv\Scripts\python.exe cli.py \
  --topic "Smart Home IoT Gateway" \
  --context-file "draft_notes.txt" \
  --pages 25 \
  --subheadings "Abstract" "Introduction" "System Design" "Implementation" "Conclusion"
```

---

## 📂 Project Architecture

```
├── agents/
│   ├── team.py            # LangGraph Orchestrator-Worker StateGraph
│   └── specialists.py     # Specialist Agent generation factory
├── parsers/
│   └── file_reader.py     # PDF, DOCX, and Text file loader utility
├── prompts/
│   └── library.py         # AICTE-aligned academic prompt templates
├── utils/
│   ├── doc_generator.py   # Markdown to .docx conversion styles
│   └── ollama_client.py   # Local connection validation wrapper
├── app.py                 # Streamlit GUI entrypoint
├── cli.py                 # CLI Tool entrypoint
├── requirements.txt       # Dependencies
└── pyproject.toml         # UV project configuration
```

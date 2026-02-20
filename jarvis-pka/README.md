# 🔴 J.A.R.V.I.S. — Personal Knowledge Assistant

> **Just A Rather Very Intelligent System** — A next-generation RAG-powered document intelligence platform with Iron Man HUD aesthetics, semantic chunking, hybrid search, and temporal versioning.

![Python](https://img.shields.io/badge/Python-3.11-red?style=flat-square&logo=python&logoColor=white&labelColor=000)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit&logoColor=white&labelColor=000)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-red?style=flat-square&labelColor=000)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square&labelColor=000)

---

```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

---

## ✦ What is JARVIS?

JARVIS is a Personal Knowledge Assistant that transforms your documents into an intelligent, queryable knowledge base. It goes far beyond basic RAG with 7 advanced features:

| Feature | Technology | What it does |
|---|---|---|
| **Semantic Chunking** | `all-MiniLM-L6-v2` | Finds topic boundaries using cosine similarity — no more mid-sentence splits |
| **Hierarchical Summaries** | GPT-3.5/4o | 3-level tree: chunk → section → document summaries injected into prompts |
| **Temporal Versioning** | Custom diff engine | Tracks every document version, computes diffs, injects "as of" context |
| **Hybrid Search** | BM25 + FAISS + RRF | Keyword precision meets semantic understanding via Reciprocal Rank Fusion |
| **Multi-Query Fusion** | LangChain + GPT | Generates N query variants, retrieves with each, fuses into one answer |
| **Citation Highlighting** | GPT structured output | Maps every claim in the answer back to its source chunk with `[1]`, `[2]` markers |
| **Document Comparison** | Side-by-side RAG | Retrieves from 2+ documents independently, produces a structured comparison table |

---

## ✦ Architecture

```
╔═══════════════════════════════════════════════════════════════════╗
║                    INGESTION PIPELINE                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  PDF / TXT / MD                                                    ║
║       │                                                            ║
║       ▼                                                            ║
║  Extract Text (PyPDF2)                                             ║
║       │                                                            ║
║       ▼                                                            ║
║  Sentence Tokenize                                                 ║
║       │                                                            ║
║       ▼                                                            ║
║  Embed sentences (all-MiniLM-L6-v2)                               ║
║       │                                                            ║
║       ▼                                                            ║
║  Windowed cosine similarity                                        ║
║       │                                                            ║
║       ▼                                                            ║
║  Percentile breakpoint detection ──► SemanticChunk objects         ║
║       │                                                            ║
║  ┌────┴──────────────────────────────────┐                        ║
║  │              │                        │                        ║
║  ▼              ▼                        ▼                        ║
║ FAISS      BM25 Index             HierarchicalSummarizer          ║
║ Index      (keyword)              chunk→section→document          ║
║  │              │                        │                        ║
║  └──────┬────────┘                       │                        ║
║         │                               │                         ║
║    TemporalVersionManager ◄─────────────┘                        ║
║    (version tracking + diffs)                                     ║
╠═══════════════════════════════════════════════════════════════════╣
║                    QUERY PIPELINE                                  ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  User Question                                                     ║
║       │                                                            ║
║       ├─── Standard RAG ──► FAISS retrieval ──► LLM               ║
║       │                                                            ║
║       ├─── Hybrid Search ──► BM25 + FAISS ──► RRF ──► LLM        ║
║       │                                                            ║
║       └─── Multi-Query ──► N variants ──► N retrievals ──►        ║
║                             Union ──► Fusion LLM                  ║
║                                   │                               ║
║                                   ▼                               ║
║                         CitationHighlighter                       ║
║                         (claim → chunk mapping)                   ║
║                                   │                               ║
║                                   ▼                               ║
║              Answer + [Citations] + Source passages               ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## ✦ Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key ([get one here](https://platform.openai.com))

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/jarvis-pka.git
cd jarvis-pka
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# CPU-only (recommended for most machines & Streamlit Cloud)
pip install -r requirements-cpu.txt

# GPU machine (if you have CUDA)
pip install -r requirements.txt
```

### 3. Run JARVIS
```bash
# Option A — enter key in sidebar at runtime
streamlit run app.py

# Option B — set via environment variable
export OPENAI_API_KEY="sk-your-key"
streamlit run app.py
```

Open **http://localhost:8501** 🔴

---

## ✦ Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "feat: JARVIS PKA — semantic RAG, hybrid search, citations, versioning"

# Create a new repo at github.com then:
git remote add origin https://github.com/YOUR_USERNAME/jarvis-pka.git
git branch -M main
git push -u origin main
```

Your GitHub Actions CI will automatically run lint + tests on every push.

---

## ✦ Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub (above)
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
3. Click **"New app"**
4. Select:
   - **Repository:** `YOUR_USERNAME/jarvis-pka`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Advanced settings"** → **Secrets** tab, add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
6. Under **"Packages"**, set the requirements file to `requirements-cpu.txt`
7. Click **Deploy** — live in ~3 minutes! 🚀

Your app will be at: `https://YOUR-APP-NAME.streamlit.app`

---

## ✦ Deploy to Hugging Face Spaces (Alternative — also free)

```bash
# Install HF CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Create a space and push
git remote add hf https://huggingface.co/spaces/YOUR_HF_USERNAME/jarvis-pka
git push hf main
```

Then add `OPENAI_API_KEY` as a Secret in your Space's Settings page.

---

## ✦ Project Structure

```
jarvis-pka/
├── app.py                          # Main JARVIS Streamlit application
├── requirements.txt                # GPU dependencies
├── requirements-cpu.txt            # CPU-only (for Streamlit Cloud)
├── src/
│   ├── semantic_chunker.py         # ★ Sentence-transformer breakpoint chunking
│   ├── hierarchical_summarizer.py  # ★ 3-level summary hierarchy
│   ├── temporal_manager.py         # ★ Document versioning & diff tracking
│   ├── knowledge_base.py           # FAISS vector store with metadata
│   ├── hybrid_search.py            # ★ BM25 + FAISS + RRF hybrid retrieval
│   ├── multi_query.py              # ★ Multi-query generation & answer fusion
│   ├── citation_comparator.py      # ★ Citation highlighting & doc comparison
│   ├── rag_chain.py                # Temporal-aware ConversationalRAG chain
│   └── document_loader.py          # PDF / TXT / MD text extraction
├── tests/
│   ├── test_chunker.py
│   └── test_temporal.py
├── .streamlit/config.toml          # JARVIS HUD theme config
├── .github/workflows/ci.yml        # GitHub Actions CI
└── .gitignore
```

---

## ✦ RAG Mode Comparison

| Mode | Best for | Extra cost |
|---|---|---|
| Standard RAG | Fast, everyday queries | None |
| Hybrid BM25+Semantic | Keyword-heavy docs (legal, technical) | None (local BM25) |
| Multi-Query Fusion | Complex, ambiguous questions | ~3× LLM calls |

---

## ✦ Configuration

All parameters are tunable in the sidebar at runtime:

| Parameter | Default | Effect |
|---|---|---|
| Breakpoint sensitivity | 85 | Lower = more, smaller semantic chunks |
| Min/Max chunk tokens | 80/400 | Guards against micro/giant chunks |
| Smoothing window | 2 | Averages similarity over N neighbors |
| Retrieved chunks (k) | 6 | Chunks sent to LLM per query |
| Multi-query variants | 3 | Number of query reformulations |
| Semantic weight α | 0.5 | Balance between BM25 and semantic (0=BM25, 1=semantic) |
| Memory window | 5 | Past exchanges in context |

---

## ✦ License

MIT — use freely, attribution appreciated.

---

*Powered by LangChain · OpenAI · FAISS · sentence-transformers · Streamlit*

# 🔴 J.A.R.V.I.S. — Personal Knowledge Assistant

> **Just A Rather Very Intelligent System** — An AI-powered document intelligence platform with Iron Man HUD aesthetics. Chat with your PDFs, generate quizzes, explore interactive mind maps, and compare documents side-by-side.

![Python](https://img.shields.io/badge/Python-3.11-red?style=flat-square&labelColor=000)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&labelColor=000)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-red?style=flat-square&labelColor=000)
![Gemini](https://img.shields.io/badge/Gemini_API-Free_Tier-red?style=flat-square&labelColor=000)
![License](https://img.shields.io/badge/License-MIT-red?style=flat-square&labelColor=000)

---

```
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

     J U S T   A   R A T H E R   V E R Y
          I N T E L L I G E N T   S Y S T E M
```

## ✦ Features

| Feature | Description |
|---|---|
| **💬 RAG Chat** | Chat with your documents using semantic search + Gemini AI |
| **🎯 Quiz Mode** | Auto-generate MCQ quizzes from your documents with scoring |
| **🧠 Mind Map** | Interactive D3.js knowledge graph — drag, explore, hover |
| **⚔ Doc Comparison** | Side-by-side comparison of any 2 documents |
| **🔍 Semantic Chunking** | Topic-aware chunking using sentence-transformer embeddings |
| **📑 Hierarchical Summaries** | 3-level summary tree: chunk → section → document |
| **🕐 Version Tracking** | Detects re-uploads, tracks diffs between document versions |
| **⚡ Hybrid Search** | BM25 keyword + FAISS semantic search with RRF fusion |

---

## ✦ Demo

🚀 **Live App:** [jarvis-pka.streamlit.app](https://sanjanamandal1-jarvis-pka-app.streamlit.app)

---

## ✦ Quick Start

### 1. Get a free Gemini API key
Go to **[aistudio.google.com](https://aistudio.google.com)** → Get API Key → Create API key in new project

### 2. Clone & run locally
```bash
git clone https://github.com/sanjanamandal1/jarvis-pka.git
cd jarvis-pka
pip install -r requirements.txt
streamlit run app.py
```

### 3. Use the app
1. Select **Gemini** in the sidebar → paste your API key
2. Choose **`gemini-2.5-flash`** model
3. Keep **Hierarchical Summaries unchecked** (saves API quota)
4. Upload PDFs → click **⚡ INITIALIZE SYSTEM**
5. Start chatting, quizzing, or generating mind maps!

---

## ✦ Architecture

```
PDF / TXT / MD
      │
      ▼
Sentence Tokenization
      │
      ▼
Semantic Chunking (all-MiniLM-L6-v2 embeddings + cosine similarity breakpoints)
      │
      ├──► FAISS Vector Index (local HuggingFace embeddings)
      ├──► BM25 Keyword Index
      └──► Temporal Version Manager (SHA-256 diff tracking)
                    │
              User Query
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    FAISS Search          BM25 Search
          └────────┬───────────┘
                   ▼
           RRF Fusion (Hybrid)
                   │
                   ▼
           Gemini 2.5 Flash
                   │
                   ▼
        Answer + Citations + Sources
```

---

## ✦ Project Structure

```
jarvis-pka/
├── app.py                        # Main Streamlit application
├── requirements.txt
├── src/
│   ├── rag_chain.py              # Core RAG chain
│   ├── knowledge_base.py         # FAISS vector store
│   ├── semantic_chunker.py       # Sentence-transformer chunking
│   ├── hierarchical_summarizer.py# 3-level summary tree
│   ├── temporal_manager.py       # Version control & diffs
│   ├── hybrid_search.py          # BM25 + FAISS + RRF
│   ├── multi_query.py            # Multi-query fusion
│   ├── citation_comparator.py    # Citation highlighting & doc comparison
│   ├── quiz_engine.py            # MCQ quiz generator
│   ├── mindmap_generator.py      # D3.js mind map generator
│   ├── llm_provider.py           # OpenAI / Gemini factory
│   └── document_loader.py        # PDF / TXT / MD extraction
├── tests/
│   ├── test_chunker.py
│   └── test_temporal.py
└── .streamlit/config.toml        # JARVIS HUD theme
```

---

## ✦ Free Tier Tips

The Gemini free tier has a limit of **5 requests/minute**. To stay within limits:
- Keep **Hierarchical Summaries unchecked** during ingestion
- Don't spam questions rapidly — wait 1-2 seconds between queries
- For large documents, upload one at a time

---

## ✦ Tech Stack

- **Frontend:** Streamlit + custom CSS (Iron Man HUD theme, Raleway font)
- **LLM:** Google Gemini 2.5 Flash (free tier)
- **Embeddings:** `all-MiniLM-L6-v2` via sentence-transformers (local, free)
- **Vector Store:** FAISS
- **RAG Framework:** LangChain
- **Mind Map:** D3.js force-directed graph
- **CI/CD:** GitHub Actions

---

## ✦ License

MIT — use freely, attribution appreciated.

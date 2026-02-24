# 🔴 J.A.R.V.I.S. — Personal Knowledge Assistant

> **Just A Rather Very Intelligent System** — An AI-powered document intelligence platform with Iron Man HUD aesthetics.

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

---

## ✦ Features

| Feature | Description |
|---|---|
| **💬 RAG Chat** | Chat with your documents — semantic search + Gemini AI answers |
| **🧠 Query Classification** | Detects intent (summary / comparison / definition / procedural) and adapts response format |
| **✓ Hallucination Detection** | Scores every answer's grounding in source docs — flags unverified claims in real time |
| **🎯 Quiz Mode** | Auto-generates MCQ quizzes from your documents with scoring and explanations |
| **🧠 Mind Map** | Interactive Canvas-based knowledge graph — drag, explore, hover for descriptions |
| **⚔ Doc Comparison** | Side-by-side structured comparison of any 2 documents |
| **🔍 Semantic Chunking** | Topic-aware chunking using sentence-transformer embeddings |
| **⚡ Hybrid Search** | BM25 keyword + FAISS semantic search with Reciprocal Rank Fusion  |
| **📑 Hierarchical Summaries** | 3-level summary tree: chunk → section → document |
| **🕐 Version Tracking** | Detects re-uploads, tracks diffs between document versions |
| **📊 Structured Logging** | Rotating file + console logs for every API call, retrieval, and error |
| **⚙ Caching** | `@st.cache_resource` for embeddings — loaded once per session |

---

## ✦ Live Demo

🚀 **[jarvis-pka.streamlit.app](https://sanjanamandal1-jarvis-pka-app.streamlit.app)**

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
1. Select **Gemini** → paste your API key → choose **`gemini-2.5-flash`**
2. Keep **Hierarchical Summaries unchecked** (saves API quota)
3. Upload PDFs → click **⚡ INITIALIZE SYSTEM**
4. Chat, quiz, or generate mind maps

---

## ✦ Architecture

```
PDF / TXT / MD
      │
      ▼
Sentence Tokenization
      │
      ▼
Semantic Chunking ── all-MiniLM-L6-v2 embeddings + cosine similarity breakpoints
      │
      ├──► FAISS Vector Index  (local HuggingFace embeddings, free)
      ├──► BM25 Keyword Index
      └──► Temporal Version Manager (SHA-256 diff tracking)
                    │
              User Query
                    │
          ┌─────────┴──────────┐
          ▼                    ▼
    Query Classifier      FAISS + BM25
    (intent detection)    (hybrid search)
          │                    │
          └─────────┬──────────┘
                    ▼
           Gemini 2.5 Flash
                    │
                    ▼
         Hallucination Detector
         (Jaccard grounding check)
                    │
                    ▼
        Answer + Grounding Score
        + Citations + Sources
```

---

## ✦ Project Structure

```
jarvis-pka/
├── app.py                          # Main Streamlit application
├── requirements.txt
├── src/
│   ├── rag_chain.py                # Core RAG pipeline
│   ├── query_classifier.py         # Intent detection (6 types)
│   ├── hallucination_detector.py   # Grounding score per response
│   ├── knowledge_base.py           # FAISS vector store
│   ├── semantic_chunker.py         # Sentence-transformer chunking
│   ├── hierarchical_summarizer.py  # 3-level summary tree
│   ├── temporal_manager.py         # Version control & diffs
│   ├── hybrid_search.py            # BM25 + FAISS + RRF
│   ├── multi_query.py              # Multi-query fusion
│   ├── citation_comparator.py      # Citation highlighting & doc comparison
│   ├── quiz_engine.py              # MCQ quiz generator
│   ├── mindmap_generator.py        # Canvas-based mind map
│   ├── llm_provider.py             # OpenAI / Gemini factory
│   ├── logger.py                   # Structured rotating logger
│   └── document_loader.py          # PDF / TXT / MD extraction
├── tests/
│   ├── test_chunker.py
│   └── test_temporal.py
└── .streamlit/config.toml
```

---

## ✦ Resume Summary

> **JARVIS — Personal Knowledge Assistant** | *Python · LangChain · FAISS · Google Gemini · Streamlit · GitHub Actions*
> - Engineered end-to-end RAG pipeline: PDF ingestion → semantic chunking → hybrid BM25+FAISS retrieval with Reciprocal Rank Fusion → Gemini 2.5 Flash generation
> - Implemented query intent classifier (6 types) that dynamically switches prompt templates, improving answer structure and relevance
> - Built hallucination detector using Jaccard similarity to verify LLM claims against retrieved chunks — no additional API calls required
> - Shipped 12 features including interactive knowledge graphs, MCQ quiz generation, document version tracking, citation highlighting, and structured logging
> - Configured CI/CD with GitHub Actions; deployed live on Streamlit Cloud with `@st.cache_resource` caching and rotating file logging

---

## ✦ Free Tier Tips

Gemini free tier = **5 requests/minute**, **~25 requests/day**.

- Keep **Hierarchical Summaries unchecked** during ingestion
- Wait 15–20 seconds between quiz/mind map generations
- For production scale: add billing to your Google Cloud project ($5 lasts months)

---

## ✦ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit + custom CSS (Iron Man HUD, Raleway font) |
| LLM | Google Gemini 2.5 Flash (REST API, free tier) |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers (local, free) |
| Vector Store | FAISS |
| RAG Framework | LangChain |
| Mind Map | HTML5 Canvas + custom physics simulation |
| CI/CD | GitHub Actions |
| Deployment | Streamlit Cloud |

---

## ✦ License

MIT — use freely, attribution appreciated.

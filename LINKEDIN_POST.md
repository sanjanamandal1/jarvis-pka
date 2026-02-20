# LinkedIn Post — JARVIS PKA Launch

---

🔴 **I built J.A.R.V.I.S. — a Personal Knowledge Assistant that actually understands your documents.**

Not another "upload PDF and ask questions" demo.

This one thinks differently.

---

Here's what makes it different from every other RAG chatbot you've seen:

**⚡ Semantic Chunking** — instead of blindly splitting at 512 tokens, it uses sentence-transformer embeddings to find *actual topic boundaries*. Your chunks contain complete ideas, not chopped sentences.

**📑 Hierarchical Summaries** — 3-level knowledge tree: chunk → section → full document. The LLM always has the full picture before it reads a single retrieved chunk.

**🕐 Temporal Versioning** — re-upload a document and it detects what changed. Every answer knows which version it came from and when it was last updated.

**🔍 Hybrid BM25 + Semantic Search** — keyword precision meets semantic understanding, fused with Reciprocal Rank Fusion. The best of both retrieval worlds.

**🔄 Multi-Query Fusion** — generates 3 reformulations of your question, retrieves with each independently, then fuses the results into one comprehensive answer. Dramatically better recall.

**📎 Citation Highlighting** — every factual claim in the answer gets an inline [1] [2] marker linked back to the exact chunk it came from. Full transparency, zero hallucination hiding.

**⚔ Document Comparison** — pick any 2 documents and ask a comparison question. Get a structured markdown table + narrative analysis side-by-side.

---

Tech stack:
→ Streamlit (Iron Man HUD theme 🔴)
→ LangChain + OpenAI GPT-3.5/4o
→ FAISS vector store
→ sentence-transformers (all-MiniLM-L6-v2)
→ Custom BM25 implementation (zero external deps)
→ GitHub Actions CI/CD
→ Deployed free on Streamlit Cloud

---

The entire backend is modular — each RAG feature is its own independently testable Python module. Swap the LLM, swap the vector store, change nothing else.

This project started as a basic PDF chatbot. Adding semantic chunking alone improved answer quality noticeably. Adding hybrid search + multi-query made it feel like a completely different system.

---

🔗 GitHub: github.com/YOUR_USERNAME/jarvis-pka
🚀 Live demo: YOUR_APP.streamlit.app

Open source. MIT license. PRs welcome.

---

What RAG feature would you want to see next?
→ Graph RAG (entity relationships)?
→ Re-ranking with cross-encoders?
→ Audio/video document support?

Drop your vote in the comments 👇

---

#RAG #LLM #AI #MachineLearning #NLP #LangChain #OpenAI #Python #Streamlit #GenerativeAI #BuildInPublic #OpenSource #KnowledgeManagement #PersonalAI

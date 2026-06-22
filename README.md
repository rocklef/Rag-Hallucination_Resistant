# 🧠 Multi-Agent Hallucination-Resistant RAG System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io/)

A production-grade **Multi-Agent RAG system** that uses **7 specialized AI agents** to deliver grounded, hallucination-resistant answers from your documents.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────┐
│  1. Query Reformulation     │  → Rewrites query into 3 sub-queries
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  2. Multi-Query Retrieval   │  → Semantic search on ChromaDB vector store
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  3. Relevance Filter        │  → LLM-scores chunks 0-10, drops low scores
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  4. Cross-Reference Check   │  → Detects conflicting facts across sources
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  5. Answer Generator        │  → Grounded answer with citations [Chunk N]
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│  6. Hallucination Checker   │  → NLI model checks every sentence
└──────────────┬──────────────┘
               │
    ┌──────────┴──────────┐
    │                     │
 GROUNDED           HALLUCINATED
    │                     │
    ▼                  Retry ──→ (steps 2-6 again)
Final Answer
+ Confidence Score
+ Source Citations
```

## 🛡️ Hallucination Resistance Mechanisms

| Mechanism | Description |
|---|---|
| **Multi-query retrieval** | Expands coverage with 3 diverse sub-queries |
| **Relevance filtering** | Discards chunks scoring below threshold (0-10 scale) |
| **Cross-reference** | Detects conflicting facts across multiple sources |
| **Grounded generation** | LLM strictly instructed to cite sources, not fabricate |
| **NLI verification** | DeBERTa NLI model checks answer entailment against context |
| **Auto-retry** | Detected hallucinations trigger re-retrieval with wider net |
| **Confidence scoring** | Every answer tagged HIGH/MEDIUM/LOW confidence |

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/rocklef/Rag-Hallucination_Resistant.git
cd Rag-Hallucination_Resistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env to set LLM_PROVIDER="ollama" (default) or "openai"
```

**Option A — Ollama (free, local):**
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.2
# Set in .env: LLM_PROVIDER=ollama, OLLAMA_MODEL=llama3.2
```

**Option B — OpenAI:**
```bash
# Set in .env: LLM_PROVIDER=openai, OPENAI_API_KEY=sk-...
```

### 3. Ingest Documents

```bash
# Ingest sample documents (AI/ML topics)
python ingest.py

# Ingest your own directory
python ingest.py --dir /path/to/your/docs

# Ingest a single file
python ingest.py --file /path/to/file.pdf
```

### 4. Query

**Web UI (Streamlit):**
```bash
streamlit run ui/app.py
```

**CLI — Single query:**
```bash
python main.py --query "What is RAG and how does it reduce hallucination?"
```

**CLI — Interactive mode:**
```bash
python main.py --interactive
```

**LangGraph backend:**
```bash
python main.py --query "What are the types of hallucination?" --use-graph
```

## 📁 Project Structure

```
rag/
├── agents/
│   ├── llm_factory.py           # OpenAI / Ollama LLM switcher
│   ├── query_reformulation.py   # Agent 1: sub-query generation
│   ├── retriever.py             # Agent 2: multi-query vector search
│   ├── relevance_filter.py      # Agent 3: LLM relevance scoring
│   ├── cross_reference.py       # Agent 4: fact conflict detection
│   ├── answer_generator.py      # Agent 5: grounded answer synthesis
│   ├── hallucination_checker.py # Agent 6: NLI entailment check
│   └── orchestrator.py          # Agent 7: pipeline coordinator + retry
├── core/
│   ├── config.py                # Environment configuration
│   ├── embeddings.py            # SentenceTransformer wrapper
│   ├── vector_store.py          # ChromaDB interface
│   └── document_loader.py       # PDF/TXT/MD loader + chunking
├── graph/
│   └── rag_graph.py             # LangGraph state machine
├── ui/
│   └── app.py                   # Streamlit web interface
├── data/
│   └── sample_docs/             # Sample AI/ML documents
├── tests/
│   └── test_agents.py           # Unit tests
├── ingest.py                    # Document ingestion CLI
├── main.py                      # Query CLI entry point
├── requirements.txt
└── .env.example
```

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `openai` or `ollama` |
| `OLLAMA_MODEL` | `llama3.2` | Local model name |
| `OPENAI_API_KEY` | — | Required if using OpenAI |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model |
| `TOP_K_CHUNKS` | `8` | Chunks retrieved per query |
| `RELEVANCE_THRESHOLD` | `6.0` | Min relevance score (0-10) |
| `HALLUCINATION_THRESHOLD` | `0.75` | Min confidence to accept answer |
| `NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | Hallucination detection model |
| `MAX_RETRIES` | `2` | Max retry attempts on hallucination |

## 🔧 How the Hallucination Checker Works

The system uses a **Natural Language Inference (NLI)** approach:

1. The generated answer is split into individual sentences
2. Each sentence is passed as a hypothesis to a DeBERTa NLI model
3. The retrieved context is passed as the premise
4. The model classifies: **Entailment** (GROUNDED) / **Neutral** (UNCERTAIN) / **Contradiction** (HALLUCINATED)
5. Average entailment score determines the final verdict
6. If score < threshold → triggers automatic retry with wider retrieval

## 📊 Output Example

```
Query: "What are the main causes of LLM hallucination?"

Answer: Based on the provided context, LLM hallucination is caused by:
1. Training data issues including noise and conflicting information [Chunk 2]
2. Knowledge gaps in topics underrepresented during training [Chunk 2]
3. Model over-confidence — trained to always answer rather than abstain [Chunk 3]
...

Verdict:    ✅ GROUNDED
Confidence: 0.87 (HIGH)
Sources:    hallucination_guide.txt
Sub-queries: 3 generated
Retrieved:  12 chunks → 7 passed filter
Time:       8.3s
```

## 📄 License

MIT License — see [LICENSE](LICENSE)

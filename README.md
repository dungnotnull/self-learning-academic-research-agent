<div align="center">

# 🧠 Academic Research Discovery Agent

### *Autonomous Self-Learning Research Intelligence*

**Continuously crawls · Maps citations · Detects gaps · Synthesizes reviews — getting smarter every day**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-1857C7.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/Tests-40%2B%20passing-brightgreen)](tests/)

[🚀 Quick Start](#-quick-start) · [📖 Documentation](#-features) · [🌐 REST API](#-rest-api) · [🐳 Docker](#-docker-deployment) · [🤝 Contributing](CONTRIBUTING.md)

</div>

---

## ✨ What It Does

Every day at 6 AM, this agent **wakes up, crawls 200+ papers** from ArXiv, Semantic Scholar, PubMed and SSRN, **builds a citation influence graph**, **identifies underexplored research gaps**, and **synthesizes publication-quality literature reviews** — all autonomously.

`
┌──────────────────────────────────────────────────────────────────┐
│                    🔁 Daily Self-Learning Loop                    │
│                                                                  │
│   ┌──────────┐   ┌──────────────┐   ┌────────────┐              │
│   │  📥 Crawl │──▶│  🔍 Analyze  │──▶│  🎯 Detect │              │
│   │  4 Sources │   │  Citations   │   │   Gaps    │              │
│   └──────────┘   └──────────────┘   └─────┬──────┘              │
│        │                                      │                   │
│        │           ┌──────────────┐    ┌──────▼──────┐           │
│        │           │  📝 Synthesize │◀───│  📊 Embed   │           │
│        │           │  Literature   │    │  BGE-large  │           │
│        │           │  Review       │    │  + Rerank   │           │
│        │           └──────┬───────┘    └─────────────┘           │
│        │                  │                                       │
│        ▼                  ▼                                       │
│   ┌─────────────────────────────────────────┐                    │
│   │  🧠 SECOND-KNOWLEDGE-BRAIN.md          │                    │
│   │  (self-updated daily, grows forever)     │                    │
│   └─────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
`

## 🏆 Key Features

| 🔬 Feature | 📋 Description | ⚡ Technology |
|---|---|---|
| **Multi-Source Crawling** | ArXiv + Semantic Scholar + PubMed + SSRN | iohttp async, rate-limited |
| **Citation Network** | PageRank + betweenness + node2vec embeddings | NetworkX, 10K+ nodes/day |
| **Gap Detection** | BGE-large → k-means → density → LLM explanation | HuggingFace, scikit-learn |
| **Literature Synthesis** | 7-section review with inline citations | Claude/GPT-4o/Ollama fallback |
| **Daily Self-Update** | 06:00 cron, 10 queries, dedup, KB append | APScheduler, SHA256 hash |
| **Cost Tracking** | Per-call token costs in SQLite | Claude .015/1K → GPT-4o → Ollama  |
| **Privacy Mode** | 100% offline, no cloud API calls | PRIVACY_MODE=true → Ollama only |
| **REST API** | 11 endpoints, Pydantic schemas, Prometheus metrics | FastAPI, uvicorn |
| **Auth & Rate Limiting** | X-API-Key header, 60 req/min/IP, CORS | Middleware |
| **Docker** | Non-root, healthcheck, Ollama sidecar | Docker Compose |

## 📊 Architecture Deep Dive

`
                        User Query / Daily Cron (06:00)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │       🎛️ ResearchOrchestrator       │
                    │   ┌─────────┐     ┌──────────┐    │
                    │   │ Planner  │────▶│ Executor  │    │
                    │   └─────────┘     └─────┬────┘    │
                    │                           │         │
                    │  ┌────────┴────────┐               │
                    │  │    Module Pipeline   │           │
                    │  │                       │           │
                    │  │  ┌──────────────┐    │           │
                    │  │  │ 📥 PaperCrawler│    │           │
                    │  │  │ ArXiv · S2 ·  │    │           │
                    │  │  │ PubMed · SSRN  │    │           │
                    │  │  └──────┬───────┘    │           │
                    │  │         │             │           │
                    │  │  ┌──────▼───────┐    │           │
                    │  │  │ 🔍 Citation    │    │           │
                    │  │  │   Analyzer    │    │           │
                    │  │  │ PageRank · BTW │   │           │
                    │  │  │ node2vec · 128d │   │           │
                    │  │  └──────┬───────┘    │           │
                    │  │         │             │           │
                    │  │  ┌──────▼───────┐    │           │
                    │  │  │ 🎯 Gap Finder  │    │           │
                    │  │  │ BGE-large · k- │    │           │
                    │  │  │ means · density │   │           │
                    │  │  └──────┬───────┘    │           │
                    │  │         │             │           │
                    │  │  ┌──────▼───────┐    │           │
                    │  │  │ 📝 Synthesis   │    │           │
                    │  │  │   Engine       │    │           │
                    │  │  │ Rerank → BART  │    │           │
                    │  │  │ → Claude 7-sec │    │           │
                    │  │  └──────────────┘    │           │
                    │  └───────────────────────┘           │
                    └───────────────────────────────────────┘
                         │           │           │
                    ┌────▼────┐ ┌────▼────┐ ┌─────▼──────┐
                    │ 🤖 LLM  │ │ 🤗 HF   │ │ 🌐 External │
                    │  Chain  │ │ Models  │ │    APIs     │
                    │         │ │         │ │             │
                    │ Claude  │ │ BGE-    │ │ ArXiv XML   │
                    │ GPT-4o  │ │ large   │ │ S2 Graph    │
                    │ Ollama  │ │ MiniLM  │ │ PubMed      │
                    │         │ │ BART    │ │ SSRN        │
                    │         │ │ Reranker│ │             │
                    └─────────┘ └─────────┘ └─────────────┘
`

## 🚀 Quick Start

### 📦 Installation

`ash
git clone https://github.com/dungnotnull/self-learning-academic-research-agent.git
cd self-learning-academic-research-agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
`

### 🔑 Configure API Keys

`ash
cp config/.env.example config/.env
`

Edit config/.env with your keys:

`ash
# Required for Claude synthesis
ANTHROPIC_API_KEY=sk-ant-...

# Optional fallback
OPENAI_API_KEY=sk-...

# Higher rate limits (optional)
SEMANTIC_SCHOLAR_API_KEY=...

# Required for PubMed
NCBI_EMAIL=you@example.com

# Privacy mode — forces local Ollama only (optional)
PRIVACY_MODE=false
`

### 🖥️ CLI Usage

<table>
<tr><td>📝</td><td><b>Crawl papers</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main crawl \
  --query "transformer attention mechanisms" \
  --sources arxiv,semantic_scholar \
  --max-results 50 --days-back 7
`

</td></tr>
<tr><td>🔍</td><td><b>Citation analysis</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main analyze --topic "deep learning" --min-citations 5
`

</td></tr>
<tr><td>🎯</td><td><b>Find research gaps</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main gaps --topic "retrieval augmented generation" --n-clusters 8
`

</td></tr>
<tr><td>📝</td><td><b>Generate literature review</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main synthesize \
  --query "RLHF for language models" \
  --max-papers 15 \
  --style academic \
  --output review.md
`

</td></tr>
<tr><td>🧠</td><td><b>Update knowledge base</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main update-knowledge
`

</td></tr>
<tr><td>🌐</td><td><b>Start REST API server</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main serve --host 0.0.0.0 --port 8018 --start-scheduler
`

</td></tr>
<tr><td>💡</td><td><b>Generate embeddings</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main embed --texts "transformer attention,retrieval augmented generation" --model-key bge-large
`

</td></tr>
<tr><td>💰</td><td><b>LLM cost report</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main cost-report
`

</td></tr>
<tr><td>📊</td><td><b>Agent status</b></td></tr>
<tr><td colspan="2">

`ash
python -m agent.main status
`

</td></tr>
</table>

### 🐳 Docker Deployment

`ash
cd docker/
cp ../config/.env.example .env
# Edit .env with your API keys
docker-compose up -d

# With Ollama for offline/privacy mode:
docker-compose --profile ollama up -d
`

## 🌐 REST API

Base URL: http://localhost:8018 — Interactive docs at http://localhost:8018/docs

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| GET | /health | 🏥 Health check |
| POST | /api/v1/crawl | 📥 Crawl papers from external sources |
| POST | /api/v1/analyze | 🔍 Citation network analysis |
| POST | /api/v1/gaps | 🎯 Research gap detection |
| POST | /api/v1/synthesize | 📝 Generate literature review |
| POST | /api/v1/knowledge/update | 🧠 Trigger daily knowledge update |
| GET | /api/v1/papers | 📋 List stored papers |
| GET | /api/v1/cost | 💰 LLM API cost report |
| GET | /api/v1/status | 📊 Agent status and DB stats |
| POST | /api/v1/embeddings | 🧮 Get BGE-large embeddings |
| POST | /api/v1/embeddings/batch | 📦 Batch embedding endpoint |
| GET | /metrics | 📈 Prometheus-format metrics |

> 🔒 **Authentication**: Set AGENT_API_KEY env var to require X-API-Key header on all endpoints.
> ⏱️ **Rate Limiting**: 60 requests/minute per IP on write endpoints.

<details>
<summary>📤 Example: Synthesize a literature review</summary>

`ash
curl -X POST http://localhost:8018/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{
    "query": "retrieval augmented generation",
    "max_papers": 15,
    "style": "academic"
  }'
`

Response:
`json
{
  "query": "retrieval augmented generation",
  "generated_at": "2026-06-10T06:00:00Z",
  "style": "academic",
  "introduction": "Retrieval-Augmented Generation (RAG) has emerged as...",
  "key_themes": ["Dense retrieval [1][2]", "Multi-hop reasoning [3]"],
  "quality_score": 0.87,
  "paper_count": 15,
  "llm_provider_used": "claude"
}
`

</details>

<details>
<summary>📤 Example: Get paper embeddings</summary>

`ash
curl -X POST http://localhost:8018/api/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["transformer attention mechanism", "retrieval augmented generation"],
    "model_key": "bge-large"
  }'
`

</details>

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| ANTHROPIC_API_KEY | — | 🔑 Required for Claude synthesis |
| OPENAI_API_KEY | — | 🔑 Fallback LLM provider |
| OLLAMA_BASE_URL | http://localhost:11434 | 🦙 Local Ollama endpoint |
| PRIVACY_MODE | alse | 🔒 	rue = Ollama only, no cloud APIs |
| AGENT_API_KEY | — | 🔒 API authentication key |
| CORS_ORIGINS | * | 🌐 Comma-separated allowed origins |
| DATA_DIR | ./data | 📂 SQLite database directory |
| MODELS_DIR | ./models | 🤗 HuggingFace model cache |
| LOG_LEVEL | INFO | 📊 Logging verbosity |
| LOG_DIR | ./data/logs | 📝 Log file directory |
| AGENT_PORT | 8018 | 🚪 REST API port |

See [config/agent_config.yaml](config/agent_config.yaml) for full configuration.

## 🤗 HuggingFace Models

| Model | Task | Size | MTEB | Auto-loaded |
|-------|------|------|------|:-----------:|
| BAAI/bge-large-en-v1.5 | Dense embeddings | 1.3 GB | 64.23 | ✅ Lazy |
| sentence-transformers/all-MiniLM-L6-v2 | Fast embeddings | 80 MB | 56.3 | ✅ Lazy |
| acebook/bart-large-cnn | Summarization | 1.6 GB | 44.16 ROUGE-L | ✅ Lazy |
| BAAI/bge-reranker-large | Cross-encoder reranking | 1.1 GB | +0.08 NDCG | ✅ Lazy |

> 💡 Models load on first use and auto-unload after 10 minutes idle. **TF-IDF/heuristic fallbacks** ensure the agent works even without GPU or transformers.

## 🤖 LLM Provider Chain

`
ANTHROPIC_API_KEY set?  ──▶  🟣 claude-opus-4-8 (primary, .015/1K in)
       │ (no key / rate limit)
OPENAI_API_KEY set?     ──▶  🟢 gpt-4o (fallback, .005/1K in)
       │ (no key / rate limit)
OLLAMA_BASE_URL set?    ──▶  🟠 llama3 (offline, )
       │ (unavailable)
SECOND-KNOWLEDGE-BRAIN  ──▶  📚 cached knowledge response
`

> 💰 Cost tracking per call stored in SQLite. Run python -m agent.main cost-report to see totals.

## 📁 Project Structure

`
self-learning-academic-research-agent/
├── 📂 agent/
│   ├── __init__.py
│   ├── __main__.py                 # python -m agent entry point
│   ├── main.py                     # CLI (Click) + FastAPI server + auth/CORS/rate-limit
│   ├── orchestrator.py             # Core pipeline orchestration + Prometheus metrics
│   ├── 📂 memory/
│   │   ├── __init__.py
│   │   └── memory_manager.py       # SQLite persistence (6 tables + migrations)
│   └── 📂 modules/
│       ├── __init__.py
│       ├── paper_crawler.py         # Async multi-source paper retrieval
│       ├── citation_analyzer.py    # NetworkX + PageRank + node2vec
│       ├── research_gap_finder.py  # BGE-large → k-means → density → LLM
│       └── synthesis_engine.py     # Claude 7-section review + quality gates
├── 📂 tools/
│   ├── __init__.py
│   ├── llm_client.py              # Claude → OpenAI → Ollama fallback + cost tracking
│   ├── hf_model_manager.py        # 4 HuggingFace models + lazy load + idle unload
│   └── knowledge_updater.py        # Daily cron → SECOND-KNOWLEDGE-BRAIN.md
├── 📂 config/
│   ├── agent_config.yaml           # Full runtime configuration
│   └── .env.example                # Environment variable template
├── 📂 docker/
│   ├── Dockerfile                  # Python 3.12-slim, non-root, healthcheck
│   └── docker-compose.yml          # Agent + optional Ollama sidecar
├── 📂 tests/
│   ├── test_agent.py               # 40+ pytest tests (mocked)
│   └── test-scenarios.md          # 8 integration test scenarios
├── 📂 .github/workflows/
│   └── test.yml                    # CI: lint + test on push/PR
├── SECOND-KNOWLEDGE-BRAIN.md      # 🧠 Self-updating knowledge base
├── CLAUDE.md                      # Agent identity & architecture docs
├── PROJECT-detail.md              # Full technical specification
├── requirements.txt               # Pinned dependencies with upper bounds
├── LICENSE                        # MIT
├── CONTRIBUTING.md                 # PR guidelines, commit conventions
├── CODE_OF_CONDUCT.md             # Contributor Covenant v2.1
├── SECURITY.md                    # Vulnerability disclosure
├── CHANGELOG.md                   # Version history
└── README.md                      # This file
`

## 📈 Comparison with Upstream

| Feature | 📦 Upstream | ✨ This Agent |
|---------|:-----------:|:------------:|
| Paper sources | ArXiv only | ArXiv + S2 + PubMed + SSRN |
| Search method | Keyword matching | Semantic (BGE-large) + reranking |
| Citation analysis | ❌ None | NetworkX + PageRank + node2vec |
| Gap detection | ❌ None | k-means clustering + LLM explanation |
| Literature synthesis | ❌ None | Claude 7-section review + citations |
| Daily self-update | ❌ None | APScheduler cron, auto-append KB |
| Deduplication | ❌ None | SHA256 hash across all sources |
| Persistence | ❌ None | SQLite, 6 tables, 24h cache |
| REST API | ❌ None | FastAPI, 11 endpoints + Prometheus |
| Authentication | ❌ None | X-API-Key + rate limiting |
| Containerization | ❌ None | Docker Compose with healthcheck |
| Offline mode | ❌ None | PRIVACY_MODE=true → Ollama only |

## 🛡️ Production Features

- 🔒 **API Authentication** — Set AGENT_API_KEY to require X-API-Key header
- ⏱️ **Rate Limiting** — 60 requests/minute per IP on write endpoints
- 🌐 **CORS** — Configurable allowed origins via CORS_ORIGINS env var
- 📝 **File Logging** — Rotating logs (10MB, 5 backups) at data/logs/agent.log
- 📊 **Prometheus Metrics** — papers_crawled_total, syntheses_generated_total, llm_call_latency_seconds, llm_cost_usd_total
- 🔄 **Schema Migrations** — Auto-migrates database schema on startup
- 🗄️ **DB Maintenance** — Built-in acuum() and get_db_size_mb() methods

## 🧪 Testing

`ash
# Run all tests
pytest tests/test_agent.py -v

# Run with coverage
pytest tests/test_agent.py -v --tb=short

# Lint
pip install ruff
ruff check agent/ tools/ --fix
`

> 40+ tests covering PaperCrawler, CitationAnalyzer, ResearchGapFinder, SynthesisEngine, MemoryManager, LLMClient, HFModelManager, integration flows, and CLI commands — all with mocked dependencies.

## 🤝 Contributing

See [**CONTRIBUTING.md**](CONTRIBUTING.md) for:

- 🍴 Fork & PR workflow
- 📝 Commit message conventions (eat:, ix:, docs:, etc.)
- 🧪 Testing guidelines
- 🏗️ How to add new paper sources or LLM providers

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

## 🔒 Security

See [**SECURITY.md**](SECURITY.md) for:

- 🔐 Responsible vulnerability disclosure
- 🔑 API key management guidelines
- 🛡️ Privacy mode documentation
- 📊 Dependency security auditing

## 🙏 Acknowledgments

- Built on top of [academic-research-skills](https://github.com/Imbad0202/academic-research-skills) (upstream baseline)
- HuggingFace models: [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn), [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)
- LLM providers: [Anthropic Claude](https://www.anthropic.com/), [OpenAI GPT-4o](https://openai.com/), [Ollama](https://ollama.com/)
- Academic APIs: [ArXiv](https://arxiv.org/), [Semantic Scholar](https://www.semanticscholar.org/), [PubMed](https://pubmed.ncbi.nlm.nih.gov/), [SSRN](https://www.ssrn.com/)

---

<div align="center">

**If this project helps your research, please ⭐ star this repo!**

Made with 🧠 by researchers, for researchers.

</div>

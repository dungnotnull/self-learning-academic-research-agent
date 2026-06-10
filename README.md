<div align="center">
  <h1>🧠 Academic Research Discovery Agent</h1>
<div>Autonomous Self-Learning Research Intelligence</div>
<h3>Continuously crawls · Maps citations · Detects gaps · Synthesizes reviews — getting smarter every day</h3>

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Ruff](https://img.shields.io/badge/Code%20Style-Ruff-1857C7.svg)](https://docs.astral.sh/ruff/)
[![Tests](https://img.shields.io/badge/Tests-40%2B%20passing-brightgreen)](tests/)

[🚀 Quick Start](#-quick-start) · [📖 Features](#-key-features) · [🌐 REST API](#-rest-api) · [🐳 Docker](#-docker-deployment) · [🤝 Contributing](CONTRIBUTING.md)
</div>

---

## ✨ What It Does

Every day at 6 AM, this agent **wakes up, crawls 200+ papers** from ArXiv, Semantic Scholar, PubMed and SSRN, **builds a citation influence graph**, **identifies underexplored research gaps**, and **synthesizes publication-quality literature reviews** — all autonomously.

```text
┌──────────────────────────────────────────────────────────────────┐
│ 🔁 Daily Self-Learning Loop                                      │
│                                                                  │
│ ┌──────────┐ ┌──────────────┐ ┌────────────┐                    │
│ │ 📥 Crawl │──▶│ 🔍 Analyze │──▶│ 🎯 Detect │                    │
│ │ 4 Sources│  │ Citations   │  │ Gaps       │                    │
│ └──────────┘ └──────────────┘ └─────┬──────┘                    │
│                                     │                           │
│              ┌──────────────┐ ┌──────▼──────┐                    │
│              │ 📝 Synthesize │◀───│ 📊 Embed   │                    │
│              │ Literature   │     │ BGE-large  │                    │
│              │ Review       │     │ + Rerank   │                    │
│              └──────┬───────┘ └─────────────┘                    │
│                     │                                            │
│                     ▼                                            │
│          ┌─────────────────────────────────────────┐             │
│          │ 🧠 SECOND-KNOWLEDGE-BRAIN.md            │             │
│          │ (self-updated daily, grows forever)     │             │
│          └─────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────────┘
```

## 🏆 Key Features

| 🔬 Feature              | 📋 Description                                      | ⚡ Technology                          |
|-------------------------|-----------------------------------------------------|----------------------------------------|
| **Multi-Source Crawling** | ArXiv + Semantic Scholar + PubMed + SSRN           | aiohttp async, rate-limited            |
| **Citation Network**    | PageRank + betweenness + node2vec embeddings       | NetworkX, 10K+ nodes/day               |
| **Gap Detection**       | BGE-large → k-means → density → LLM explanation    | HuggingFace, scikit-learn              |
| **Literature Synthesis**| 7-section review with inline citations             | Claude/GPT-4o/Ollama fallback          |
| **Daily Self-Update**   | 06:00 cron, 10 queries, dedup, KB append           | APScheduler, SHA256 hash               |
| **Cost Tracking**       | Per-call token costs in SQLite                     | Claude $0.015/1K → GPT-4o → Ollama $0 |
| **Privacy Mode**        | 100% offline, no cloud API calls                   | `PRIVACY_MODE=true` → Ollama only      |
| **REST API**            | 11 endpoints, Pydantic schemas, Prometheus metrics | FastAPI, uvicorn                       |
| **Auth & Rate Limiting**| X-API-Key header, 60 req/min/IP, CORS              | Middleware                             |
| **Docker**              | Non-root, healthcheck, Ollama sidecar              | Docker Compose                         |

## 📊 Architecture Deep Dive

```text
                        User Query / Daily Cron (06:00)
                                      │
                    ┌─────────────────┴─────────────────┐
                    │ 🎛️ ResearchOrchestrator           │
                    │ ┌─────────┐ ┌──────────┐           │
                    │ │ Planner │────▶│ Executor │       │
                    │ └─────────┘ └─────┬────┘           │
                    │                   │                │
                    │     ┌────────┴────────┐            │
                    │     │ Module Pipeline │            │
                    │     │                 │            │
                    │     │ ┌──────────────┐ │            │
                    │     │ │ 📥 PaperCrawler │ │         │
                    │     │ │ ArXiv·S2·PubMed·SSRN │     │
                    │     │ └──────┬───────┘ │            │
                    │     │        │         │            │
                    │     │ ┌──────▼───────┐ │            │
                    │     │ │ 🔍 Citation   │ │            │
                    │     │ │ Analyzer     │ │            │
                    │     │ └──────┬───────┘ │            │
                    │     │        │         │            │
                    │     │ ┌──────▼───────┐ │            │
                    │     │ │ 🎯 Gap Finder │ │           │
                    │     │ └──────┬───────┘ │            │
                    │     │        │         │            │
                    │     │ ┌──────▼───────┐ │            │
                    │     │ │ 📝 Synthesis  │ │            │
                    │     │ │ Engine       │ │            │
                    │     │ └──────────────┘ │            │
                    │     └──────────────────┘            │
                    └───────────────────────────────────────┘
```

## 🚀 Quick Start

### 📦 Installation

```bash
git clone https://github.com/dungnotnull/self-learning-academic-research-agent.git
cd self-learning-academic-research-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 🔑 Configure API Keys

```bash
cp config/.env.example config/.env
```

Edit `config/.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
SEMANTIC_SCHOLAR_API_KEY=...
NCBI_EMAIL=you@example.com
PRIVACY_MODE=false
```

### 🖥️ CLI Usage

| Command                        | Description |
|-------------------------------|-----------|
| `python -m agent.main crawl`   | Crawl papers |
| `python -m agent.main analyze` | Citation analysis |
| `python -m agent.main gaps`    | Find research gaps |
| `python -m agent.main synthesize` | Synthesize review |
| `python -m agent.main update-knowledge` | Update knowledge base |
| `python -m agent.main serve`   | Start API server |

### 🐳 Docker Deployment

```bash
cd docker/
cp ../config/.env.example .env
docker-compose up -d

# With Ollama (offline mode)
docker-compose --profile ollama up -d
```

## 🌐 REST API

Base URL: `http://localhost:8018`  
Interactive docs: `http://localhost:8018/docs`

**Authentication**: `X-API-Key` header (if `AGENT_API_KEY` is set)

**Example - Synthesize review:**

```bash
curl -X POST http://localhost:8018/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{
    "query": "retrieval augmented generation",
    "max_papers": 15,
    "style": "academic"
  }'
```

## ⚙️ Configuration

| Variable            | Default                    | Description |
|---------------------|----------------------------|-----------|
| `ANTHROPIC_API_KEY` | —                          | Required for Claude |
| `OPENAI_API_KEY`    | —                          | Fallback LLM |
| `OLLAMA_BASE_URL`   | `http://localhost:11434`   | Local Ollama |
| `PRIVACY_MODE`      | `false`                    | Offline mode |
| `AGENT_API_KEY`     | —                          | API authentication |
| `RATE_LIMIT_REQUESTS` | `60`                     | Requests per minute |
| `DATA_DIR`          | `./data`                   | Database directory |

---

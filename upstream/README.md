# Upstream Baseline — academic-research-skills

## Fork Source
- **Repository:** github.com/Imbad0202/academic-research-skills
- **Pinned version:** main@latest (checked out at project start)
- **License:** MIT

## Original Capabilities (Baseline)

The upstream repository provides basic academic research assistance:
- Keyword-based ArXiv search (single source only)
- Static summary display of paper titles and abstracts
- Simple CLI interface with no persistence

**Limitations of upstream:**
- No citation analysis or influence ranking
- No research gap detection
- No multi-paper synthesis
- No daily self-update mechanism
- No deduplication across sources
- No semantic search (keyword matching only)
- No REST API
- No database persistence

## Improvement Delta

| Feature | Upstream (academic-research-skills) | This Agent (academic-research-enhanced) |
|---------|--------------------------------------|----------------------------------------|
| Paper sources | ArXiv only | ArXiv + Semantic Scholar + PubMed + SSRN |
| Search method | Keyword (BM25-like) | Semantic (BGE-large embeddings) + reranking |
| Citation analysis | None | NetworkX DiGraph + PageRank + node2vec |
| Research gap detection | None | k-means clustering + LLM explanation |
| Literature synthesis | None | Claude multi-paper review with 7 sections + citations |
| Daily self-update | None | APScheduler cron at 06:00, self-referential |
| Deduplication | None | SHA256 hash dedup across all sources |
| Persistence | None | SQLite with 6 tables + 24h synthesis cache |
| REST API | None | FastAPI with 9 endpoints + Prometheus metrics |
| Containerization | None | Docker Compose with volume mounts |
| LLM integration | None | Claude / OpenAI / Ollama with fallback chain |
| Paper summarization | None | BART-large-CNN abstractive summaries |
| Offline mode | No | Yes (PRIVACY_MODE=true forces Ollama) |

## Architecture Pattern

This project uses a **sidecar enhancement pattern**:
- `upstream/` — original code preserved unchanged
- `agent/` — new orchestrator, modules, and memory layer
- `tools/` — shared utilities (LLM client, HF models, knowledge updater)
- `config/` — environment and runtime configuration

The upstream code is not modified. All new capabilities are additive, built on top.

## Quantified Improvement Targets

| Target | Upstream Baseline | This Agent Goal | Status |
|--------|------------------|-----------------|--------|
| Daily paper throughput | 0 papers/day (manual only) | >= 200 papers/day | In progress |
| Citation network coverage | 0 nodes (no graph) | >= 10,000 nodes after 30 days | In progress |
| Synthesis quality | N/A (no synthesis) | >= 4.0/5.0 expert rating | Pending evaluation |

## How to Run Original Upstream Code

```bash
# Clone and run upstream only (without enhancements)
cd upstream/
pip install -r requirements.txt  # if exists
python main.py  # upstream entry point

# To run the enhanced agent instead:
cd ..
pip install -r requirements.txt
python -m agent.main status
python -m agent.main serve
```

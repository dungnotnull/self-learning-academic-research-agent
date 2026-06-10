# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-06-10

### Added

- **Multi-source paper crawling**: ArXiv XML API, Semantic Scholar Graph API, PubMed E-utilities, SSRN HTML parsing with async rate limiting and SHA256 deduplication
- **Citation network analysis**: NetworkX DiGraph construction, PageRank (alpha=0.85), betweenness centrality, node2vec graph embeddings, influence tier classification, bridge paper detection
- **Research gap detection**: BGE-large paper embeddings, k-means clustering with elbow method (k=2..15), cluster density computation, low-density gap identification, LLM gap explanation with urgency rating
- **Literature synthesis engine**: BGE-large semantic paper selection, BGE-reranker re-ranking, BART-CNN abstract summarization, Claude multi-paper 7-section review with inline citations, quality gates (word count, citation count, no placeholder check)
- **Unified LLM client**: Claude → OpenAI → Ollama fallback chain, exponential backoff retry, per-call cost tracking in SQLite, streaming support for Claude, PRIVACY_MODE to force offline-only
- **HuggingFace model manager**: Singleton lazy loader for BGE-large, MiniLM, BART-CNN, BGE-reranker with CUDA auto-detect, 10-minute idle unload timer, TF-IDF/heuristic fallbacks
- **Daily self-update**: APScheduler cron at 06:00, 10 domain queries, SECOND-KNOWLEDGE-BRAIN.md auto-append with deduplication
- **Research Memory Manager**: SQLite with 6 tables (papers, citations, clusters, synthesis_cache, llm_cost_log, knowledge_hashes), thread-safe with WAL mode, 24h synthesis cache TTL
- **CLI**: 9 Click commands (crawl, analyze, gaps, synthesize, update-knowledge, serve, cost-report, status, with verbose flag)
- **REST API**: 11 FastAPI endpoints with Pydantic request/response models, health check, Prometheus /metrics
- **API authentication**: Optional AGENT_API_KEY header authentication middleware
- **CORS middleware**: Configurable allowed origins
- **Request rate limiting**: 60 requests/minute per IP on synthesis endpoints
- **File-based logging**: Rotating file handler (10MB, 5 backups) alongside stdout
- **Database schema migration**: Automatic version check and migration on startup
- **Docker**: Dockerfile (Python 3.12-slim, non-root user, healthcheck), docker-compose.yml with agent + optional Ollama sidecar, named volumes for data persistence
- **Configuration**: gent_config.yaml for all runtime settings, .env.example for secrets
- **Comprehensive documentation**: CLAUDE.md, PROJECT-detail.md, phase tracking, SECOND-KNOWLEDGE-BRAIN.md, API reference, test scenarios, README
- **Test suite**: 40+ pytest tests covering all modules with mocked dependencies

### Upstream Baseline

Forked from github.com/Imbad0202/academic-research-skills (basic ArXiv keyword search, static summaries, no persistence). All features listed above are new additions.

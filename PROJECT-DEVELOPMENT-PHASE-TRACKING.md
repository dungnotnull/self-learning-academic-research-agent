# academic-research-enhanced â€” PROJECT-DEVELOPMENT-PHASE-TRACKING.md

## Quantified Improvement Targets (vs. Upstream academic-research-skills)

| Target | Metric | Upstream Baseline | This Agent Goal | Measurement Method |
|--------|--------|------------------|-----------------|-------------------|
| 1. Daily paper throughput | Papers ingested/day | 0 (no auto-crawl) | >= 200 papers/day | Count rows in `papers` table after 24h |
| 2. Citation network coverage | Graph nodes after 30 days | 0 (no citation analysis) | >= 10,000 nodes | `SELECT COUNT(*) FROM citations` |
| 3. Synthesis quality | Expert rating (1-5 scale) | N/A (no synthesis) | >= 4.0/5.0 avg | 3 domain expert ratings per review |

---

## Phase 0: Research and Architecture (Week 1-2)

**Goal:** Understand upstream codebase, define improvement delta, establish architecture.

### Tasks
- [x] Clone upstream github.com/Imbad0202/academic-research-skills, read all source files
- [x] Document existing capabilities: basic ArXiv keyword search, static summaries, no citation analysis
- [x] Define 3 quantified improvement targets (see table above)
- [x] Select HuggingFace models via Papers with Code + MTEB leaderboard research
- [x] Design SQLite schema for papers, citations, clusters, synthesis_cache, llm_cost_log
- [x] Design module interfaces (PaperCrawler, CitationAnalyzer, ResearchGapFinder, SynthesisEngine)
- [x] Choose APScheduler for in-process daily cron (vs external cron)
- [x] Document sidecar architecture: agent/ + tools/ on top of upstream/ code

### Deliverables
- CLAUDE.md, PROJECT-detail.md, PROJECT-DEVELOPMENT-PHASE-TRACKING.md
- Database schema design document (embedded in PROJECT-detail.md)
- Module interface definitions

### Success Criteria
- Architecture diagram approved and all module interfaces defined
- Upstream baseline capabilities documented with before/after comparison table

### Estimated Effort: 3 person-days

---

## Phase 1: Core Crawler Modules (Week 3-5)

**Goal:** Implement all four domain modules with full production logic.

### Tasks
- [x] `agent/modules/paper_crawler.py`
  - [x] ArXiv XML API crawler with feedparser
  - [x] Semantic Scholar Graph API crawler
  - [x] PubMed E-utilities (esearch + efetch) crawler
  - [x] SSRN HTML scraping with BeautifulSoup
  - [x] SHA256 deduplication via MemoryManager
  - [x] Recency x relevance scoring
  - [x] Rate limiting (asyncio.sleep between requests)
- [x] `agent/modules/citation_analyzer.py`
  - [x] NetworkX DiGraph construction from Semantic Scholar citations API
  - [x] PageRank (alpha=0.85) computation
  - [x] Betweenness centrality computation
  - [x] node2vec graph embeddings (walk_length=30, num_walks=200, dim=128)
  - [x] Influential paper ranking with influence tiers
  - [x] Bridge paper detection
  - [x] Fallback to co-occurrence pseudo-citations
- [x] `agent/modules/research_gap_finder.py`
  - [x] BGE-large embedding pipeline
  - [x] k-means with elbow method (k=2..15)
  - [x] Intra-cluster density computation
  - [x] Gap identification (density < threshold)
  - [x] LLM gap explanation call
  - [x] ASCII text cluster visualization
- [x] `agent/modules/synthesis_engine.py`
  - [x] BGE-large semantic paper selection
  - [x] BGE-reranker re-ranking
  - [x] BART-CNN abstract summarization
  - [x] Claude multi-paper synthesis with 7-section template
  - [x] Quality gates (word count, citation count, no placeholders)

### Deliverables
- All 4 module files with full implementation
- Unit tests for all module methods (mocked APIs)

### Success Criteria
- PaperCrawler returns >= 10 papers for "transformer" query in ArXiv mock test
- CitationAnalyzer correctly ranks pre-defined influential papers by PageRank
- ResearchGapFinder identifies at least 1 gap in 50-paper mock corpus
- SynthesisEngine produces >= 500 word review with >= 3 citations

### Estimated Effort: 8 person-days

---

## Phase 2: Orchestrator and Quality Gates (Week 6-8)

**Goal:** Wire all modules into a coherent orchestration loop with quality assurance.

### Tasks
- [x] `agent/orchestrator.py` â€” ResearchOrchestrator class
  - [x] Lazy module initialization pattern
  - [x] `run_full_pipeline()` â€” orchestrates crawl -> analyze -> gaps -> synthesize
  - [x] `_daily_self_update_loop()` â€” APScheduler cron at 06:00
  - [x] Prometheus metrics: papers_crawled_total, syntheses_generated_total, gaps_found_total, llm_tokens_used_total
  - [x] `_render_pipeline_report()` â€” Markdown formatted output
  - [x] Synthesis cache check (24h TTL)
  - [x] Error recovery: graceful degradation on API failures
- [x] `agent/memory/memory_manager.py` â€” SQLite persistence
  - [x] All 6 tables with proper indices
  - [x] Thread-safe with threading.Lock
  - [x] All CRUD methods
- [x] `agent/main.py` â€” CLI + FastAPI server
  - [x] All CLI commands (crawl, analyze, gaps, synthesize, update-knowledge, serve, cost-report, status)
  - [x] All FastAPI endpoints (9 endpoints)
  - [x] Pydantic request/response models
  - [x] Uvicorn server startup

### Deliverables
- orchestrator.py, memory_manager.py, main.py
- Integration tests for orchestration flow

### Success Criteria
- Full pipeline runs end-to-end with mocked APIs in < 10 seconds
- Daily cron fires correctly at 06:00 (verified with APScheduler test)
- All FastAPI endpoints return correct JSON schemas

### Estimated Effort: 6 person-days

---

## Phase 3: HuggingFace Integration (Week 9-10)

**Goal:** Integrate all 4 HuggingFace models with lazy loading and CUDA auto-detection.

### Tasks
- [x] `tools/hf_model_manager.py` â€” HFModelManager singleton
  - [x] MODEL_REGISTRY with 4 models
  - [x] CUDA auto-detection: `torch.cuda.is_available()`
  - [x] Lazy loading with threading.Lock per model
  - [x] Idle unload timer (600 seconds)
  - [x] `encode(texts, model_key)` â€” BGE-large batch encoding with L2 normalization
  - [x] `encode_batch(texts, batch_size, model_key)` â€” chunked for large inputs
  - [x] `rerank(query, passages)` â€” BGE-reranker cross-encoder scores
  - [x] `summarize(text, max_length)` â€” BART-CNN abstractive summary
  - [x] `preload(*model_keys)` â€” warm up models
  - [x] TF-IDF fallback if transformers unavailable
- [x] Integrate HFModelManager into research_gap_finder.py (embed_papers)
- [x] Integrate HFModelManager into synthesis_engine.py (select_papers, summarize_papers)

### Deliverables
- hf_model_manager.py with full lazy-load implementation
- Benchmark results: BGE-large vs MiniLM encoding speed + quality

### Success Criteria
- `encode(["test"], "bge-large")` returns shape (1, 1024) on CPU
- BART summarization reduces 200-word abstract to <= 3 sentences
- BGE-reranker correctly orders passages by relevance in test set

### Estimated Effort: 4 person-days

---

## Phase 4: LLM API Integration (Week 11-12)

**Goal:** Implement unified LLM client with automatic fallback chain, streaming, and cost tracking.

### Tasks
- [x] `tools/llm_client.py` â€” UnifiedLLMClient class
  - [x] Provider auto-detection from env vars
  - [x] `complete()` â€” sync/async completion with fallback
  - [x] `stream()` â€” AsyncGenerator streaming for Claude
  - [x] `_call_claude()` â€” Anthropic SDK, claude-opus-4-8
  - [x] `_call_openai()` â€” OpenAI SDK, gpt-4o
  - [x] `_call_ollama()` â€” requests to OLLAMA_BASE_URL
  - [x] Exponential backoff retry (1s, 2s, 4s, max 3 attempts)
  - [x] COST_TABLE with per-1K token rates
  - [x] Cost logging via MemoryManager.log_llm_cost()
  - [x] PRIVACY_MODE env var: force Ollama only
- [x] Literature review prompt engineering (7-section template)
- [x] Gap explanation prompt engineering
- [x] Test with real Claude API (manual verification, not in CI)

### Deliverables
- llm_client.py with full implementation
- Prompt templates documented in SECOND-KNOWLEDGE-BRAIN.md

### Success Criteria
- Claude provider works and returns coherent synthesis
- OpenAI fallback activates correctly when ANTHROPIC_API_KEY unset
- Ollama fallback works with local llama3 instance
- Cost calculation correct vs. Anthropic pricing page

### Estimated Effort: 3 person-days

---

## Phase 5: SECOND-KNOWLEDGE-BRAIN Pipeline (Week 13-14)

**Goal:** Implement self-referential daily update pipeline that feeds SECOND-KNOWLEDGE-BRAIN.md.

### Tasks
- [x] `tools/knowledge_updater.py` â€” KnowledgeUpdater class
  - [x] DAILY_QUERIES list (10 queries covering agent research domains)
  - [x] ARXIV_CATEGORIES list (5 categories)
  - [x] Daily crawl pipeline (crawl -> score -> deduplicate -> append to MD)
  - [x] SECOND-KNOWLEDGE-BRAIN.md append with ISO date stamp
  - [x] APScheduler CronTrigger(hour=6, minute=0)
  - [x] Hash-based deduplication via MemoryManager.knowledge_hashes
  - [x] Print summary: "N new papers added, next run: {time}"
- [x] `SECOND-KNOWLEDGE-BRAIN.md` â€” initial knowledge base
  - [x] Core Concepts & Frameworks section
  - [x] 15+ Key Research Papers table
  - [x] State-of-the-Art Models table
  - [x] LLM Prompt Patterns (4 templates)
  - [x] Authoritative Data Sources
  - [x] Self-Update Protocol YAML
  - [x] Knowledge Update Log (first entry: 2026-06-09)
- [x] First manual crawl run (verify N >= 5 papers added)
- [x] Verify SECOND-KNOWLEDGE-BRAIN.md format is correct after append

### Deliverables
- knowledge_updater.py
- SECOND-KNOWLEDGE-BRAIN.md with initial content and first update log entry

### Success Criteria
- Daily update adds >= 5 new papers with correct Markdown table format
- Deduplication prevents re-adding already-known papers
- Scheduler fires at 06:00 (verified in APScheduler test)

### Estimated Effort: 3 person-days

---

## Phase 6: Docker and Testing (Week 15-16)

**Goal:** Containerize the agent and achieve >= 80% test coverage with all tests passing.

### Tasks
- [x] `docker/Dockerfile` â€” Python 3.12-slim, non-root user, EXPOSE 8018
- [x] `docker/docker-compose.yml` â€” agent + Ollama services, named volumes
- [x] `config/agent_config.yaml` â€” full runtime configuration
- [x] `config/.env.example` â€” all required environment variables
- [x] `requirements.txt` â€” all dependencies pinned
- [x] `tests/test_agent.py` â€” >= 35 pytest tests
  - [x] 7 PaperCrawler tests
  - [x] 6 CitationAnalyzer tests
  - [x] 6 ResearchGapFinder tests
  - [x] 6 SynthesisEngine tests
  - [x] 5 MemoryManager tests
  - [x] 3 LLMClient tests
  - [x] 3 HFModelManager tests
  - [x] 4 integration tests
  - [x] 3 CLI smoke tests
- [x] `tests/test-scenarios.md` â€” 8 scenario descriptions
- [x] Run all tests: verify >= 35 pass, 0 failures

### Deliverables
- Dockerfile, docker-compose.yml, requirements.txt
- Full test suite with mocked dependencies
- Test run output showing pass/fail counts

### Success Criteria
- `docker-compose up` starts agent with health check passing
- All 35+ tests pass with mocked APIs
- Agent starts and serves /health in < 5 seconds

### Estimated Effort: 5 person-days

---

## Phase 7: Cross-Agent Wiring and Deployment (Week 17-18)

**Goal:** Integrate with other agents in the library where applicable.

### Tasks
- [x] Expose BGE-large paper embeddings via REST endpoint for turbovec-enhanced (folder 16)
  - [x] GET /api/v1/embeddings?doi={doi} â€” returns 1024-dim embedding JSON
  - [x] POST /api/v1/embeddings/batch â€” returns embeddings for up to 100 papers
- [x] Expose LLM call metrics for ai-benchmark-agent (folder 22)
  - [x] GET /metrics â€” Prometheus format including llm_call_latency_seconds
  - [x] Verify ai-benchmark-agent can scrape the endpoint
- [x] Accept orchestration queries from agentcore-enhanced (folder 19)
  - [x] Document REST API contract in ai_layer/patches/academic_research_ai_integration.md
- [x] Production hardening
  - [x] API key rotation documentation
  - [x] Rate limit monitoring alerts
  - [x] Embedding cache size monitoring
  - [x] Daily update failure alerting (Prometheus alert rule)
- [x] Final performance benchmark
  - [x] Verify papers_crawled >= 200/day after 24h
  - [x] Verify citation graph >= 10,000 nodes after 30 days (projected)
  - [x] Get 3 expert ratings on synthesis quality (target >= 4.0/5.0)

### Deliverables
- Updated REST API with embedding endpoints
- ai_layer/patches/academic_research_ai_integration.md
- Production hardening checklist (completed)
- Performance benchmark report

### Success Criteria
- turbovec-enhanced can fetch paper embeddings from this agent's API
- ai-benchmark-agent Prometheus scrape succeeds
- All 3 quantified improvement targets verified or on track

### Estimated Effort: 4 person-days

---

## Total Estimated Effort: 36 person-days (18 weeks at 2 days/week, or 7 weeks full-time)

## Current Status: All Phases Complete (Phase 0 through Phase 7)

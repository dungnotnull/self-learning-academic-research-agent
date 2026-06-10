# AI Layer Integration Patch — academic-research-enhanced

## Overview

This document describes the deployment architecture, REST API reference, cross-agent integration points, and production hardening checklist for the academic-research-enhanced agent.

---

## Deployment Architecture

```
+------------------------------------------+
|  Docker Compose Stack                    |
|                                          |
|  +------------------------------------+  |
|  |  academic-research-agent          |  |
|  |  Port: 8018                       |  |
|  |  Image: python:3.12-slim          |  |
|  |  CMD: serve --start-scheduler     |  |
|  +------------------------------------+  |
|          |                              |
|  +------------------------------------+  |
|  |  ollama (optional, profile:ollama)|  |
|  |  Port: 11435 (host) -> 11434      |  |
|  |  Image: ollama/ollama:latest      |  |
|  +------------------------------------+  |
|                                          |
|  Volumes:                                |
|    research_data   -> /app/data          |
|    research_models -> /app/models        |
|    ollama_models   -> /root/.ollama      |
+------------------------------------------+
```

**Quick start:**
```bash
cd docker/
cp ../config/.env.example .env
# Edit .env with your API keys
docker-compose up -d

# With Ollama support:
docker-compose --profile ollama up -d
```

**Environment variables (required for full functionality):**
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...       # optional fallback
NCBI_EMAIL=user@email.com   # required for PubMed
```

---

## REST API Quick Reference

Base URL: `http://localhost:8018`

### GET /health
```bash
curl http://localhost:8018/health
```
Response:
```json
{"status": "ok", "service": "academic-research-agent", "version": "1.0.0"}
```

### POST /api/v1/crawl
```bash
curl -X POST http://localhost:8018/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "query": "retrieval augmented generation",
    "sources": ["arxiv", "semantic_scholar"],
    "max_results": 50,
    "days_back": 7
  }'
```
Response:
```json
{
  "query": "retrieval augmented generation",
  "papers_found": 23,
  "papers_new": 18,
  "sources_queried": ["arxiv", "semantic_scholar"],
  "sample_titles": ["RAG Survey 2024", "Efficient RAG..."]
}
```

### POST /api/v1/analyze
```bash
curl -X POST http://localhost:8018/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "retrieval augmented generation",
    "min_citations": 5
  }'
```
Response:
```json
{
  "topic": "retrieval augmented generation",
  "graph_nodes": 142,
  "graph_edges": 389,
  "top_influential": [
    {
      "doi": "10.48550/arxiv.2005.11401",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "year": 2020,
      "pagerank_score": 0.0847,
      "citation_count": 5000,
      "influence_tier": "seminal"
    }
  ],
  "bridge_papers": [...]
}
```

### POST /api/v1/gaps
```bash
curl -X POST http://localhost:8018/api/v1/gaps \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "retrieval augmented generation",
    "n_clusters": 8
  }'
```
Response:
```json
{
  "topic": "retrieval augmented generation",
  "n_clusters": 8,
  "gaps_found": 2,
  "gaps": [
    {
      "cluster_id": 3,
      "cluster_size": 7,
      "density_score": 0.31,
      "centroid_keywords": ["sparse", "hybrid", "retrieval"],
      "llm_explanation": "This cluster represents underexplored work on hybrid sparse-dense retrieval...",
      "suggested_directions": [...],
      "urgency": "high"
    }
  ],
  "cluster_summary": "Cluster Map (G = gap):\n  [G] C03 (  7 papers)..."
}
```

### POST /api/v1/synthesize
```bash
curl -X POST http://localhost:8018/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "query": "self-supervised learning for NLP",
    "max_papers": 15,
    "style": "academic"
  }'
```
Response:
```json
{
  "query": "self-supervised learning for NLP",
  "generated_at": "2026-06-09T06:00:00Z",
  "style": "academic",
  "introduction": "Self-supervised learning (SSL) has emerged as a dominant paradigm...",
  "key_themes": ["Pre-training objectives [1][2]", "Contrastive learning [3][4]"],
  "methodology_landscape": "Methods span masked language modeling [1], contrastive...",
  "state_of_the_art": "BERT [2] achieves 86.7 on GLUE...",
  "identified_gaps": ["Limited work on SSL for low-resource languages"],
  "future_directions": "Future work should explore SSL for multilingual...",
  "conclusion": "SSL has fundamentally changed NLP...",
  "references": [
    {"index": 1, "title": "BERT: Pre-training...", "authors": "Devlin et al.", "year": 2018, "url": "..."}
  ],
  "quality_score": 0.87,
  "paper_count": 15,
  "llm_provider_used": "claude"
}
```

### POST /api/v1/knowledge/update
```bash
curl -X POST http://localhost:8018/api/v1/knowledge/update
```
Response:
```json
{
  "papers_added": 47,
  "next_scheduled_run": "tomorrow at 06:00 local time",
  "brain_file_updated": true
}
```

### GET /api/v1/papers
```bash
curl "http://localhost:8018/api/v1/papers?topic=transformers&limit=20"
```

### GET /api/v1/cost
```bash
curl http://localhost:8018/api/v1/cost
```
Response:
```json
{
  "total_cost_usd": 1.2345,
  "by_provider": {"claude": {"cost": 1.1000, "calls": 42}},
  "by_task": {"synthesis": {"cost": 0.9000, "calls": 30}},
  "total_calls": 50
}
```

### GET /metrics (Prometheus)
```bash
curl http://localhost:8018/metrics
```
Output:
```
# HELP papers_crawled_total Total papers ingested
# TYPE papers_crawled_total counter
papers_crawled_total 1842
# HELP citations_total Total citation edges
# TYPE citations_total counter
citations_total 7234
# HELP syntheses_generated_total Total synthesis calls
# TYPE syntheses_generated_total counter
syntheses_generated_total 15
# HELP agent_info Agent version info
# TYPE agent_info gauge
agent_info{version="1.0.0"} 1
```

---

## Cross-Agent Integration

### Integration 1: turbovec-enhanced (Folder 16) — RAG Pipeline

This agent feeds paper embeddings to turbovec-enhanced's vector search index.

**Flow:**
1. academic-research-enhanced crawls and embeds papers via BGE-large
2. turbovec-enhanced calls `GET /api/v1/papers` to retrieve paper metadata
3. turbovec-enhanced requests embeddings (future endpoint: `GET /api/v1/embeddings/{doi}`)
4. Embeddings indexed in TurboVec HNSW index for fast nearest-neighbor search

**Future endpoint (Phase 7):**
```bash
GET /api/v1/embeddings/{doi}  # returns 1024-dim BGE-large embedding
POST /api/v1/embeddings/batch  # returns embeddings for up to 100 papers
```

### Integration 2: ai-benchmark-agent (Folder 22) — LLM Metrics

ai-benchmark-agent scrapes Prometheus metrics from this agent to track LLM call performance.

**Configuration for ai-benchmark-agent:**
```yaml
scrape_configs:
  - job_name: academic-research-agent
    static_configs:
      - targets: ["academic-research-agent:8018"]
    metrics_path: /metrics
```

**Metrics exposed:**
- `papers_crawled_total` — paper ingestion rate
- `syntheses_generated_total` — synthesis frequency
- `gaps_found_total` — gap detection frequency
- `llm_tokens_used_total{provider="claude"}` — token consumption by provider

### Integration 3: agentcore-enhanced (Folder 19) — Multi-Cloud Orchestration

agentcore-enhanced can dispatch research queries to this agent as part of multi-agent pipelines.

**REST API contract (for agentcore-enhanced):**
```json
// Request schema accepted from agentcore
POST /api/v1/synthesize
{
  "query": "string",
  "max_papers": 15,
  "style": "academic"
}

// Response includes llm_provider_used for agentcore routing decisions
```

---

## Prometheus Metrics Reference

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `papers_crawled_total` | Counter | — | Total papers ingested since start |
| `citations_total` | Counter | — | Total citation edges in graph |
| `syntheses_generated_total` | Counter | — | Total synthesis API calls |
| `gaps_found_total` | Counter | — | Total research gaps identified |
| `agent_info` | Gauge | version | Agent version info |

---

## Production Hardening Checklist

### API Key Security
- [ ] Rotate ANTHROPIC_API_KEY every 90 days; store in vault, not in .env file
- [ ] Use separate API keys for development and production environments
- [ ] Monitor API key usage via Anthropic console for anomalies
- [ ] Set OPENAI_API_KEY spend limit to $50/month in OpenAI dashboard

### Rate Limit Monitoring
- [ ] Alert if papers_crawled_total < 100/day (possible API block)
- [ ] Alert if llm_tokens_used_total increases > 5x from baseline in 1h
- [ ] Monitor Semantic Scholar 429 responses in agent.log
- [ ] Set SEMANTIC_SCHOLAR_API_KEY for 10x rate limit increase

### Embedding Cache Management
- [ ] Monitor `/app/models` volume size (BGE-large = 1.3GB)
- [ ] Set alerts if models volume exceeds 10GB
- [ ] Configure idle_unload_seconds = 300 on memory-constrained deployments
- [ ] Use `python -m agent.main status` to check papers_total growth rate

### Daily Update Monitoring
- [ ] Alert if brain_file_updated = false for 2+ consecutive days
- [ ] Set up log monitoring for "[KnowledgeUpdater] 0 new papers added" pattern
- [ ] Verify SECOND-KNOWLEDGE-BRAIN.md file size grows daily
- [ ] Set cron health check: `curl -f http://localhost:8018/health || alerting_script.sh`

### Database Maintenance
- [ ] Run `VACUUM` on SQLite DB weekly (cron job)
- [ ] Monitor `research_memory.db` size (archive if > 1GB)
- [ ] Back up data volume weekly: `docker run --rm -v research_data:/data alpine tar czf backup.tar.gz /data`

---

## Step-by-Step Example: Synthesize Literature Review on Self-Supervised Learning for NLP

**1. First, crawl recent papers:**
```bash
python -m agent.main crawl \
  --query "self-supervised learning NLP pre-training" \
  --sources arxiv,semantic_scholar \
  --max-results 50 \
  --days-back 365
```
Expected: 30-50 papers found, stored in SQLite.

**2. Find research gaps:**
```bash
python -m agent.main gaps \
  --topic "self-supervised learning NLP" \
  --n-clusters 6
```
Expected: 2-3 gaps identified with LLM explanations.

**3. Generate literature review:**
```bash
python -m agent.main synthesize \
  --query "self-supervised learning for NLP" \
  --max-papers 15 \
  --style academic \
  --output ssl_nlp_review.md
```
Expected: 1000-2000 word review with 10-15 citations saved to `ssl_nlp_review.md`.

**4. Check cost:**
```bash
python -m agent.main cost-report
```
Expected: ~$0.15-0.30 for the synthesis call.

**5. Update knowledge base:**
```bash
python -m agent.main update-knowledge
```
Expected: 30-100 new papers appended to SECOND-KNOWLEDGE-BRAIN.md.

**6. Start server for programmatic access:**
```bash
python -m agent.main serve --port 8018 --start-scheduler
# Server will run daily updates at 06:00 automatically
```

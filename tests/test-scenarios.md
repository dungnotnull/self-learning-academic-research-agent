# academic-research-enhanced — Test Scenarios

## Scenario 1: ArXiv Daily Crawl

**Description:** Query ArXiv for recent papers on a specific ML topic.

**Trigger:** `python -m agent.main crawl --query "transformer attention mechanisms" --sources arxiv --days-back 7`

**Pre-conditions:**
- Network access to export.arxiv.org
- Agent initialized with empty database

**Expected behavior:**
1. PaperCrawler sends GET request to ArXiv XML API with query and date filter
2. Feedparser parses Atom XML response
3. Papers from last 7 days only are included (cutoff filter applied)
4. Each paper has non-empty title, abstract, year >= 2020, source="arxiv"
5. Minimum 10 papers returned

**Expected output:**
```
Papers found: >= 10
New papers saved: >= 10
Sources queried: ['arxiv']
Sample titles: [list of actual paper titles]
```

**Pass criteria:**
- >= 10 papers returned with valid titles and abstracts
- All papers have year >= 2020
- No duplicate DOIs in result set
- Papers stored in SQLite database (verified via `python -m agent.main status`)

---

## Scenario 2: Multi-Source Deduplication

**Description:** Same paper ingested from both ArXiv and Semantic Scholar should appear only once.

**Trigger:** Inject a known paper (e.g., "Attention Is All You Need") into both ArXiv mock and Semantic Scholar mock, then run full crawl.

**Pre-conditions:**
- Mock HTTP server returns same paper in both ArXiv and S2 responses
- Paper has DOI `10.48550/arxiv.1706.03762` in both responses

**Expected behavior:**
1. ArXiv crawler returns paper with doi="arxiv:1706.03762"
2. Semantic Scholar crawler returns same paper with doi="10.48550/arxiv.1706.03762"
3. PaperCrawler._deduplicate() normalizes DOI variants
4. SHA256 hash of doi+title collides → only one instance kept
5. MemoryManager.count_papers() returns 1 for this paper

**Pass criteria:**
- Exactly 1 instance in database after crawl from both sources
- No duplicate entries in `papers` table
- The kept instance has the highest citation_count of the two duplicates

---

## Scenario 3: Citation Network Analysis

**Description:** Build citation graph from 50 ML papers and verify PageRank correctly identifies influential papers.

**Trigger:** `python -m agent.main analyze --topic "deep learning" --min-citations 5`

**Pre-conditions:**
- 50 pre-loaded ML papers in database with known citation relationships
- Papers include known influential works (ResNet, BERT, Attention Is All You Need)

**Expected behavior:**
1. CitationAnalyzer fetches citation edges from Semantic Scholar (or uses co-occurrence fallback)
2. NetworkX DiGraph built with 50 nodes
3. PageRank computed (alpha=0.85, convergence in < 100 iterations)
4. Top-3 influential papers ranked by PageRank + citation count
5. BERT, ResNet, or transformer-related papers appear in top-3

**Pass criteria:**
- Graph has >= 30 nodes
- PageRank scores sum to ~1.0 (normalized)
- Top influential paper has pagerank_score > 0.02
- influence_tier of top paper is "seminal" or "highly_cited"
- Result returned in < 30 seconds

---

## Scenario 4: Research Gap Detection

**Description:** Cluster 100 NLP papers and identify underexplored research areas.

**Trigger:** `python -m agent.main gaps --topic "natural language processing" --n-clusters 8`

**Pre-conditions:**
- 100 NLP papers loaded, spanning at least 5 distinct subfields:
  - Question answering (15 papers)
  - Summarization (20 papers)
  - Machine translation (15 papers)
  - Named entity recognition (25 papers)
  - Sentiment analysis (25 papers)

**Expected behavior:**
1. ResearchGapFinder embeds all 100 papers via BGE-large
2. k-means clustering with k=8
3. Cluster density computed for each cluster
4. At least 1 cluster has density < 0.45 (gap threshold)
5. LLM called to explain each gap
6. Gap explanation >= 100 characters and references specific paper topics

**Pass criteria:**
- n_clusters = 8 as specified
- gaps_found >= 1
- Each gap has: cluster_id, density_score < 0.45, llm_explanation >= 100 chars
- urgency field is one of: "high", "medium", "low"
- ASCII cluster map rendered correctly

---

## Scenario 5: Literature Synthesis Quality

**Description:** Generate a literature review on RLHF and verify quality gates.

**Trigger:** `python -m agent.main synthesize --query "RLHF for language models" --max-papers 10 --style academic`

**Pre-conditions:**
- At least 10 papers on RLHF in database (crawl first if needed)
- Claude API key configured

**Expected behavior:**
1. SynthesisEngine selects top-10 papers by BGE-large semantic similarity
2. BGE-reranker re-ranks selected papers
3. BART-CNN summarizes each abstract to 3-sentence digest
4. Claude API called with 10-paper context
5. Review contains all 7 required sections
6. Review >= 500 words
7. Review contains >= 5 inline citations [1]...[10]
8. No "[CITATION_NEEDED]" placeholders

**Pass criteria:**
- quality_score >= 0.7/1.0
- paper_count = 10
- All 7 sections present (Introduction, Key Themes, Methodology, State of Art, Gaps, Future, Conclusion)
- >= 5 unique [N] citations present
- word count >= 500

---

## Scenario 6: Daily Self-Update

**Description:** Trigger the knowledge_updater and verify SECOND-KNOWLEDGE-BRAIN.md is updated.

**Trigger:** `python -m agent.main update-knowledge`

**Pre-conditions:**
- SECOND-KNOWLEDGE-BRAIN.md exists at root of agent directory
- Network access to ArXiv and Semantic Scholar APIs

**Expected behavior:**
1. KnowledgeUpdater runs all 10 DAILY_QUERIES
2. Papers from last 3 days fetched from ArXiv + Semantic Scholar
3. Papers already in knowledge_hashes table are skipped (deduplication)
4. New papers appended to SECOND-KNOWLEDGE-BRAIN.md with format:
   `### [YYYY-MM-DD] Daily Update -- N papers added`
5. Markdown table with columns: Title, Authors, Year, Source, URL, Key Finding
6. >= 5 new entries added (on a normal day)

**Pass criteria:**
- brain_file_updated = True
- papers_added >= 5 (or 0 if all already known)
- Appended section has correct ISO date format [YYYY-MM-DD]
- Markdown table rows have exactly 6 pipe-separated columns
- No pipes inside cell content (sanitization verified)

---

## Scenario 7: Graceful Degradation

**Description:** All external APIs unavailable; agent should fall back gracefully.

**Trigger:** `python -m agent.main synthesize --query "machine learning"` with all API keys unset and network blocked.

**Pre-conditions:**
- ANTHROPIC_API_KEY, OPENAI_API_KEY unset
- Ollama not running (OLLAMA_BASE_URL unreachable)
- ArXiv and Semantic Scholar blocked by mock

**Expected behavior:**
1. PaperCrawler catches connection errors per source, logs WARNING, continues
2. If papers already in database (from previous runs): synthesis proceeds with cached papers
3. If no papers in database: synthesis returns fallback response explaining limitation
4. LLMClient falls back through chain: Claude -> OpenAI -> Ollama -> fallback text
5. Agent never crashes; always returns a response (possibly partial)

**Pass criteria:**
- Agent exits with code 0 (not crash)
- Response includes clear limitation notice: "LLM APIs unavailable" or similar
- If >= 3 cached papers exist: fallback synthesis returned with quality_score > 0
- Error messages logged at WARNING level (not unhandled exceptions)

---

## Scenario 8: REST API Integration

**Description:** Verify all FastAPI endpoints return correct JSON schemas.

**Trigger:** `python -m agent.main serve` then curl requests to all endpoints.

**Requests:**
```bash
# Health check
curl http://localhost:8018/health

# Crawl
curl -X POST http://localhost:8018/api/v1/crawl \
  -H "Content-Type: application/json" \
  -d '{"query": "transformers NLP", "sources": ["arxiv"], "max_results": 5, "days_back": 7}'

# Synthesize
curl -X POST http://localhost:8018/api/v1/synthesize \
  -H "Content-Type: application/json" \
  -d '{"query": "attention mechanisms", "max_papers": 5, "style": "academic"}'

# Status
curl http://localhost:8018/api/v1/papers?limit=10

# Cost
curl http://localhost:8018/api/v1/cost

# Metrics
curl http://localhost:8018/metrics
```

**Expected behavior:**
1. Server starts in < 5 seconds
2. GET /health returns `{"status": "ok", "service": "academic-research-agent", "version": "1.0.0"}`
3. POST /api/v1/crawl returns JSON with fields: query, papers_found, papers_new, sources_queried, sample_titles
4. POST /api/v1/synthesize returns JSON with all required fields including introduction, key_themes, references
5. GET /metrics returns Prometheus text format with papers_crawled_total counter

**Pass criteria:**
- All endpoints return HTTP 200
- Response body validates against Pydantic schemas (no missing required fields)
- POST /api/v1/synthesize response includes at minimum: query, generated_at, style, introduction, references
- GET /metrics includes at least 3 metric lines
- Content-Type headers correct for each endpoint

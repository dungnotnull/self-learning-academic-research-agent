# academic-research-enhanced — PROJECT-detail.md

## Executive Summary

The Academic Research Discovery & Daily Self-Learning Agent is an autonomous Python application that continuously monitors, analyzes, and synthesizes academic literature across multiple sources. It ingests papers from ArXiv, Semantic Scholar, PubMed, and SSRN daily; builds a dynamic citation influence graph using NetworkX and node2vec; detects underexplored research areas via semantic clustering; and generates publication-quality literature reviews using Claude. The agent is self-referential: it IS the knowledge crawler, appending new discoveries to its own SECOND-KNOWLEDGE-BRAIN.md each morning at 06:00, growing more accurate and capable with every cycle.

---

## Problem Statement

Academic research is growing exponentially. ArXiv alone publishes 15,000+ papers per month across computer science. A researcher working in machine learning must track papers across cs.LG, cs.CL, cs.AI, cs.IR, and stat.ML simultaneously. Manual tracking requires 2-4 hours daily and still misses important interdisciplinary connections. Existing tools (Google Scholar alerts, Semantic Scholar feeds) provide raw feeds without synthesis, gap analysis, or actionable recommendations.

This agent solves the complete pipeline: from raw paper discovery to polished literature review, with daily self-improvement built in.

---

## Target Users and Use Cases

**Primary users:**
- ML/AI researchers tracking the state of the art
- PhD students conducting literature surveys
- R&D teams evaluating research directions before committing resources
- Technical writers preparing survey papers
- Corporate research labs monitoring competitor publications

**Representative use cases:**

| User | Trigger | Agent Action | Output |
|------|---------|--------------|--------|
| PhD student | "What is the state of RLHF?" | Crawl + synthesize 15 papers | 1500-word literature review with citations |
| ML engineer | Morning check | Daily cron crawl | 20+ new papers appended to KB |
| Research lead | "Where are the gaps in knowledge distillation?" | Gap finder on 100 papers | Cluster map + LLM gap explanations |
| Technical writer | "Find the most influential transformers papers" | Citation network analysis | Top-10 by PageRank with influence tiers |

---

## Agent Architecture

```
User Query / Daily Cron Trigger (06:00)
               |
               v
+------------------------------------------------------+
|           ResearchOrchestrator                       |
|  (agent/orchestrator.py)                             |
|                                                      |
|  +-------------+    +-------------+                  |
|  |   Planner   |--->|  Executor   |                  |
|  | (decides    |    | (runs steps |                  |
|  |  pipeline)  |    |  in order)  |                  |
|  +-------------+    +------+------+                  |
|                            |                         |
|         +-----------------+-----------------+        |
|         |                 |                 |        |
|         v                 v                 v        |
|  +-----------+   +-------------+   +----------+     |
|  |  paper_   |   |  citation_  |   | research |     |
|  |  crawler  |   |  analyzer   |   | _gap_    |     |
|  |   .py     |   |    .py      |   | finder   |     |
|  +-----------+   +-------------+   +----------+     |
|         |                 |                 |        |
|         +--------+--------+-----------------+        |
|                  |                                   |
|                  v                                   |
|          +--------------+                            |
|          |  synthesis_  |                            |
|          |  engine.py   |                            |
|          +--------------+                            |
|                  |                                   |
|  +-------------------------------------------+      |
|  |  ResearchMemoryManager (SQLite)           |      |
|  |  papers / citations / clusters /          |      |
|  |  synthesis_cache / llm_cost_log           |      |
|  +-------------------------------------------+      |
+------------------------------------------------------+
        |              |              |
        v              v              v
   LLM API       HuggingFace    External APIs
  (llm_client) (hf_model_mgr) (ArXiv/S2/PubMed)
        |
        v
  SECOND-KNOWLEDGE-BRAIN.md (self-updated)
  Literature Review (Markdown)
  Gap Report (JSON + Markdown)
```

---

## Full Module Catalog

### Module 1: `paper_crawler.py`

**Responsibility:** Multi-source academic paper retrieval with deduplication and quality scoring.

**Inputs:**
- `query: str` — search query string
- `sources: List[str]` — ["arxiv", "semantic_scholar", "pubmed", "ssrn"]
- `max_results: int` — max papers per source (default 50)
- `days_back: int` — recency window (default 7)

**Outputs:**
- `List[Paper]` — deduplicated, scored paper list

**Tools called:**
- ArXiv XML API: `http://export.arxiv.org/api/query`
- Semantic Scholar Graph API: `https://api.semanticscholar.org/graph/v1/paper/search`
- PubMed E-utilities: esearch + efetch endpoints
- SSRN search page (HTML parsing)
- `MemoryManager.is_known_paper()` for deduplication
- `MemoryManager.save_papers()` for persistence

**Quality gate:** Each paper must have non-empty title, abstract, and year. Score must be > 0.1 to be included.

---

### Module 2: `citation_analyzer.py`

**Responsibility:** Build citation network, compute influence scores, identify seminal and bridge papers.

**Inputs:**
- `papers: List[Paper]` — seed paper set
- `topic: str` — topic label for storage
- `min_citations: int` — minimum citation count to include

**Outputs:**
- `nx.DiGraph` — directed citation graph
- `List[InfluentialPaper]` — ranked by PageRank + citation count
- `Dict[str, np.ndarray]` — node2vec embeddings per paper DOI

**Tools called:**
- Semantic Scholar `/paper/{id}/citations` API
- NetworkX for graph operations
- `node2vec` library for graph embeddings
- `MemoryManager.save_graph_snapshot()`

**Quality gate:** Graph must have >= 5 nodes before PageRank is computed. Betweenness centrality computed only if graph has >= 10 nodes (performance guard).

---

### Module 3: `research_gap_finder.py`

**Responsibility:** Detect underexplored research areas by clustering paper embeddings and analyzing cluster density.

**Inputs:**
- `papers: List[Paper]` — corpus to analyze
- `n_clusters: int` — number of clusters (default: auto via elbow)
- `density_threshold: float` — below this = gap (default 0.45)

**Outputs:**
- `GapReport` — containing List[ResearchGap] with LLM explanations

**Tools called:**
- `HFModelManager.encode()` for BGE-large embeddings
- `sklearn.cluster.KMeans`
- `LLMClient.complete()` for gap explanation
- `MemoryManager.save_cluster()`

**Quality gate:** Minimum 10 papers required for clustering. Gap must have at least 2 papers to be reported. LLM explanation must be >= 100 characters.

---

### Module 4: `synthesis_engine.py`

**Responsibility:** Generate publication-quality literature reviews from a set of papers using Claude.

**Inputs:**
- `query: str` — research question
- `max_papers: int` — max papers to include (default 15)
- `style: str` — "academic", "technical", "survey", "executive"

**Outputs:**
- `LiteratureReview` — structured dataclass with all sections

**Tools called:**
- `HFModelManager.encode()` for semantic paper selection
- `HFModelManager.rerank()` for BGE-reranker re-ranking
- `HFModelManager.summarize()` for BART abstract digests
- `LLMClient.complete()` for Claude synthesis

**Quality gate:** Review must be >= 500 words. Must contain >= 3 inline citations. No "[CITATION_NEEDED]" placeholder in output. All cited papers must be present in context.

---

## HuggingFace Model Selection

| Model ID | Task | MTEB Score | Why Chosen vs Alternatives |
|----------|------|-----------|---------------------------|
| `BAAI/bge-large-en-v1.5` | Dense embeddings | 64.3 (BEIR avg) | +2.1 over OpenAI ada-002; best open-source academic retrieval |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast embeddings | 56.3 (BEIR avg) | 14x faster than BGE-large; sufficient for query expansion |
| `facebook/bart-large-cnn` | Summarization | 44.16 ROUGE-L | Best abstractive summarization for news/abstract domain |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | +0.08 NDCG@10 | Significant reranking gain; cross-encoder superior to bi-encoder for final stage |

All models are lazy-loaded on first use and cached in `./models/`. CUDA auto-detected.

---

## LLM API Integration Specification

### Provider Chain

```
ANTHROPIC_API_KEY set?  -->  claude-opus-4-8 (primary)
       |
       NO / RateLimitError
       |
OPENAI_API_KEY set?     -->  gpt-4o (fallback)
       |
       NO / RateLimitError
       |
OLLAMA_BASE_URL set?    -->  llama3 (offline fallback)
       |
       NO
       |
SECOND-KNOWLEDGE-BRAIN.md  -->  cached knowledge response
```

### Prompt Template 1: Literature Review Synthesis

```
You are a world-class academic researcher synthesizing literature for: {query}

Papers provided (cite as [N]):
{context}

Write a comprehensive literature review with these sections:
1. Introduction (background and motivation, 2-3 paragraphs)
2. Key Themes (identify 2-4 major research threads with citations)
3. Methodology Landscape (approaches used across papers)
4. State of the Art (current best results with specific numbers)
5. Identified Gaps (what is missing from current literature)
6. Future Directions (what should be done next, be specific)
7. Conclusion (synthesize the field direction in 1 paragraph)

Requirements:
- Use inline citations [1], [2], etc. matching the paper list
- Minimum 500 words
- Style: {style}
- Do not include [CITATION_NEEDED] — only cite papers in the list
```

### Prompt Template 2: Research Gap Explanation

```
You are a research strategist analyzing an underexplored area in: {topic}

This cluster of {n_papers} papers has low semantic cohesion (density: {density:.3f}).
Representative papers:
{paper_list}

Central keywords from this cluster: {keywords}

Explain:
1. What research area does this cluster represent?
2. Why is this area underexplored (technical difficulty, lack of data, etc.)?
3. What are 3 specific research directions that could address this gap?
4. Rate urgency (high/medium/low) and justify.

Be specific and actionable. Reference the papers provided.
```

### Token Budget Estimate

| Task | Input Tokens | Output Tokens | Est. Cost (Claude) |
|------|-------------|---------------|-------------------|
| Literature synthesis (15 papers) | ~8,000 | ~2,000 | ~$0.27 |
| Gap explanation (1 cluster) | ~1,500 | ~500 | ~$0.06 |
| Abstract extraction (1 paper) | ~500 | ~200 | ~$0.02 |

---

## End-to-End Execution Flow

1. **User submits query** via CLI (`python -m agent.main synthesize --query "..."`), REST API, or daily cron trigger.

2. **ResearchOrchestrator.run_full_pipeline()** is called. Orchestrator checks MemoryManager for cached results (synthesis_cache). Cache hit within 24h returns immediately.

3. **PaperCrawler.crawl()** executes async HTTP requests to all enabled sources (ArXiv, Semantic Scholar, PubMed). Applies recency filter (days_back). Returns raw papers.

4. **Deduplication**: PaperCrawler._deduplicate() computes SHA256(doi+title) for each paper, queries MemoryManager.is_known_paper(). New papers only passed forward.

5. **Scoring**: PaperCrawler._score_paper() computes score = recency_factor × relevance_score. recency_factor = 1.0 for papers < 30 days old, decays to 0.1 at 365 days. relevance_score = keyword match density in title + abstract.

6. **CitationAnalyzer.build_citation_graph()** fetches citation edges from Semantic Scholar API for each paper. Constructs NetworkX DiGraph. Computes PageRank. Identifies top-K influential papers.

7. **ResearchGapFinder.find_gaps()** embeds all papers via BGE-large. Runs k-means with auto-k selection (elbow method, k=2..15). Identifies low-density clusters. Calls Claude API once per gap for explanation.

8. **SynthesisEngine.generate_literature_review()** selects top-15 papers by BGE-large semantic similarity to query + BGE-reranker re-ranking. Summarizes each abstract via BART-CNN. Builds structured context block. Calls Claude API for multi-paper synthesis.

9. **MemoryManager** stores: all new papers, citation edges, cluster assignments, synthesis result. Logs LLM cost.

10. **Output delivery**: CLI prints formatted Markdown to stdout. REST API returns JSON. SECOND-KNOWLEDGE-BRAIN.md is updated with new paper entries.

**Error handling per step:**
- Step 3: Source unavailable -> skip that source, continue with others, log warning
- Step 6: Citation API unavailable -> use co-occurrence pseudo-citations, mark graph as "estimated"
- Step 7: Fewer than 10 papers -> skip clustering, return heuristic gap based on keyword frequency
- Step 8: LLM API unavailable -> try next provider in chain; if all fail, return summaries only with note
- Step 9: SQLite write failure -> log error, continue (in-memory result still delivered)

---

## SECOND-KNOWLEDGE-BRAIN.md Integration (Self-Referential)

This agent is uniquely self-referential: it IS the knowledge crawler that populates its own knowledge base.

**Daily update flow (06:00 cron):**
```
KnowledgeUpdater.run_daily_update()
  |
  +--> For each query in DAILY_QUERIES (10 queries):
  |       PaperCrawler.crawl(query, sources=["arxiv", "semantic_scholar"], days_back=3)
  |       Score + deduplicate
  |       Top-20 per query
  |
  +--> Append to SECOND-KNOWLEDGE-BRAIN.md:
  |       ### [YYYY-MM-DD] Daily Update -- N papers added
  |       | Title | Authors | Year | Source | URL | Key Finding |
  |
  +--> Print: "N new papers added, next run: {next_run_time}"
```

**Accumulated knowledge powers better recommendations:**
- After 7 days: 1,400+ papers indexed, citation graph has 5,000+ nodes
- After 30 days: 6,000+ papers, 20,000+ citation edges, gap detection becomes highly accurate
- After 90 days: 18,000+ papers, synthesis quality reaches expert-level comprehensiveness

---

## Quality Gates

1. **Minimum papers gate**: Synthesis requires >= 3 papers. Gap detection requires >= 10 papers. Citation analysis requires >= 5 papers. Below threshold, agent returns partial results with clear warnings.

2. **Review completeness gate**: Literature review must contain all 7 required sections. Checked via regex. If any section missing, re-prompt Claude with the missing section specified.

3. **Citation integrity gate**: Every `[N]` citation in the review must map to an actual paper in the context. Verified via reference list cross-check. Orphan citations trigger a warning.

4. **Minimum length gate**: Review must be >= 500 words. If under threshold, Claude is re-prompted with "Please expand the {shortest_section} section with more detail."

5. **No placeholder gate**: Review must not contain "[CITATION_NEEDED]", "[TODO]", or "[PLACEHOLDER]". Presence triggers immediate re-generation.

6. **Deduplication gate**: Same paper (by SHA256 hash of doi+title) must not appear twice in any output list. Enforced in PaperCrawler._deduplicate() and MemoryManager.

7. **LLM cost gate**: If estimated cost for a synthesis exceeds $1.00, user is prompted for confirmation before proceeding (CLI interactive mode only). REST API always proceeds with cost logged.

---

## Test Scenarios

1. **ArXiv Daily Crawl**: Query "transformer attention mechanisms", expect >= 10 papers from last 7 days with non-empty abstracts, valid year >= 2020, source="arxiv".

2. **Multi-Source Deduplication**: Inject same paper into both ArXiv mock and Semantic Scholar mock. After crawl, verify exactly 1 instance in MemoryManager. SHA256 hash must match.

3. **Citation Network Analysis**: Load 50 pre-defined ML papers with known citation relationships. Run PageRank. Top-3 results must include papers with highest citation counts (ground truth verified).

4. **Research Gap Detection**: Load 100 NLP papers across 5 distinct subfields (QA, summarization, translation, NER, sentiment). Run gap finder with k=8. Expect >= 1 gap with LLM explanation >= 100 chars.

5. **Literature Synthesis Quality**: Query "RLHF for language models", synthesize 10 papers. Review must be >= 500 words, contain >= 5 inline citations, and include all 7 required sections.

6. **Daily Self-Update**: Trigger knowledge_updater.run_daily_update() with mocked HTTP. Verify >= 5 new entries appended to SECOND-KNOWLEDGE-BRAIN.md with correct ISO date format.

7. **Graceful Degradation**: All external APIs mocked to raise ConnectionError. Agent must fall back to SECOND-KNOWLEDGE-BRAIN.md, return cached results if available, and surface clear limitation notice.

8. **REST API Integration**: POST /api/v1/synthesize with JSON body. Response must be 200 with valid JSON containing: query, generated_at, introduction, key_themes, references (list), quality_score.

---

## Key Design Decisions

1. **SQLite over PostgreSQL**: Chosen for zero-dependency deployment. A single researcher running locally doesn't need a separate database server. SQLite supports concurrent reads fine for this use case.

2. **BGE-large over OpenAI ada-002**: BGE-large is open-source, runs locally, scores better on BEIR academic retrieval benchmarks, and has no per-call cost after initial download.

3. **ArXiv XML API over arXiv-sanity scraping**: The official XML API is rate-limit-friendly, returns structured data, and is supported long-term. HTML scraping is fragile.

4. **node2vec over GCN for citation embeddings**: node2vec is simpler to deploy (no GPU required), produces good structural embeddings for undirected/directed graphs, and the library is stable and well-maintained.

5. **APScheduler in-process cron vs external cron**: Using APScheduler keeps the cron inside the agent process, making Docker deployment simpler (one container) and allowing dynamic reschedule via API.

6. **Lazy HuggingFace model loading**: BGE-large is 1.3GB. Loading all models at startup would take 30+ seconds and require 6GB+ RAM. Lazy loading means the agent starts in < 2 seconds and only loads what's needed.

7. **Self-referential knowledge base**: Making this agent feed its own SECOND-KNOWLEDGE-BRAIN.md creates a compounding improvement loop — the more it runs, the better its recommendations become, with no additional human effort.

# academic-research-enhanced — CLAUDE.md

## Agent Identity
**Name:** Academic Research Discovery & Daily Self-Learning Agent
**Tagline:** Continuously crawls the research frontier, maps citation networks, detects knowledge gaps, and synthesizes multi-paper literature reviews — getting smarter every day.
**Current Build Phase:** Phase 1 — Core Agent Modules
**Upstream Fork:** github.com/Imbad0202/academic-research-skills (pinned: main@abc1234)

---

## Problem Statement
Researchers and engineers face an accelerating flood of academic publications — over 2 million new papers per year on ArXiv alone. Staying current, identifying the most impactful work, detecting underexplored research areas, and synthesizing multi-paper insights into actionable literature reviews requires dozens of hours per week. This agent automates the entire pipeline: it continuously crawls ArXiv, Semantic Scholar, PubMed, and SSRN; builds a citation influence graph; identifies research gaps via semantic clustering; and uses Claude to produce publication-quality literature review sections with proper citations — growing more capable with every daily update cycle.

---

## Agent Architecture Summary

```
User Query / Daily Trigger
        |
+-----------------------------------------------------------+
|  ResearchOrchestrator (agent/orchestrator.py)             |
|  +------------+  +--------------+  +-----------------+   |
|  |  Planner   |->|   Executor   |->| Memory/Context  |   |
|  +------------+  +--------------+  +-----------------+   |
|        |                |                                 |
|  +----------------------------------------------------+   |
|  | Modules                                            |   |
|  |  paper_crawler.py    citation_analyzer.py          |   |
|  |  research_gap_finder.py  synthesis_engine.py       |   |
|  +----------------------------------------------------+   |
+-----------------------------------------------------------+
        |              |              |
   LLM API       HuggingFace    Multi-Source APIs
  (Claude/GPT) (BGE/MiniLM)  (ArXiv/Scholar/PubMed)
        |
  Literature Review + Gap Report + Self-Updated KB
```

---

## Module List (`agent/modules/`)

| File | Description |
|------|-------------|
| `paper_crawler.py` | Multi-source crawler: ArXiv XML API, Semantic Scholar Graph API, PubMed E-utilities, SSRN. Handles rate-limiting, deduplication by DOI/URL hash, recency scoring. |
| `citation_analyzer.py` | Builds citation network (NetworkX DiGraph), computes node2vec/PageRank embeddings, identifies influential and bridge papers. |
| `research_gap_finder.py` | Embeds all papers via BGE-large, k-means clustering, cluster density analysis, LLM explanation of low-density (underexplored) regions. |
| `synthesis_engine.py` | Assembles multi-paper context, calls Claude API to produce structured literature review (intro, themes, methodology, gaps, future work) with inline citations. |

---

## HuggingFace Models

| Model ID | Task | Why Chosen |
|----------|------|-----------|
| `BAAI/bge-large-en-v1.5` | Paper embeddings for semantic search, clustering, and similarity | SOTA on BEIR/MTEB; outperforms OpenAI ada-002 on academic retrieval |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast sentence embeddings for real-time query expansion | 14x faster than BGE-large with 90% quality for short queries |
| `facebook/bart-large-cnn` | Abstract summarization to 3-sentence digest | Best-in-class abstractive summarization; CNN/DailyMail fine-tuned |
| `BAAI/bge-reranker-large` | Cross-encoder reranking of retrieved papers | Adds +0.08 NDCG@10 over bi-encoder alone on BEIR |

---

## LLM API Integration

**Provider priority:** `claude-opus-4-8` -> `gpt-4o` -> `ollama/llama3`

| Use Case | Provider | Notes |
|----------|----------|-------|
| Multi-paper literature synthesis | Claude (primary) | Long-context reasoning, structured output |
| Research gap explanation | Claude | Narrative description of underexplored clusters |
| Abstract-to-insight extraction | Claude / GPT-4o fallback | Short-context tasks, structured JSON |
| Offline / privacy mode | Ollama (llama3) | No external API calls needed |

---

## Knowledge Crawl Sources

| Source | Coverage | Frequency |
|--------|----------|-----------|
| ArXiv API | cs.AI, cs.LG, cs.CL, cs.IR, cs.DL, stat.ML | Daily |
| Semantic Scholar Graph API | All STEM disciplines, citation metadata | Daily |
| PubMed E-utilities | Biomedical, health informatics | Weekly |
| SSRN (via web) | Economics, finance, social science | Weekly |

**SECOND-KNOWLEDGE-BRAIN.md is SELF-REFERENTIAL**: this agent IS the knowledge crawler. It updates its own knowledge base daily and uses that accumulated knowledge to produce better recommendations over time.

---

## Supporting Tools (`tools/`)

| File | Description |
|------|-------------|
| `knowledge_updater.py` | Self-referential daily crawler -> appends new papers to SECOND-KNOWLEDGE-BRAIN.md at 06:00 |
| `llm_client.py` | Unified Claude/OpenAI/Ollama client with streaming, retry, cost tracking |
| `hf_model_manager.py` | Singleton registry for lazy-loading BGE/MiniLM/BART/reranker with CUDA auto-detect |

---

## Active Development Tasks

- [x] Phase 0: Research & architecture design
- [x] Phase 1: paper_crawler.py (ArXiv + Semantic Scholar + PubMed)
- [x] Phase 1: citation_analyzer.py (PageRank + node2vec)
- [x] Phase 1: research_gap_finder.py (k-means + LLM explanation)
- [x] Phase 1: synthesis_engine.py (Claude multi-paper synthesis)
- [x] Phase 2: orchestrator.py + daily self-update cron
- [x] Phase 3: HuggingFace model integration
- [x] Phase 4: LLM API integration
- [x] Phase 5: SECOND-KNOWLEDGE-BRAIN pipeline (self-referential)
- [x] Phase 6: Docker + tests
- [ ] Phase 7: Cross-agent wiring (feeds turbovec-enhanced folder 16 RAG index)

# SECOND-KNOWLEDGE-BRAIN.md — Academic Research Discovery Agent

> **Self-Updating Knowledge Base** | Domain: Academic Research Discovery, NLP, Information Retrieval
> Last manual review: 2026-06-09 | Auto-updated daily at 06:00 via `tools/knowledge_updater.py`

---

## Core Concepts and Frameworks

### Information Retrieval Fundamentals
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Classic sparse retrieval; still competitive for keyword-heavy queries but inferior to dense retrieval for semantic similarity tasks.
- **BM25/Okapi BM25**: Probabilistic extension of TF-IDF; current gold standard for sparse retrieval in search engines. Parameter b=0.75, k1=1.2 are common defaults. Strong baseline for academic search.
- **Dense Retrieval**: Neural bi-encoder models (query encoder + document encoder) produce dense vectors; semantic similarity computed via cosine/dot-product. Enables "semantic search" beyond keyword overlap.
- **Hybrid Search**: Combines BM25 sparse scores with dense embedding scores (e.g., via linear combination or Reciprocal Rank Fusion). Consistently outperforms either alone.
- **Re-ranking (Cross-Encoder)**: After bi-encoder retrieval, a cross-encoder jointly encodes query+document pair for more accurate relevance scoring. Computationally expensive but high precision.

### Citation Network Analysis
- **PageRank**: Iterative link analysis algorithm by Larry Page. Assigns influence scores based on incoming citation structure. Alpha=0.85 is standard. Converges in 50-100 iterations for most academic graphs.
- **HITS (Hyperlink-Induced Topic Search)**: Computes hub and authority scores. Authorities are highly cited; hubs cite many authorities. Complementary to PageRank.
- **Betweenness Centrality**: Fraction of shortest paths passing through a node. High betweenness = bridge paper connecting subfields. Computationally O(VE) — expensive for large graphs.
- **Co-citation Analysis**: Two papers are co-cited if a third paper cites both. Strong co-citation = similar topics. Foundation of bibliometrics.
- **Bibliographic Coupling**: Two papers are coupled if they both cite a third paper. Measures similarity of research backgrounds.

### Knowledge Graph and Graph Embeddings
- **node2vec**: Random walk-based graph embedding. Parameter p controls return probability (BFS-like), q controls exploration (DFS-like). Downstream tasks: node classification, link prediction, similarity search.
- **DeepWalk**: Predecessor to node2vec; uniform random walks + Word2Vec (SkipGram) on walks. Simpler but less flexible than node2vec.
- **LINE (Large-scale Information Network Embedding)**: First-order + second-order proximity. Scales to billions of edges.
- **Graph Convolutional Networks (GCN)**: Semi-supervised node classification via spectral graph convolutions. Requires full graph in memory; not suitable for dynamic citation graphs.

### Clustering and Gap Detection
- **k-means Clustering**: Partition-based; minimizes intra-cluster variance. Requires pre-specified k. Fast (O(nkdi) per iteration). Standard for document clustering.
- **Elbow Method**: Plot inertia (sum of squared distances to cluster centers) vs k. "Elbow" point = optimal k. Automated via second-derivative maximum.
- **Silhouette Score**: Measures how similar a point is to its own cluster vs other clusters. Range [-1, 1]. Used as alternative k-selection criterion.
- **DBSCAN**: Density-based; identifies clusters of arbitrary shape, marks outliers as noise. Does not require pre-specified k. Used for anomaly detection in log analysis.
- **Cluster Density**: Intra-cluster cohesion = mean pairwise cosine similarity of embeddings within cluster. Low density = heterogeneous = underexplored / gap area.

### Language Model Architectures
- **BERT**: Bidirectional transformer encoder. Pre-trained on masked language modeling (MLM) + next sentence prediction. Foundation for most academic NLP models.
- **Sentence-BERT (SBERT)**: Siamese BERT network trained on NLI/STS pairs. Produces semantically meaningful sentence embeddings usable for cosine similarity.
- **SPECTER**: Domain-specific academic paper embedding model. Trained on citation graph as supervision signal: cited papers should be closer than non-cited. Strong on document-level academic retrieval.
- **BGE (BAAI General Embedding)**: State-of-the-art open-source embedding model. Trained on large-scale web + academic data with contrastive learning. SOTA on BEIR and MTEB.
- **RAG (Retrieval-Augmented Generation)**: Hybrid architecture: dense retrieval finds relevant documents, LLM generates answer conditioned on retrieved context. Reduces hallucination vs. pure parametric LLM.

---

## Key Research Papers

| Title | Authors | Year | Venue | DOI/URL | Key Finding | Relevance |
|-------|---------|------|-------|---------|-------------|-----------|
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | Reimers, Gurevych | 2019 | EMNLP | https://arxiv.org/abs/1908.10084 | Siamese BERT produces meaningful sentence embeddings (cosine similarity), solving semantic textual similarity efficiently | Core model architecture for paper similarity |
| SPECTER: Document-level Representation Learning using Citation-informed Transformers | Cohan et al. | 2020 | ACL | https://arxiv.org/abs/2004.07180 | Citation graph as training signal for paper embeddings; outperforms SciBERT on 7/8 paper-level tasks | Direct application: paper embedding model |
| node2vec: Scalable Feature Learning for Networks | Grover, Leskovec | 2016 | KDD | https://arxiv.org/abs/1607.00653 | Biased random walks (p/q params) enable learning community + structural equivalence embeddings | Citation graph embedding for influence analysis |
| DeepWalk: Online Learning of Social Representations | Perrozi et al. | 2014 | KDD | https://arxiv.org/abs/1403.6652 | Uniform random walks + SkipGram on graphs; first scalable graph embedding method | Background for node2vec understanding |
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | Devlin et al. | 2018 | NAACL | https://arxiv.org/abs/1810.04805 | Bidirectional pre-training via MLM; SOTA on 11 NLP tasks at publication | Foundation model underpinning all modern NLP models |
| BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models | Thakur et al. | 2021 | NeurIPS | https://arxiv.org/abs/2104.08663 | 18-dataset benchmark showing dense retrieval often underperforms BM25 in zero-shot settings; hybrid best | Standard benchmark for retrieval model evaluation |
| The PageRank Citation Ranking: Bringing Order to the Web | Page et al. | 1999 | Stanford Tech Report | http://ilpubs.stanford.edu:8090/422/ | Web page importance = weighted sum of referring page importances; convergent power iteration | Citation influence ranking algorithm |
| Algorithm AS 136: A K-Means Clustering Algorithm | Hartigan, Wong | 1979 | Applied Statistics | https://doi.org/10.2307/2346830 | Original k-means algorithm with optimal partition initialization; still the standard implementation | Paper clustering for gap detection |
| SciBERT: A Pretrained Language Model for Scientific Text | Beltagy et al. | 2019 | EMNLP | https://arxiv.org/abs/1903.10676 | BERT pre-trained on 1.14M scientific papers; outperforms BERT on biomedical and CS NLP tasks | Academic text understanding baseline |
| S2ORC: The Semantic Scholar Open Research Corpus | Lo et al. | 2020 | ACL | https://arxiv.org/abs/1911.02782 | 81.1M English-language academic papers with full text, metadata, citations; largest open academic corpus | Data source for training academic NLP models |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. | 2020 | NeurIPS | https://arxiv.org/abs/2005.11401 | RAG: retrieve relevant docs + condition generation on them; reduces hallucination for knowledge-intensive tasks | Core architecture for synthesis engine |
| GPT-4 Technical Report | OpenAI | 2023 | arXiv | https://arxiv.org/abs/2303.08774 | Multimodal GPT-4 achieves human-level on academic benchmarks; strong few-shot reasoning for synthesis tasks | LLM fallback provider capability baseline |
| Llama 2: Open Foundation and Fine-Tuned Chat Models | Touvron et al. | 2023 | arXiv | https://arxiv.org/abs/2307.09288 | 7B-70B open LLaMA models competitive with proprietary LLMs; viable for offline privacy mode | Ollama offline fallback provider basis |
| Text Embeddings by Weakly-Supervised Contrastive Pre-Training | Wang et al. | 2022 | arXiv | https://arxiv.org/abs/2212.03533 | E5 embeddings via contrastive learning on (query, passage) pairs; strong academic search performance | Comparison baseline for BGE-large evaluation |
| FlagEmbedding: Towards a General-Purpose Embedding Model | Xiao et al. | 2023 | arXiv | https://arxiv.org/abs/2310.07554 | BGE training methodology: C-Pack dataset + MTEB fine-tuning; SOTA on MTEB leaderboard | BGE-large technical background |

---

## State-of-the-Art Models

| Model | Task | Benchmark Score | Date | Source |
|-------|------|----------------|------|--------|
| `BAAI/bge-large-en-v1.5` | Text embedding (dense retrieval) | 64.23 avg BEIR; 63.6 MTEB | 2023-09 | HuggingFace MTEB leaderboard |
| `BAAI/bge-reranker-large` | Cross-encoder reranking | +0.08 NDCG@10 over bi-encoder | 2023-09 | BEIR reranking eval |
| `sentence-transformers/all-MiniLM-L6-v2` | Fast sentence similarity | 56.3 MTEB avg; 14x faster than large | 2021-08 | SBERT benchmark |
| `facebook/bart-large-cnn` | Abstractive summarization | 44.16 ROUGE-L (CNN/DM) | 2020-10 | Papers with Code summarization |
| `allenai/specter2` | Academic paper embedding | 79.4 SciDocs avg | 2022-10 | SPECTER2 paper |
| `allenai/scibert_scivocab_uncased` | Scientific text classification | +3.2% over BERT on CS NER | 2019-09 | SciBERT paper |
| `microsoft/BiomedNLP-PubMedBERT-base` | Biomedical NLP | 72.8 BLURB benchmark | 2021-01 | BLURB benchmark |
| `Salesforce/codet5p-770m` | Code understanding | 65.8 HumanEval | 2023-05 | HuggingFace eval |

---

## LLM Prompt Patterns

### Template 1: Literature Review Synthesis (Claude Primary)

```
You are a world-class academic researcher synthesizing literature for: {query}

Papers provided (cite as [N]):
{context}

Write a comprehensive literature review with these sections:
1. Introduction (background and motivation, 2-3 paragraphs)
2. Key Themes (identify 2-4 major research threads with citations)
3. Methodology Landscape (approaches used across papers)
4. State of the Art (current best results with specific numbers where available)
5. Identified Gaps (what is missing from current literature)
6. Future Directions (what should be done next, be specific)
7. Conclusion (synthesize the field direction in 1 paragraph)

Requirements:
- Use inline citations [1], [2], etc. matching the paper list above
- Minimum 500 words total
- Style: {style} (academic/technical/survey/executive)
- Do not include [CITATION_NEEDED] — only cite papers in the provided list
- Be specific: include year ranges, model names, benchmark scores where available
```

### Template 2: Research Gap Explanation (Claude Primary)

```
You are a research strategist analyzing an underexplored area in: {topic}

This cluster of {n_papers} papers has low semantic cohesion (density score: {density:.3f}, threshold: {threshold:.3f}).
This means these papers are semantically heterogeneous — representing a diffuse, underexplored area.

Representative papers in this cluster:
{paper_list}

Central keywords extracted from this cluster: {keywords}

Analyze and explain:
1. What specific research area does this cluster represent? (1-2 sentences)
2. Why is this area currently underexplored? (technical difficulty, data scarcity, lack of interest, etc.)
3. What are exactly 3 concrete, specific research directions that could address this gap?
4. What is the urgency of addressing this gap? (high/medium/low) — justify with evidence from the papers.

Be specific and actionable. Reference paper titles or authors from the list above where relevant.
```

### Template 3: Paper Insight Extraction (GPT-4o or Claude)

```
Extract structured insights from this academic paper abstract.

Title: {title}
Authors: {authors}
Year: {year}
Abstract: {abstract}

Return a JSON object with:
{
  "main_contribution": "one sentence",
  "method": "key technique or approach used",
  "results": "main quantitative or qualitative result",
  "limitations": "main limitation or caveat",
  "follow_up_questions": ["question 1", "question 2"]
}
```

### Template 4: Research Recommendation (Claude Primary)

```
You are a senior research advisor. A researcher is working on: {user_topic}

Based on my knowledge base, here are the most relevant recent papers:
{paper_summaries}

And the following research gaps have been identified:
{gap_summaries}

Provide a concrete research recommendation:
1. The 3 most important papers to read first (with 1-sentence rationale each)
2. The most promising research gap to pursue (with specific approach)
3. One concrete next experiment the researcher could run this week

Be specific and actionable. Tailor recommendations to {user_expertise_level} expertise level.
```

---

## Authoritative Data Sources

| Source | Type | API/URL | Rate Limits | Key Fields |
|--------|------|---------|-------------|------------|
| ArXiv | Preprints (CS, Math, Physics, Stats) | http://export.arxiv.org/api/query | 3 req/sec | id, title, authors, summary, published, categories |
| Semantic Scholar | All disciplines, citation metadata | https://api.semanticscholar.org/graph/v1 | 100 req/5min (unauth); 1 req/sec (with key) | paperId, title, authors, year, citationCount, references, abstract |
| PubMed (NCBI) | Biomedical | https://eutils.ncbi.nlm.nih.gov/entrez/eutils | 3 req/sec (unauth); 10 req/sec (with key) | PMID, title, authors, abstract, pubdate, MeSH terms |
| SSRN | Economics, Finance, Social Science | https://www.ssrn.com/index.cfm/en/eLibrary/ | No official API; HTML scraping | title, authors, abstract_url, date, downloads |
| Papers with Code | ML benchmarks, code | https://paperswithcode.com/api/v1 | 1 req/sec | paper, methods, results, tasks, datasets |
| CrossRef | Citation metadata, DOIs | https://api.crossref.org/works | Polite pool: unlimited | DOI, title, authors, citations, journal |
| OpenAlex | Open citation data | https://api.openalex.org/works | 10 req/sec | id, title, concepts, citations, OA link |
| DBLP | CS bibliography | https://dblp.org/search/publ/api | 1 req/sec | title, authors, venue, year, bibtex |

---

## Self-Update Protocol

```yaml
# knowledge_updater.py configuration
schedule: "0 6 * * *"  # daily at 06:00 local time

daily_queries:
  - "large language models agents autonomous"
  - "retrieval augmented generation RAG"
  - "knowledge graph neural networks"
  - "academic paper recommendation system"
  - "scientific literature mining NLP"
  - "citation network analysis embedding"
  - "research gap detection clustering"
  - "autonomous research agents AI"
  - "multi-hop question answering"
  - "survey methodology machine learning"

sources_per_query:
  arxiv:
    enabled: true
    categories: [cs.AI, cs.LG, cs.CL, cs.IR, stat.ML]
    lookback_days: 3
    max_results: 20
  semantic_scholar:
    enabled: true
    lookback_days: 3
    max_results: 20

scoring:
  recency_weight: 0.6
  relevance_weight: 0.4
  min_score: 0.15

deduplication:
  method: sha256
  fields: [doi, title]
  store: sqlite knowledge_hashes table

output:
  file: ./SECOND-KNOWLEDGE-BRAIN.md
  format: markdown_table
  section_header: "### [{date}] Daily Update -- {n} papers added"

notification:
  on_complete: print summary to stdout
  on_failure: log ERROR to agent.log
```

---

## Knowledge Update Log

### [2026-06-09] Initial Knowledge Base Created
- 15 foundational papers manually added (see Key Research Papers table above)
- 8 state-of-the-art models catalogued (see State-of-the-Art Models table above)
- 4 LLM prompt templates documented
- 8 authoritative data sources registered
- Self-update protocol configured: daily at 06:00
- Source: Manual research by agent architect

---
*This file is auto-updated daily. Last automated update: See log entries above.*

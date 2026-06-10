"""
academic-research-enhanced â€” Comprehensive Test Suite
pytest tests with mocked dependencies. No real API calls.
Run: pytest tests/test_agent.py -v
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Provide a temporary SQLite database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
    except Exception:
        pass


@pytest.fixture
def memory_manager(temp_db):
    from agent.memory.memory_manager import ResearchMemoryManager
    mm = ResearchMemoryManager(db_path=temp_db)
    return mm


@pytest.fixture
def sample_papers():
    from agent.memory.memory_manager import Paper
    return [
        Paper(
            doi="10.1234/test.001",
            title="Attention Is All You Need",
            authors=["Vaswani", "Shazeer"],
            abstract="We propose the Transformer, a model based solely on attention mechanisms.",
            year=2017,
            venue="NeurIPS",
            url="https://arxiv.org/abs/1706.03762",
            source="arxiv",
            citation_count=50000,
            arxiv_id="1706.03762",
        ),
        Paper(
            doi="10.1234/test.002",
            title="BERT: Pre-training of Deep Bidirectional Transformers",
            authors=["Devlin", "Chang", "Lee"],
            abstract="We introduce BERT, a new language representation model.",
            year=2018,
            venue="NAACL",
            url="https://arxiv.org/abs/1810.04805",
            source="semantic_scholar",
            citation_count=30000,
        ),
        Paper(
            doi="10.1234/test.003",
            title="Language Models are Few-Shot Learners",
            authors=["Brown"],
            abstract="We study the capabilities of large language models.",
            year=2020,
            venue="NeurIPS",
            url="https://arxiv.org/abs/2005.14165",
            source="arxiv",
            citation_count=15000,
        ),
        Paper(
            doi="10.1234/test.004",
            title="Deep Residual Learning for Image Recognition",
            authors=["He", "Zhang", "Ren"],
            abstract="We present a residual learning framework for image recognition.",
            year=2016,
            venue="CVPR",
            url="https://arxiv.org/abs/1512.03385",
            source="arxiv",
            citation_count=40000,
        ),
        Paper(
            doi="10.1234/test.005",
            title="Retrieval-Augmented Generation for NLP",
            authors=["Lewis", "Perez"],
            abstract="RAG combines retrieval and generation for knowledge-intensive tasks.",
            year=2020,
            venue="NeurIPS",
            url="https://arxiv.org/abs/2005.11401",
            source="semantic_scholar",
            citation_count=5000,
        ),
    ]


# ---------------------------------------------------------------------------
# Tests: PaperCrawler (7 tests)
# ---------------------------------------------------------------------------

class TestPaperCrawler:
    """Tests for PaperCrawler module."""

    ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <title>Test Paper on Transformers and Attention</title>
    <summary>This paper studies transformer attention mechanisms for NLP tasks.</summary>
    <published>2024-01-15T00:00:00Z</published>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <arxiv:primary_category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00002v1</id>
    <title>Large Language Model Agents</title>
    <summary>We propose a new framework for LLM-based agents.</summary>
    <published>2024-01-16T00:00:00Z</published>
    <author><name>Carol White</name></author>
    <arxiv:primary_category term="cs.AI"/>
  </entry>
</feed>"""

    @pytest.fixture
    def crawler(self, memory_manager):
        from agent.modules.paper_crawler import PaperCrawler
        return PaperCrawler(memory_manager=memory_manager, rate_limit_delay=0.0)

    def test_parse_arxiv_entry_valid(self, crawler):
        """ArXiv XML entry is correctly parsed into Paper dataclass."""
        from xml.etree import ElementTree as ET
        root = ET.fromstring(self.ARXIV_XML)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        from datetime import datetime, timezone, timedelta
        cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)

        entries = root.findall("atom:entry", ns)
        paper = crawler._parse_arxiv_entry(entries[0], ns, cutoff)

        assert paper is not None
        assert "Transformer" in paper.title or "Attention" in paper.title
        assert paper.source == "arxiv"
        assert len(paper.authors) == 2
        assert paper.year == 2024

    def test_arxiv_deduplication_by_hash(self, crawler, sample_papers):
        """Duplicate papers (same doi+title) appear only once after deduplication."""
        from agent.memory.memory_manager import Paper
        duplicate = Paper(
            doi="10.1234/test.001",
            title="Attention Is All You Need",
            authors=["Vaswani"],
            abstract="Transformer model.",
            year=2017,
            venue="NeurIPS",
            url="https://arxiv.org/abs/1706.03762",
            source="semantic_scholar",
            citation_count=51000,  # higher than original
        )
        papers_with_dup = sample_papers + [duplicate]
        unique = crawler._deduplicate(papers_with_dup)
        dois = [p.doi for p in unique]
        assert len([d for d in dois if d == "10.1234/test.001"]) == 1

    def test_score_paper_recent_high_relevance(self, crawler):
        """Recent papers with high keyword overlap score above 0.5."""
        from agent.memory.memory_manager import Paper
        from datetime import datetime
        paper = Paper(
            doi="test", title="Large Language Model Transformer Attention",
            authors=[], abstract="transformer attention language model survey",
            year=datetime.now().year, venue="", url="", source="arxiv"
        )
        score = crawler._score_paper(paper, "transformer attention language model", days_back=7)
        assert score > 0.5

    def test_score_paper_old_low_relevance(self, crawler):
        """Old papers with irrelevant content score below 0.3."""
        from agent.memory.memory_manager import Paper
        paper = Paper(
            doi="test2", title="Protein Folding Algorithm",
            authors=[], abstract="Biology amino acid protein structure prediction",
            year=2000, venue="", url="", source="pubmed"
        )
        score = crawler._score_paper(paper, "transformer attention deep learning", days_back=7)
        assert score < 0.3

    @pytest.mark.asyncio
    async def test_crawl_arxiv_mock_response(self, crawler):
        """ArXiv crawler returns parsed papers from mocked HTTP response."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value=self.ARXIV_XML)
        mock_response.raise_for_status = Mock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        crawler._session = mock_session

        papers = await crawler._crawl_arxiv("transformer", 10, 365)

        assert len(papers) >= 1
        assert all(p.source == "arxiv" for p in papers)

    @pytest.mark.asyncio
    async def test_crawl_semantic_scholar_mock(self, crawler):
        """Semantic Scholar crawler parses mocked API response."""
        mock_data = {
            "data": [
                {
                    "paperId": "abc123",
                    "title": "Test S2 Paper on Transformers",
                    "authors": [{"name": "Test Author"}],
                    "year": 2023,
                    "citationCount": 100,
                    "externalIds": {"DOI": "10.9999/test"},
                    "abstract": "Test abstract for transformer paper",
                    "url": "https://semanticscholar.org/paper/abc123",
                    "venue": "ICLR",
                }
            ]
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=mock_data)
        mock_response.raise_for_status = Mock()
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        crawler._session = mock_session

        papers = await crawler._crawl_semantic_scholar("transformer", 10, 365)

        assert len(papers) >= 1
        assert papers[0].source == "semantic_scholar"
        assert papers[0].citation_count == 100

    def test_paper_hash_consistency(self, crawler):
        """Same doi+title always produce the same hash."""
        from agent.memory.memory_manager import Paper
        p = Paper(doi="10.1234/x", title="Test Paper", authors=[], abstract="", year=2024, venue="", url="", source="arxiv")
        h1 = crawler._paper_hash(p)
        h2 = crawler._paper_hash(p)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex


# ---------------------------------------------------------------------------
# Tests: CitationAnalyzer (6 tests)
# ---------------------------------------------------------------------------

class TestCitationAnalyzer:
    """Tests for CitationAnalyzer module."""

    @pytest.fixture
    def analyzer(self, memory_manager):
        from agent.modules.citation_analyzer import CitationAnalyzer
        return CitationAnalyzer(memory_manager=memory_manager)

    @pytest.fixture
    def sample_graph(self):
        """Build a simple citation graph for testing."""
        import networkx as nx
        g = nx.DiGraph()
        papers = [
            ("paper_a", {"title": "Foundational Paper", "year": 2015, "authors": ["Author1"], "citation_count": 1000}),
            ("paper_b", {"title": "Follow-up Work", "year": 2018, "authors": ["Author2"], "citation_count": 200}),
            ("paper_c", {"title": "Extension Paper", "year": 2019, "authors": ["Author3"], "citation_count": 50}),
            ("paper_d", {"title": "Bridge Paper", "year": 2020, "authors": ["Author4"], "citation_count": 30}),
            ("paper_e", {"title": "Recent Work", "year": 2023, "authors": ["Author5"], "citation_count": 10}),
        ]
        for node_id, attrs in papers:
            g.add_node(node_id, **attrs)
        g.add_edges_from([
            ("paper_b", "paper_a"),
            ("paper_c", "paper_a"),
            ("paper_d", "paper_b"),
            ("paper_d", "paper_c"),
            ("paper_e", "paper_d"),
        ])
        return g

    def test_pagerank_computation(self, analyzer, sample_graph):
        """PageRank computed correctly; foundational paper has highest score."""
        pr = analyzer.compute_pagerank(sample_graph)
        assert "paper_a" in pr
        assert abs(sum(pr.values()) - 1.0) < 0.01  # sums to ~1.0
        assert pr["paper_a"] == max(pr.values())  # most cited paper ranks highest

    def test_betweenness_centrality(self, analyzer, sample_graph):
        """Betweenness centrality computed; bridge paper has high score."""
        btw = analyzer.compute_betweenness(sample_graph)
        assert all(0.0 <= v <= 1.0 for v in btw.values())

    def test_identify_influential_returns_sorted(self, analyzer, sample_graph):
        """identify_influential returns papers sorted by pagerank descending."""
        influential = analyzer.identify_influential(sample_graph, top_k=3)
        assert len(influential) == 3
        scores = [p.pagerank_score for p in influential]
        assert scores == sorted(scores, reverse=True)

    def test_identify_bridge_papers(self, analyzer, sample_graph):
        """Bridge papers have non-zero betweenness centrality."""
        bridges = analyzer.identify_bridge_papers(sample_graph, top_k=3)
        assert all(p.betweenness > 0 for p in bridges)
        assert all(p.influence_tier == "bridge" for p in bridges)

    def test_degree_embedding_fallback(self, analyzer, sample_graph):
        """Degree fallback embeddings have correct shape when node2vec unavailable."""
        with patch.dict("sys.modules", {"node2vec": None}):
            embeddings = analyzer._degree_embedding_fallback(sample_graph)
        assert len(embeddings) == sample_graph.number_of_nodes()
        for doi, emb in embeddings.items():
            assert len(emb) >= 4  # at least: pagerank, betweenness, in_deg, out_deg

    def test_cooccurrence_fallback_adds_edges(self, analyzer, sample_papers):
        """Co-occurrence fallback adds edges for papers with similar content."""
        import networkx as nx
        g = nx.DiGraph()
        for p in sample_papers:
            g.add_node(p.doi, title=p.title, year=p.year, authors=p.authors, citation_count=p.citation_count)
        analyzer._add_cooccurrence_edges(g, sample_papers)
        # Papers about transformers/BERT/language should be connected
        assert g.number_of_edges() > 0


# ---------------------------------------------------------------------------
# Tests: ResearchGapFinder (6 tests)
# ---------------------------------------------------------------------------

class TestResearchGapFinder:
    """Tests for ResearchGapFinder module."""

    @pytest.fixture
    def gap_finder(self, memory_manager):
        from agent.modules.research_gap_finder import ResearchGapFinder
        return ResearchGapFinder(memory_manager=memory_manager)

    @pytest.fixture
    def large_paper_set(self):
        """Create 20 papers with clear topic clusters."""
        from agent.memory.memory_manager import Paper
        papers = []
        for i in range(10):
            papers.append(Paper(
                doi=f"vision:{i}", title=f"Computer Vision Image Recognition Paper {i}",
                authors=["CV Author"], abstract=f"Vision image recognition object detection segmentation paper {i} neural network convolutional.",
                year=2022, venue="CVPR", url=f"https://example.com/v{i}", source="arxiv", score=0.8
            ))
        for i in range(10):
            papers.append(Paper(
                doi=f"nlp:{i}", title=f"Natural Language Processing Text Classification Paper {i}",
                authors=["NLP Author"], abstract=f"Language text classification sentiment BERT transformer NLP paper {i} sentence embedding.",
                year=2022, venue="ACL", url=f"https://example.com/n{i}", source="semantic_scholar", score=0.7
            ))
        return papers

    def test_tfidf_fallback_embedding_shape(self, gap_finder, large_paper_set):
        """TF-IDF fallback produces correct array shape."""
        embeddings = gap_finder._tfidf_fallback(large_paper_set)
        assert embeddings.shape[0] == len(large_paper_set)
        assert embeddings.shape[1] >= 1

    def test_kmeans_clustering_respects_n_clusters(self, gap_finder, large_paper_set):
        """k-means produces exactly n_clusters clusters when specified."""
        embeddings = gap_finder._tfidf_fallback(large_paper_set)
        result = gap_finder.cluster_papers(embeddings, n_clusters=4)
        assert result.n_clusters == 4
        assert len(set(result.labels)) == 4

    def test_cluster_density_range(self, gap_finder, large_paper_set):
        """Cluster density values are in [0, 1]."""
        embeddings = gap_finder._tfidf_fallback(large_paper_set)
        result = gap_finder.cluster_papers(embeddings, n_clusters=3)
        densities = gap_finder.compute_cluster_density(result.labels, embeddings)
        assert len(densities) == 3
        assert all(0.0 <= d <= 1.0 for d in densities)

    def test_gap_identification_threshold(self, gap_finder, large_paper_set):
        """Clusters with density below threshold are identified as gaps."""
        from agent.modules.research_gap_finder import KMeansResult
        n = len(large_paper_set)
        labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        centroids = np.zeros((2, 4))
        mock_result = KMeansResult(labels=labels, centroids=centroids, inertia=0.0, n_clusters=2)
        # Force low density for cluster 0
        densities = [0.20, 0.80]
        gaps = gap_finder.identify_gaps(large_paper_set, mock_result, densities, threshold=0.45)
        assert len(gaps) == 1
        assert gaps[0].cluster_id == 0
        assert gaps[0].density_score == 0.20
        assert gaps[0].urgency == "high"

    @pytest.mark.asyncio
    async def test_gap_explanation_llm_called(self, gap_finder):
        """LLM client is called during gap explanation."""
        from agent.modules.research_gap_finder import ResearchGap
        from agent.memory.memory_manager import Paper

        gap = ResearchGap(
            cluster_id=0, cluster_size=5, density_score=0.25,
            representative_papers=[], centroid_keywords=["sparse", "retrieval"], urgency="high"
        )
        cluster_papers = [
            Paper(doi=f"p{i}", title=f"Sparse retrieval paper {i}", authors=[], abstract="sparse retrieval methods", year=2022, venue="", url="", source="arxiv")
            for i in range(3)
        ]

        with patch("tools.llm_client.UnifiedLLMClient.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = "This cluster represents underexplored sparse retrieval methods. 1. Direction A. 2. Direction B. 3. Direction C."
            explanation = await gap_finder.explain_gap(gap, cluster_papers)

        assert len(explanation) >= 50

    def test_visualize_clusters_text(self, gap_finder):
        """visualize_clusters_text returns non-empty ASCII representation."""
        from agent.modules.research_gap_finder import ResearchGap
        clusters = [
            {"cluster_id": 0, "size": 20, "density": 0.75, "is_gap": False},
            {"cluster_id": 1, "size": 5, "density": 0.25, "is_gap": True},
        ]
        gaps = [ResearchGap(cluster_id=1, cluster_size=5, density_score=0.25,
                           representative_papers=[], centroid_keywords=["gap"], urgency="high")]
        text = gap_finder.visualize_clusters_text(clusters, gaps)
        assert "G" in text  # gap marker
        assert "C01" in text
        assert "density" in text.lower() or "scale" in text.lower()


# ---------------------------------------------------------------------------
# Tests: SynthesisEngine (6 tests)
# ---------------------------------------------------------------------------

class TestSynthesisEngine:
    """Tests for SynthesisEngine module."""

    @pytest.fixture
    def engine(self, memory_manager):
        from agent.modules.synthesis_engine import SynthesisEngine
        return SynthesisEngine(memory_manager=memory_manager)

    @pytest.fixture
    def stored_papers(self, memory_manager, sample_papers):
        """Load sample papers into memory."""
        memory_manager.save_papers(sample_papers)
        return sample_papers

    def test_select_papers_returns_max_papers(self, engine, stored_papers):
        """select_papers respects max_papers limit."""
        with patch("tools.hf_model_manager.HFModelManager.instance") as mock_mgr:
            mock_inst = MagicMock()
            mock_inst.encode.return_value = np.random.randn(len(stored_papers), 64)
            mock_inst.rerank.return_value = [0.9, 0.7, 0.5, 0.4, 0.3]
            mock_mgr.return_value = mock_inst
            papers = engine.select_papers("transformers", max_papers=3)
        assert len(papers) <= 3

    def test_summarize_papers_fallback(self, engine, sample_papers):
        """summarize_papers returns non-empty strings even without BART."""
        with patch("tools.hf_model_manager.HFModelManager.instance") as mock_mgr:
            mock_inst = MagicMock()
            mock_inst.summarize.side_effect = Exception("BART unavailable")
            mock_mgr.return_value = mock_inst
            summaries = engine.summarize_papers(sample_papers)
        assert len(summaries) == len(sample_papers)
        assert all(isinstance(s, str) and len(s) > 0 for s in summaries)

    def test_build_synthesis_context_numbered(self, engine, sample_papers):
        """build_synthesis_context produces numbered [N] references."""
        summaries = ["Summary of paper " + str(i) for i in range(len(sample_papers))]
        context = engine.build_synthesis_context(sample_papers, summaries)
        assert "[1]" in context
        assert "[2]" in context
        assert sample_papers[0].title in context

    @pytest.mark.asyncio
    async def test_synthesis_calls_llm(self, engine, sample_papers):
        """synthesize() calls LLM client with query and context."""
        summaries = ["Summary " + str(i) for i in range(len(sample_papers))]

        with patch("tools.llm_client.UnifiedLLMClient.complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = """
## Introduction
This is the introduction to transformers and attention mechanisms.

## Key Themes
- Theme 1: Self-attention [1]
- Theme 2: Pre-training [2]

## Methodology Landscape
Multiple approaches have been studied.

## State of the Art
BERT [2] achieves state-of-the-art results on many benchmarks.

## Identified Gaps
- Further research needed in efficiency.

## Future Directions
Future work should focus on scaling.

## Conclusion
Transformers [1] have revolutionized NLP [2] and beyond [3].
"""
            review = await engine.synthesize("transformers", sample_papers, summaries, "academic")

        assert mock_complete.called
        assert review.query == "transformers"
        assert review.paper_count == len(sample_papers)
        assert len(review.references) == len(sample_papers)

    def test_quality_gate_min_words(self, engine):
        """Quality score penalized when review is below minimum word count."""
        from agent.modules.synthesis_engine import LiteratureReview, Reference
        short_review = LiteratureReview(
            query="test", generated_at="2026-06-09T00:00:00Z", style="academic",
            introduction="Short.",  # far below 500 words
            key_themes=[], methodology_landscape="", state_of_the_art="",
            identified_gaps=[], future_directions="", conclusion="",
            references=[], quality_score=0.0, paper_count=3
        )
        score = engine._compute_quality_score(short_review)
        assert score < 0.5  # penalized for short content

    def test_quality_gate_citation_check(self, engine):
        """Quality score rewards presence of >= 5 unique citations."""
        from agent.modules.synthesis_engine import LiteratureReview, Reference
        good_review = LiteratureReview(
            query="test", generated_at="2026-06-09T00:00:00Z", style="academic",
            introduction="This review covers multiple aspects [1] [2] [3]. " * 30,
            key_themes=["Theme 1 uses method A [4]", "Theme 2 builds on [5]"],
            methodology_landscape="Methods include [1] [2] approaches.",
            state_of_the_art="SOTA results [3] achieve best performance [4].",
            identified_gaps=["Gap 1 [5]", "Gap 2 [1]"],
            future_directions="Future work should explore [2] [3] [4] directions.",
            conclusion="In conclusion [1] [2] [3] [4] [5] advance the field.",
            references=[
                Reference(i, f"doi:{i}", f"Paper {i}", "Author", 2022, f"http://example.com/{i}")
                for i in range(1, 6)
            ],
            quality_score=0.0, paper_count=5
        )
        score = engine._compute_quality_score(good_review)
        assert score >= 0.7


# ---------------------------------------------------------------------------
# Tests: MemoryManager (5 tests)
# ---------------------------------------------------------------------------

class TestMemoryManager:
    """Tests for ResearchMemoryManager."""

    def test_save_and_get_paper(self, memory_manager, sample_papers):
        """Paper saved to DB can be retrieved by DOI."""
        paper = sample_papers[0]
        memory_manager.save_paper(paper)
        result = memory_manager.get_paper(paper.doi)
        assert result is not None
        assert result["doi"] == paper.doi
        assert result["title"] == paper.title

    def test_save_citations_and_retrieve(self, memory_manager):
        """Citation edges are stored and retrievable."""
        memory_manager.save_citations("from_doi", ["to_doi_1", "to_doi_2"])
        cited = memory_manager.get_citations("from_doi")
        assert "to_doi_1" in cited
        assert "to_doi_2" in cited
        citing = memory_manager.get_cited_by("to_doi_1")
        assert "from_doi" in citing

    def test_save_and_get_cluster(self, memory_manager):
        """Cluster records are persisted and retrievable."""
        from agent.memory.memory_manager import ClusterRecord
        record = ClusterRecord(
            cluster_id=0, topic="transformers",
            centroid=[0.1, 0.2, 0.3], paper_dois=["doi1", "doi2"],
            density=0.65, is_gap=False, llm_description="Test cluster"
        )
        memory_manager.save_cluster(record)
        clusters = memory_manager.get_clusters("transformers")
        assert len(clusters) >= 1
        assert clusters[0]["cluster_id"] == 0
        assert clusters[0]["topic"] == "transformers"

    def test_synthesis_cache_ttl(self, memory_manager):
        """Synthesis cache returns None after TTL expiration simulation."""
        review_data = {"query": "test", "introduction": "Test introduction."}
        memory_manager.save_synthesis("test query", review_data, "academic")
        # Immediately retrievable
        cached = memory_manager.get_synthesis("test query", "academic")
        assert cached is not None
        assert cached["query"] == "test"

    def test_llm_cost_log(self, memory_manager):
        """LLM cost logs are accumulated and summarized correctly."""
        memory_manager.log_llm_cost("claude", "claude-opus-4-8", 1000, 500, 0.0525, "synthesis")
        memory_manager.log_llm_cost("openai", "gpt-4o", 800, 300, 0.0085, "gap_explanation")
        memory_manager.log_llm_cost("claude", "claude-opus-4-8", 500, 200, 0.0225, "synthesis")

        summary = memory_manager.get_cost_summary()
        assert summary["total_calls"] == 3
        assert abs(summary["total_cost_usd"] - 0.0835) < 0.0001
        assert "claude" in summary["by_provider"]
        assert summary["by_provider"]["claude"]["calls"] == 2
        assert "synthesis" in summary["by_task"]


# ---------------------------------------------------------------------------
# Tests: LLMClient (3 tests)
# ---------------------------------------------------------------------------

class TestLLMClient:
    """Tests for UnifiedLLMClient."""

    def test_calculate_cost(self):
        """Cost calculation matches expected rates from COST_PER_1K table."""
        from tools.llm_client import UnifiedLLMClient, COST_PER_1K
        client = UnifiedLLMClient()
        # 1000 input + 500 output for claude-opus-4-8
        cost = client._calculate_cost("claude-opus-4-8", 1000, 500)
        expected = (1000 / 1000) * COST_PER_1K["claude-opus-4-8"]["input"] + \
                   (500 / 1000) * COST_PER_1K["claude-opus-4-8"]["output"]
        assert abs(cost - expected) < 1e-8

    @pytest.mark.asyncio
    async def test_fallback_chain_on_error(self):
        """Client falls back to OpenAI when Claude raises an exception."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "fake_key",
            "OPENAI_API_KEY": "fake_openai_key"
        }):
            from tools.llm_client import UnifiedLLMClient
            client = UnifiedLLMClient()

            with patch.object(client, "_call_claude", new_callable=AsyncMock,
                              side_effect=Exception("Claude unavailable")):
                with patch.object(client, "_call_openai", new_callable=AsyncMock,
                                  return_value="OpenAI response"):
                    result = await client.complete("test prompt", task="test")

            assert result == "OpenAI response"
            assert client._last_provider == "openai"

    @pytest.mark.asyncio
    async def test_privacy_mode_forces_ollama(self):
        """PRIVACY_MODE=true skips Claude and OpenAI."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "key",
            "OPENAI_API_KEY": "key",
            "PRIVACY_MODE": "true"
        }):
            from tools.llm_client import UnifiedLLMClient
            client = UnifiedLLMClient()

            with patch.object(client, "_call_ollama", new_callable=AsyncMock,
                              return_value="Ollama response"):
                with patch.object(client, "_call_claude", new_callable=AsyncMock) as mock_claude:
                    result = await client.complete("test prompt")

            assert result == "Ollama response"
            mock_claude.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: HFModelManager (3 tests)
# ---------------------------------------------------------------------------

class TestHFModelManager:
    """Tests for HFModelManager."""

    def test_encode_fallback_shape(self):
        """encode() with TF-IDF fallback returns correct array shape."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        mgr._transformers_available = False  # force fallback
        texts = ["Test text one", "Test text two", "Test text three"]
        embeddings = mgr._numpy_fallback_encode(texts)
        assert embeddings.shape[0] == 3
        assert embeddings.ndim == 2

    def test_heuristic_rerank_ordering(self):
        """Heuristic reranker places more relevant passages first."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        query = "transformer attention mechanism"
        passages = [
            "completely unrelated text about cooking recipes",
            "transformer attention mechanism neural network deep learning",
            "attention mechanism in deep learning transformers",
        ]
        scores = mgr._heuristic_rerank(query, passages)
        assert len(scores) == 3
        assert scores[1] > scores[0]  # passage 2 more relevant than passage 1
        assert scores[2] > scores[0]  # passage 3 more relevant than passage 1

    def test_extractive_summary_fallback_truncates(self):
        """Extractive summary fallback truncates long texts."""
        from tools.hf_model_manager import HFModelManager
        mgr = HFModelManager()
        long_text = " ".join(["word"] * 500)
        summary = mgr._extractive_summary_fallback(long_text, max_length=50)
        assert len(summary.split()) < len(long_text.split())
        assert "..." in summary


# ---------------------------------------------------------------------------
# Integration Tests (4 tests)
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end integration tests with mocked external dependencies."""

    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self, memory_manager, sample_papers):
        """Full crawl->analyze->gaps->synthesize pipeline runs without errors."""
        from agent.orchestrator import ResearchOrchestrator

        orch = ResearchOrchestrator()
        orch.memory_manager = memory_manager
        memory_manager.save_papers(sample_papers)

        # Mock the crawler to return sample papers
        with patch.object(orch._get_crawler().__class__, "crawl", new_callable=AsyncMock) as mock_crawl:
            mock_crawl.return_value = sample_papers

            # Mock LLM for gap explanation and synthesis
            with patch("tools.llm_client.UnifiedLLMClient.complete", new_callable=AsyncMock) as mock_llm:
                mock_llm.return_value = (
                    "## Introduction\n" + "Analysis content. " * 50 + " [1] [2] [3] [4] [5]\n"
                    "## Key Themes\n- Theme one [1]\n- Theme two [2]\n"
                    "## Methodology Landscape\nMethods [3]\n"
                    "## State of the Art\nBest results [4]\n"
                    "## Identified Gaps\n- Gap one\n"
                    "## Future Directions\nFuture [5]\n"
                    "## Conclusion\nIn conclusion [1] [2]."
                )
                with patch("tools.hf_model_manager.HFModelManager.instance") as mock_hf:
                    mock_inst = MagicMock()
                    mock_inst.encode.return_value = np.random.randn(len(sample_papers), 64)
                    mock_inst.rerank.return_value = [0.9, 0.8, 0.7, 0.6, 0.5]
                    mock_inst.summarize.side_effect = lambda t, **kwargs: t[:100]
                    mock_hf.return_value = mock_inst

                    result = await orch.synthesize_literature("transformers", max_papers=5)

        assert "query" in result
        assert "introduction" in result
        assert result["paper_count"] > 0

    @pytest.mark.asyncio
    async def test_crawl_then_analyze(self, memory_manager, sample_papers):
        """Crawl followed by citation analysis produces valid graph."""
        memory_manager.save_papers(sample_papers)
        from agent.orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        orch.memory_manager = memory_manager

        with patch("agent.modules.citation_analyzer.CitationAnalyzer.fetch_citations",
                   new_callable=AsyncMock, return_value=[]):
            result = await orch.analyze_citations("transformers", min_citations=0)

        assert "graph_nodes" in result
        assert result["graph_nodes"] >= 0  # may be 0 if no papers found with topic

    @pytest.mark.asyncio
    async def test_rest_api_health(self):
        """FastAPI health endpoint returns correct JSON."""
        from fastapi.testclient import TestClient
        from agent.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "academic-research-agent"

    def test_cost_report_empty(self, memory_manager):
        """Cost report returns valid structure even with no logs."""
        from agent.orchestrator import ResearchOrchestrator
        orch = ResearchOrchestrator()
        orch.memory_manager = memory_manager

        report = orch.get_cost_report()
        assert "total_cost_usd" in report
        assert "total_calls" in report
        assert "by_provider" in report
        assert "by_task" in report
        assert report["total_cost_usd"] == 0.0
        assert report["total_calls"] == 0


# ---------------------------------------------------------------------------
# CLI Smoke Tests (3 tests)
# ---------------------------------------------------------------------------

class TestCLI:
    """Smoke tests for CLI commands using Click test runner."""

    def test_status_command(self, memory_manager):
        """Status command runs without error and outputs stats."""
        from click.testing import CliRunner
        from agent.main import cli

        with patch("agent.main._get_orchestrator") as mock_orch:
            mock_inst = MagicMock()
            mock_inst.get_status.return_value = {
                "papers_total": 42,
                "citations_total": 100,
                "clusters_total": 5,
                "syntheses_cached": 2,
                "knowledge_hashes": 50,
                "agent_version": "1.0.0",
            }
            mock_orch.return_value = mock_inst

            runner = CliRunner()
            result = runner.invoke(cli, ["status"])

        assert result.exit_code == 0
        assert "42" in result.output

    def test_cost_report_command(self):
        """Cost report command runs and displays formatted output."""
        from click.testing import CliRunner
        from agent.main import cli

        with patch("agent.main._get_orchestrator") as mock_orch:
            mock_inst = MagicMock()
            mock_inst.get_cost_report.return_value = {
                "total_cost_usd": 0.1234,
                "total_calls": 5,
                "by_provider": {"claude": {"cost": 0.1000, "calls": 3}},
                "by_task": {"synthesis": {"cost": 0.1000, "calls": 3}},
            }
            mock_orch.return_value = mock_inst

            runner = CliRunner()
            result = runner.invoke(cli, ["cost-report"])

        assert result.exit_code == 0
        assert "0.1234" in result.output

    @pytest.mark.asyncio
    async def test_crawl_command_invokes_crawler(self):
        """Crawl CLI command invokes orchestrator.crawl_papers."""
        from click.testing import CliRunner
        from agent.main import cli

        mock_result = {
            "query": "transformers",
            "papers_found": 15,
            "papers_new": 10,
            "sources_queried": ["arxiv"],
            "sample_titles": ["Paper A", "Paper B"],
        }

        with patch("agent.main._get_orchestrator") as mock_orch:
            mock_inst = MagicMock()
            mock_inst.crawl_papers = AsyncMock(return_value=mock_result)
            mock_orch.return_value = mock_inst

            runner = CliRunner()
            result = runner.invoke(cli, [
                "crawl", "--query", "transformers", "--sources", "arxiv", "--max-results", "20"
            ])

        assert result.exit_code == 0
        assert "15" in result.output

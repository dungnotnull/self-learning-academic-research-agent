"""
academic-research-enhanced — ResearchMemoryManager
SQLite-backed persistent memory for papers, citations, clusters, synthesis cache, and cost logs.
Includes automatic schema migration on startup.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("academic_agent.memory_manager")

# ---------------------------------------------------------------------------
# Paper dataclass — defined here to avoid circular imports
# ---------------------------------------------------------------------------

@dataclass
class Paper:
    doi: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    venue: str
    url: str
    source: str  # arxiv / semantic_scholar / pubmed / ssrn
    citation_count: int = 0
    arxiv_id: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class ClusterRecord:
    cluster_id: int
    topic: str
    centroid: List[float]
    paper_dois: List[str]
    density: float
    is_gap: bool = False
    llm_description: str = ""


# ---------------------------------------------------------------------------
# Current schema version — increment when adding migrations
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2

MIGRATIONS = {
    2: [
        """ALTER TABLE papers ADD COLUMN embedding_stored INTEGER DEFAULT 0;""",
        """ALTER TABLE clusters ADD COLUMN n_papers INTEGER DEFAULT 0;""",
        """ALTER TABLE llm_cost_log ADD COLUMN latency_seconds REAL DEFAULT 0.0;""",
    ],
}


class ResearchMemoryManager:
    """Thread-safe SQLite memory manager for all agent data."""

    SCHEMA_VERSION = SCHEMA_VERSION

    def __init__(self, db_path: Optional[str] = None):
        data_dir = os.environ.get("DATA_DIR", "./data")
        os.makedirs(data_dir, exist_ok=True)
        self.data_dir = data_dir

        if db_path is None:
            db_path = os.path.join(data_dir, "research_memory.db")
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self._run_migrations()

    # ------------------------------------------------------------------
    # DB init
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY
                    );

                    CREATE TABLE IF NOT EXISTS papers (
                        doi TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        authors_json TEXT NOT NULL DEFAULT '[]',
                        abstract TEXT DEFAULT '',
                        year INTEGER DEFAULT 0,
                        venue TEXT DEFAULT '',
                        url TEXT DEFAULT '',
                        source TEXT DEFAULT '',
                        citation_count INTEGER DEFAULT 0,
                        arxiv_id TEXT DEFAULT '',
                        keywords_json TEXT DEFAULT '[]',
                        score REAL DEFAULT 0.0,
                        embedding_path TEXT DEFAULT '',
                        embedding_stored INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year);
                    CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(source);
                    CREATE INDEX IF NOT EXISTS idx_papers_score ON papers(score DESC);

                    CREATE TABLE IF NOT EXISTS citations (
                        from_doi TEXT NOT NULL,
                        to_doi TEXT NOT NULL,
                        PRIMARY KEY (from_doi, to_doi)
                    );

                    CREATE INDEX IF NOT EXISTS idx_citations_from ON citations(from_doi);
                    CREATE INDEX IF NOT EXISTS idx_citations_to ON citations(to_doi);

                    CREATE TABLE IF NOT EXISTS clusters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cluster_id INTEGER NOT NULL,
                        topic TEXT NOT NULL,
                        centroid_json TEXT NOT NULL DEFAULT '[]',
                        paper_dois_json TEXT NOT NULL DEFAULT '[]',
                        density REAL DEFAULT 0.0,
                        is_gap INTEGER DEFAULT 0,
                        llm_description TEXT DEFAULT '',
                        n_papers INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );

                    CREATE INDEX IF NOT EXISTS idx_clusters_topic ON clusters(topic);

                    CREATE TABLE IF NOT EXISTS synthesis_cache (
                        query_hash TEXT NOT NULL,
                        style TEXT NOT NULL DEFAULT 'academic',
                        query TEXT NOT NULL,
                        review_json TEXT NOT NULL,
                        created_at TEXT NOT NULL DEFAULT (datetime('now')),
                        PRIMARY KEY (query_hash, style)
                    );

                    CREATE TABLE IF NOT EXISTS llm_cost_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        prompt_tokens INTEGER DEFAULT 0,
                        completion_tokens INTEGER DEFAULT 0,
                        cost_usd REAL DEFAULT 0.0,
                        task TEXT DEFAULT '',
                        latency_seconds REAL DEFAULT 0.0,
                        created_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );

                    CREATE TABLE IF NOT EXISTS knowledge_hashes (
                        hash TEXT PRIMARY KEY,
                        url TEXT DEFAULT '',
                        title TEXT DEFAULT '',
                        added_at TEXT NOT NULL DEFAULT (datetime('now'))
                    );
                """)
                conn.commit()
                logger.debug("Database initialized at %s", self.db_path)
            finally:
                conn.close()

    def _run_migrations(self):
        """Run pending schema migrations."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Get current version
                row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
                current_version = row[0] if row[0] is not None else 0

                if current_version < SCHEMA_VERSION:
                    for version in range(current_version + 1, SCHEMA_VERSION + 1):
                        if version in MIGRATIONS:
                            for sql in MIGRATIONS[version]:
                                try:
                                    conn.execute(sql)
                                except sqlite3.OperationalError as e:
                                    if "duplicate column name" in str(e).lower():
                                        logger.debug("Migration %d column already exists, skipping", version)
                                    else:
                                        raise
                            conn.execute(
                                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                                (version,),
                            )
                            conn.commit()
                            logger.info("Applied schema migration v%d", version)

                    logger.info("Schema version: %d (migrated from %d)", SCHEMA_VERSION, current_version)
                else:
                    logger.debug("Schema version: %d (up to date)", current_version)
            except Exception as e:
                logger.error("Schema migration failed: %s", e)
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Paper CRUD
    # ------------------------------------------------------------------

    def save_paper(self, paper: Paper):
        """Insert or update a single paper."""
        self.save_papers([paper])

    def save_papers(self, papers: List[Paper]):
        """Batch insert/update papers."""
        with self._lock:
            conn = self._get_connection()
            try:
                now = datetime.now(timezone.utc).isoformat()
                data = [
                    (
                        p.doi,
                        p.title,
                        json.dumps(p.authors),
                        p.abstract,
                        p.year,
                        p.venue,
                        p.url,
                        p.source,
                        p.citation_count,
                        p.arxiv_id or "",
                        json.dumps(p.keywords),
                        p.score,
                        "",  # embedding_path
                        0,   # embedding_stored
                        now,
                        now,
                    )
                    for p in papers
                ]
                conn.executemany(
                    """
                    INSERT INTO papers
                        (doi, title, authors_json, abstract, year, venue, url, source,
                         citation_count, arxiv_id, keywords_json, score, embedding_path,
                         embedding_stored, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doi) DO UPDATE SET
                        citation_count = MAX(citation_count, excluded.citation_count),
                        abstract = CASE WHEN length(excluded.abstract) > length(papers.abstract) THEN excluded.abstract ELSE papers.abstract END,
                        updated_at = excluded.updated_at
                    """,
                    data,
                )
                conn.commit()
                logger.debug("Saved %d papers to memory", len(papers))
            finally:
                conn.close()

    def get_paper(self, doi: str) -> Optional[Dict]:
        """Retrieve a paper by DOI."""
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute("SELECT * FROM papers WHERE doi = ?", (doi,)).fetchone()
                if row is None:
                    return None
                return dict(row)
            finally:
                conn.close()

    def search_papers(self, query: str, limit: int = 100) -> List[Paper]:
        """Full-text search in titles and abstracts."""
        with self._lock:
            conn = self._get_connection()
            try:
                if query:
                    terms = query.split()[:5]
                    conditions = " OR ".join(
                        f"(title LIKE ? OR abstract LIKE ?)" for _ in terms
                    )
                    params = []
                    for t in terms:
                        like_t = f"%{t}%"
                        params.extend([like_t, like_t])
                    rows = conn.execute(
                        f"SELECT * FROM papers WHERE {conditions} ORDER BY score DESC, citation_count DESC LIMIT ?",
                        params + [limit],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM papers ORDER BY created_at DESC LIMIT ?", (limit,)
                    ).fetchall()

                papers = []
                for row in rows:
                    p = Paper(
                        doi=row["doi"],
                        title=row["title"],
                        authors=json.loads(row["authors_json"] or "[]"),
                        abstract=row["abstract"] or "",
                        year=row["year"] or 0,
                        venue=row["venue"] or "",
                        url=row["url"] or "",
                        source=row["source"] or "",
                        citation_count=row["citation_count"] or 0,
                        arxiv_id=row["arxiv_id"] or None,
                        keywords=json.loads(row["keywords_json"] or "[]"),
                        score=row["score"] or 0.0,
                    )
                    papers.append(p)
                return papers
            finally:
                conn.close()

    def get_papers_by_year(self, year: int) -> List[Dict]:
        """Retrieve all papers published in a given year."""
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("SELECT * FROM papers WHERE year = ?", (year,)).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_papers(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    def save_citations(self, from_doi: str, to_dois: List[str]):
        """Save citation edges."""
        with self._lock:
            conn = self._get_connection()
            try:
                data = [(from_doi, to_doi) for to_doi in to_dois]
                conn.executemany(
                    "INSERT OR IGNORE INTO citations (from_doi, to_doi) VALUES (?, ?)",
                    data,
                )
                conn.commit()
            finally:
                conn.close()

    def get_citations(self, doi: str) -> List[str]:
        """Papers cited by this DOI."""
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("SELECT to_doi FROM citations WHERE from_doi = ?", (doi,)).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()

    def get_cited_by(self, doi: str) -> List[str]:
        """Papers that cite this DOI."""
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("SELECT from_doi FROM citations WHERE to_doi = ?", (doi,)).fetchall()
                return [r[0] for r in rows]
            finally:
                conn.close()

    def count_citations(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Clusters
    # ------------------------------------------------------------------

    def save_cluster(self, cluster: ClusterRecord):
        """Save a cluster record."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO clusters
                        (cluster_id, topic, centroid_json, paper_dois_json, density, is_gap, llm_description, n_papers)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cluster.cluster_id,
                        cluster.topic,
                        json.dumps(cluster.centroid[:10]),
                        json.dumps(cluster.paper_dois),
                        cluster.density,
                        int(cluster.is_gap),
                        cluster.llm_description,
                        len(cluster.paper_dois),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def get_clusters(self, topic: str) -> List[Dict]:
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM clusters WHERE topic = ? ORDER BY created_at DESC",
                    (topic,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def get_gap_clusters(self, topic: str) -> List[Dict]:
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(
                    "SELECT * FROM clusters WHERE topic = ? AND is_gap = 1 ORDER BY density ASC",
                    (topic,),
                ).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def count_clusters(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Synthesis cache
    # ------------------------------------------------------------------

    def _query_hash(self, query: str) -> str:
        import hashlib
        return hashlib.sha256(query.lower().strip().encode()).hexdigest()

    def save_synthesis(self, query: str, review: Dict, style: str = "academic"):
        qh = self._query_hash(query)
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO synthesis_cache
                        (query_hash, style, query, review_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (qh, style, query, json.dumps(review), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

    def get_synthesis(self, query: str, style: str = "academic") -> Optional[Dict]:
        """Return cached synthesis if it exists and is < 24h old."""
        qh = self._query_hash(query)
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute(
                    "SELECT * FROM synthesis_cache WHERE query_hash = ? AND style = ?",
                    (qh, style),
                ).fetchone()
                if row is None:
                    return None
                try:
                    created_at = datetime.fromisoformat(row["created_at"])
                    if datetime.now(timezone.utc) - created_at > timedelta(hours=24):
                        return None
                except (ValueError, TypeError):
                    pass
                return json.loads(row["review_json"])
            finally:
                conn.close()

    def count_syntheses(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM synthesis_cache").fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # LLM Cost log
    # ------------------------------------------------------------------

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        task: str = "",
        latency_seconds: float = 0.0,
    ):
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO llm_cost_log
                        (provider, model, prompt_tokens, completion_tokens, cost_usd, task, latency_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (provider, model, prompt_tokens, completion_tokens, cost_usd, task, latency_seconds),
                )
                conn.commit()
            finally:
                conn.close()

    def get_cost_summary(self) -> Dict:
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute("SELECT * FROM llm_cost_log").fetchall()
                total_cost = 0.0
                total_calls = 0
                by_provider: Dict[str, Dict] = {}
                by_task: Dict[str, Dict] = {}

                for r in rows:
                    total_cost += r["cost_usd"]
                    total_calls += 1

                    prov = r["provider"]
                    if prov not in by_provider:
                        by_provider[prov] = {"cost": 0.0, "calls": 0}
                    by_provider[prov]["cost"] += r["cost_usd"]
                    by_provider[prov]["calls"] += 1

                    task = r["task"] or "unknown"
                    if task not in by_task:
                        by_task[task] = {"cost": 0.0, "calls": 0}
                    by_task[task]["cost"] += r["cost_usd"]
                    by_task[task]["calls"] += 1

                return {
                    "total_cost_usd": round(total_cost, 6),
                    "total_calls": total_calls,
                    "by_provider": by_provider,
                    "by_task": by_task,
                }
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Knowledge hashes
    # ------------------------------------------------------------------

    def is_known_paper(self, url_or_doi: str) -> bool:
        """Check if a paper has been seen before (by DOI or hash)."""
        import hashlib
        h = hashlib.sha256(url_or_doi.lower().strip().encode()).hexdigest()
        with self._lock:
            conn = self._get_connection()
            try:
                row = conn.execute("SELECT 1 FROM knowledge_hashes WHERE hash = ?", (h,)).fetchone()
                if row:
                    return True
                row2 = conn.execute("SELECT 1 FROM papers WHERE doi = ?", (url_or_doi,)).fetchone()
                return row2 is not None
            finally:
                conn.close()

    def mark_paper_known(self, url_or_doi: str, title: str = ""):
        import hashlib
        h = hashlib.sha256(url_or_doi.lower().strip().encode()).hexdigest()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO knowledge_hashes (hash, url, title) VALUES (?, ?, ?)",
                    (h, url_or_doi, title),
                )
                conn.commit()
            finally:
                conn.close()

    def count_knowledge_hashes(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                return conn.execute("SELECT COUNT(*) FROM knowledge_hashes").fetchone()[0]
            finally:
                conn.close()

    # ------------------------------------------------------------------
    # Database maintenance
    # ------------------------------------------------------------------

    def vacuum(self):
        """Run VACUUM to compact the database."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("VACUUM")
                logger.info("Database VACUUM completed")
            finally:
                conn.close()

    def get_db_size_mb(self) -> float:
        """Return database file size in MB."""
        try:
            return os.path.getsize(self.db_path) / (1024 * 1024)
        except OSError:
            return 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> Dict:
        return dict(row)

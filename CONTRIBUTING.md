# Contributing to academic-research-enhanced

Thank you for your interest in improving this project! Contributions of all kinds are welcome.

## How to Contribute

### Bug Reports

1. Search existing [GitHub Issues](../../issues) to avoid duplicates.
2. Open a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version, OS, and relevant environment details

### Feature Requests

1. Open an issue with the enhancement label.
2. Describe the use case, expected behavior, and why it matters.
3. If you plan to implement it yourself, mention that in the issue.

### Pull Requests

1. **Fork** the repository and create a feature branch from main:
   `ash
   git checkout -b feature/my-feature
   `
2. **Install** development dependencies:
   `ash
   pip install -r requirements.txt
   pip install pytest pytest-asyncio pytest-mock ruff
   `
3. **Write code** that follows the existing style:
   - Use type hints for all function signatures
   - Use logging module, never print() in library code
   - Add docstrings to all public classes and methods
   - Keep functions focused — one responsibility per function
4. **Test** your changes:
   `ash
   pytest tests/ -v
   `
5. **Run linting**:
   `ash
   ruff check agent/ tools/ --fix
   `
6. **Commit** with clear messages:
   `ash
   git commit -m "feat: add new embedding endpoint for batch processing"
   `
7. **Push** and open a Pull Request against main.

### Commit Message Convention

| Prefix | Usage |
|--------|-------|
| eat: | New feature |
| ix: | Bug fix |
| docs: | Documentation change |
| efactor: | Code restructuring without behavior change |
| 	est: | Adding or updating tests |
| chore: | Build, CI, or tooling changes |

## Code Style

- **Python 3.12** target
- Type hints on all function signatures
- uff for formatting and linting
- Max line length: 120 characters
- Use sync/await for all I/O-bound operations
- Use dataclasses for structured data, not raw dicts
- Every module gets a module-level docstring

## Adding a New Paper Source

To add a new crawl source (e.g., CrossRef, OpenAlex):

1. Add an async method _crawl_<source> to PaperCrawler in gent/modules/paper_crawler.py
2. Return a List[Paper] with proper deduplication hash
3. Add the source name to the sources parameter in crawl()
4. Add a test in 	ests/test_agent.py
5. Update config/agent_config.yaml and SECOND-KNOWLEDGE-BRAIN.md

## Adding a New LLM Provider

1. Add provider detection in UnifiedLLMClient.__init__
2. Add an async _call_<provider> method
3. Add cost rates to COST_PER_1K
4. Add the provider to the fallback chain in complete()
5. Test with mocked API responses

## Questions?

Open a [GitHub Discussion](../../discussions) or ask in an issue with the question label.

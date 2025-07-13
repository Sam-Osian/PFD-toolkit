# Contributing to PFD Toolkit

Welcome to the PFD Toolkit contributor guide. It provides a walkthrough of the repository structure, explains the
purpose of each major module and outlines how to set up a development environment.

---

## Repository layout

```
PFD-toolkit/
├── README.md              - High-level project summary and install instructions
├── LICENCE                - AGPL-3.0 licence text
├── pyproject.toml         - Package metadata and dependencies
├── src/                   - Source code for the `pfd_toolkit` package
├── tests/                 - Automated unit tests
├── docs/                  - MkDocs documentation site
├── tutorials/             - Jupyter notebooks demonstrating usage
├── scripts/               - Helper scripts for scraping and dataset updates
└── ...
```

Only the `src/` directory is installed as the Python package. Everything else provides tooling or documentation to aid development.

---

## `src/pfd_toolkit` package

This folder contains all runtime functionality. The modules work together to scrape, clean and analyse Prevention of Future Deaths (PFD) reports.

### Package entry point

- **`__init__.py`** – Exposes the main classes (`Scraper`, `Cleaner`, `LLM`, `Screener`, `Extractor`) and configuration objects. Importing from `pfd_toolkit` gives direct access to these helpers.

### Core modules

- **`loader.py`** – Downloads the latest dataset from GitHub releases and caches it locally. The `load_reports()` function filters by date and returns a pandas DataFrame.
- **`scraper/`** – Subpackage responsible for turning raw judiciary web pages into structured rows. It contains:
  - `scraper.py` with the `Scraper` class orchestrating HTML parsing, PDF fallback and optional LLM extraction.
  - `html_extractor.py` and `pdf_extractor.py` which handle low-level parsing logic for web pages and PDFs.
  - `llm_extractor.py` which is used when HTML/PDF extraction fails and the text is delegated to a language model.
- **`cleaner.py`** – Uses the `LLM` helper to tidy extracted fields. Prompts are provided for each column and results are written back to a new DataFrame.
- **`screener.py`** – Filters reports by topic. It builds an LLM prompt from user queries and flags reports that match.
- **`extractor.py`** – Performs custom feature extraction using an LLM, optionally validating responses with pydantic models.
- **`llm.py`** – Wrapper around the OpenAI SDK. Handles batching, throttling and conversion of PDF files to images when vision models are needed.
- **`config.py`** – Central place for constants. `GeneralConfig` defines column names and placeholders. `ScraperConfig` stores all network and prompt settings used by `Scraper`.
- **`text_utils.py`** – Shared text-processing helpers (date parsing, whitespace cleanup, etc.).

### Data files

The CSV containing live PFD report data is not held inside the repo itself, but rather the `dataset-latest` GitHub asset.

---

## Dependencies

We use [`uv`](https://docs.astral.sh/uv/getting-started/installation/) for depependency management. Once installed, load dependencies with:

```bash
uv sync --dev
```

`uv` is used for all CLI operations that rely on project dependencies. 

For example, to add new dependencies, replace `pip install` with:

```bash
uv add <package-name>
```

For everything else, append `uv run` to the start of each line. For example, to serve project documentation:

```bash
uv run mkdocs serve
```

...And so on.

---

## Tests

Unit tests live in the `tests/` directory and cover the major components:

- Scraper behaviour and extractor utilities
- LLM wrapper methods
- Cleaning and screening pipelines
- Config defaults and text utilities

Tests use `pytest` with coverage reports enabled. Fixtures in `tests/fixtures/` provide very minimal sample data for deterministic testing.

To run the tests locally, install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and run:

```bash
uv sync --dev   # install dependencies (requires `uv`)
uv run pytest
```

---

## GitHub workflows

Continuous integration runs automatically via GitHub Actions:

- **test.yml** – runs the `pytest` suite on every pull request.
- **uv.yml** – verifies package builds across Python versions.
- **pfd_update.yml** – keeps the cached dataset fresh by running a scheduled scrape.

---

## Development workflow

1. **Install dependencies** – The project relies on [uv](https://github.com/astral-sh/uv) for reproducible environments. Run `uv sync --dev` to create a virtual environment with all runtime and developer packages. Alternatively you can install packages from `pyproject.toml` using `pip`.
2. **Create a branch** – Develop new features or fixes on a feature branch.
3. **Run tests** – Use `pytest` to ensure existing functionality remains stable. Add new tests for any new behaviour.
4. **Style checks** – `ruff` is included in the dev dependencies. Run `ruff check .` to lint and `ruff format .` to apply formatting.
5. **Open a pull request** – Open a pull request on GitHub, describing your changes and including steps to reproduce the issue or feature, if relevant. 


---

## Editing tips

- Keep modules focused. For example, scraping logic stays under `scraper/` while text processing utilities belong in `text_utils.py`.
- Try to use the configuration objects in `config.py` rather than hard-coding values, particularly where these values are used cross-module. This keeps defaults consistent across the package.
- Unit tests provide concrete examples of how each class is expected to behave. Reviewing them may be the quickest way to understand the code.

### Commit messages

Write short, focused commit messages that list the concrete changes made.
Each commit should explain what was changed and why. 

---

## Further resources

The `docs/` folder contains user-facing documentation built with MkDocs. While this overview lives outside the documentation site, you may still find additional context there (tutorials, API reference, contribution guidelines).

If you run into problems or have questions about the architecture, open an issue on GitHub or ping [@sam-osian](https://github.com/sam-osian) for guidance.

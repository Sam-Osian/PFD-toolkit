# PFD-toolkit

A Python toolkit for using cleaned **Prevention of Future Death (PFD) reports** from the UK Judiciary website.

## Features

- Load ready-to-use and fully-cleaned PFD report datasets - all updated once a week.
- Query reports to find matches for your specific research questions.
- Create custom categorisation systems for tailored data curation.
- Generate concise summaries of reports.
- Call a web scraper for custom data collection.


## Installation

```bash
pip install pfd-toolkit
```

If you need .docx -> .pdf conversion, install with

```bash
pip install pfd-toolkit[docx-conversion]
```

## UV (Package Manager)
### Installation
1.  Download and install UV using [this guide](https://docs.astral.sh/uv/getting-started/installation/).

2. Add UV to Path.
    - Windows Powershell - `$env:Path = "C:\Users\jonat\.local\bin;$env:Path"`.
    - Linux - `ehhh dunno`.

3. Install required Python version using `uv python install 3.12.3`.

4. Install pfd-toolkit using `uv sync`.

5. Activate uv environment.
    - Linux - `source .venv/bin/activate`.
    - Windows Powershell - `.venv\Scripts\Activate.ps1`.

6. When adding dependencies or modifying the pyproject.toml file, it's advisable to run `uv sync` to ensure the uv.lock file is up to date.
### Usage
1. Add dependencies with `uv add {package_name}` eg: `uv add pandas`, `uv add pandas==2.0.1`.
2. Remove dependencies with `uv remove {package_name}` eg: `uv remove pandas`.
3. Update packages using add command but use `==` to specify version.

## License
This project is licensed under the terms of the MIT Licence. For more information, see `LICENSE`.
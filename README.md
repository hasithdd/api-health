
# Cybersecurity Threat Detection EDA

This repo contains an exploratory data analysis notebook `eda.ipynb` for the dataset `cybersecurity_threat_detection_logs.csv`.

## Prerequisites

- OS: Linux (commands below assume `bash`)
- `git`
- `uv` (Python package manager)

The project is configured for Python **3.13** (see `.python-version` and `pyproject.toml`).

## 1) Install `uv`

### Option A (recommended): Official installer

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your shell (close/open terminal), or load your shell rc file.

Verify:

```bash
uv --version
```

### Option B: With pipx (if you already use it)

```bash
pipx install uv
uv --version
```

## 2) Set up the environment (`uv sync`)

From the repo root:

```bash
cd /home/hasith/Personal/cybersec
uv sync
```

Notes:

- `uv sync` creates/updates the local virtual environment (typically `.venv/`) using `uv.lock`.
- For fully reproducible installs (fail if the lock can’t be honored), use:

```bash
uv sync --frozen
```

### If Python 3.13 is missing

If you don’t have Python 3.13 available locally, install it via uv:

```bash
uv python install 3.13
uv sync
```

## 3) Run `eda.ipynb`

The notebook reads the dataset with a *relative path*:

```python
pl.read_csv("cybersecurity_threat_detection_logs.csv")
```

So you must run Jupyter with the working directory set to the repo root (the folder containing the CSV).

### Option A (VS Code) — easiest

1. Open this folder in VS Code.
2. Install the VS Code extensions:
	- **Python**
	- **Jupyter**
3. Select the interpreter from the project venv:
	- Command Palette → **Python: Select Interpreter** → pick `.venv/bin/python`.
4. Open `eda.ipynb`.
5. In the notebook kernel selector (top-right), choose the same `.venv` Python.
6. Run cells (or **Run All**).

If VS Code doesn’t show the `.venv` kernel, run this once:

```bash
uv run python -m ipykernel install --user --name cybersec --display-name "cybersec (.venv)"
```

Then re-open the notebook and pick `cybersec (.venv)` as the kernel.

### Option B (Terminal) — JupyterLab / Notebook

This project includes `ipykernel` but does **not** include `jupyterlab`/`notebook` by default. Pick one of these approaches:

#### B1) Install JupyterLab into the project (persistent)

```bash
uv add jupyterlab
uv sync
uv run jupyter lab
```

Then open `eda.ipynb` in the browser UI.

#### B2) Install JupyterLab as a `uv` tool (doesn’t modify project deps)

```bash
uv tool install jupyterlab
uv tool run jupyter lab --notebook-dir .
```

If the notebook can’t find your project packages in this mode, register the `.venv` kernel first:

```bash
uv run python -m ipykernel install --user --name cybersec --display-name "cybersec (.venv)"
```

## Troubleshooting

### `FileNotFoundError: cybersecurity_threat_detection_logs.csv`

- Ensure you launched Jupyter from the repo root:

```bash
pwd
ls -1 cybersecurity_threat_detection_logs.csv eda.ipynb
```

### `ModuleNotFoundError: polars/pandas/seaborn/scipy/...`

- Your kernel is probably not using the project environment.
- In VS Code: re-select `.venv/bin/python` as interpreter and kernel.
- In terminal: run with `uv run ...` so it uses the `.venv`:

```bash
uv run python -c "import polars, pandas, seaborn, scipy; print('OK')"
```

### Kernel doesn’t appear in VS Code/Jupyter

Register it:

```bash
uv run python -m ipykernel install --user --name cybersec --display-name "cybersec (.venv)"
```

## What’s in the notebook

- Loads `cybersecurity_threat_detection_logs.csv`
- Summarizes fields, checks for missing dates
- Engineers categorical features (`source_ip_type`, `user_agent_type`, etc.)
- Performs association analysis (Cramér’s V) and label leakage checks


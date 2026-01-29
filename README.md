# Cybersecurity Threat Detection Pipeline

A complete machine learning pipeline for detecting cybersecurity threats (benign, suspicious, malicious) from network logs using Polars for data processing and scikit-learn/XGBoost for classification.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [What is `uv`?](#what-is-uv)
4. [Installation](#installation)
5. [Understanding `pyproject.toml`](#understanding-pyprojecttoml)
6. [Package Management with `uv`](#package-management-with-uv)
7. [Repository Structure](#repository-structure)
8. [Pipeline Architecture](#pipeline-architecture)
9. [Running the Pipeline](#running-the-pipeline)
10. [Using Notebooks](#using-notebooks)
11. [Configuration & Parameters](#configuration--parameters)
12. [CPU/GPU Configuration](#cpugpu-configuration)
13. [Inference API](#inference-api)
14. [Deployment](#deployment)
15. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a **security-focused threat detection system** that:

- Processes network security logs using **Polars** (lazy evaluation for memory efficiency)
- Engineers 24 behavioral and structural features from raw log data
- Trains and compares 3 models: **Logistic Regression**, **Random Forest**, **XGBoost**
- Prioritizes **malicious recall** over accuracy (security-first approach)
- Supports both **CPU** and **GPU** (CUDA) training

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Polars over Pandas** | 10-100x faster, lazy evaluation, memory efficient for large logs |
| **Stratified splitting** | Preserves class distribution (threat types are imbalanced) |
| **Custom class weights** | Inversely proportional to frequency - prevents majority class dominance |
| **Security-first metrics** | Malicious recall > Suspicious recall > Macro F1 |
| **Modified Huber loss** | Robust to outliers in SGDClassifier (logistic regression) |

---

## Prerequisites

- **OS**: Linux (commands assume `bash`)
- **Python**: 3.13+ (auto-installed by `uv` if missing)
- **Git**: For version control
- **CUDA** (optional): For GPU-accelerated XGBoost training

---

## What is `uv`?

[`uv`](https://docs.astral.sh/uv/) is a **blazing-fast Python package manager** written in Rust by Astral (creators of Ruff). It replaces `pip`, `pip-tools`, `virtualenv`, `conda`, and `pyenv` with a single tool.

### Why `uv` over `pip`/`conda`?

| Feature | `pip` | `conda` | `uv` |
|---------|-------|---------|------|
| Speed | Slow | Very slow | **10-100x faster** |
| Lock file | ❌ | ❌ | ✅ `uv.lock` |
| Reproducibility | Poor | Moderate | **Exact** |
| Python management | ❌ | ✅ | ✅ |
| Virtual env | Manual | Automatic | **Automatic** |
| Disk space | High | Very high | **Low** |

---

## Installation

### Step 1: Install `uv`

#### Option A: Official Installer (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal, then verify:

```bash
uv --version
```

#### Option B: Using pipx

```bash
pipx install uv
```

#### Option C: Using Homebrew (macOS/Linux)

```bash
brew install uv
```

### Step 2: Clone and Setup

```bash
git clone <your-repo-url>
cd cybersec

# Sync environment (creates .venv/ and installs all dependencies)
uv sync
```

If Python 3.13 is not installed:

```bash
uv python install 3.13
uv sync
```

### Step 3: Verify Installation

```bash
uv run python -c "import polars, sklearn, xgboost; print('✅ All packages installed!')"
```

---

## Understanding `pyproject.toml`

The `pyproject.toml` is the **modern Python project configuration file** (PEP 518/621). It replaces `setup.py`, `requirements.txt`, and `setup.cfg`.

```toml
[project]
name = "cybersec"
version = "0.1.0"
description = "Cybersecurity threat detection pipeline"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "ipykernel>=7.1.0",      # Jupyter kernel support
    "joblib>=1.5.3",         # Parallel processing
    "matplotlib>=3.10.8",    # Plotting
    "polars>=1.36.1",        # Fast DataFrame library
    "scikit-learn>=1.5.0",   # ML algorithms
    "xgboost>=3.1.3",        # Gradient boosting
    # ... more dependencies
]
```

### Key Sections

| Section | Purpose |
|---------|---------|
| `[project]` | Project metadata (name, version, Python version) |
| `dependencies` | Runtime packages (like `requirements.txt`) |
| `[project.optional-dependencies]` | Dev/test dependencies |
| `[tool.uv]` | uv-specific configuration |

---

## Package Management with `uv`

### Adding Packages

```bash
# Equivalent to: pip install pandas
uv add pandas

# Add with version constraint
uv add "pandas>=2.0"

# Add development dependency
uv add --dev pytest
```

### Removing Packages

```bash
uv remove pandas
```

### Comparison with pip/conda

| Task | pip | conda | uv |
|------|-----|-------|-----|
| Install package | `pip install pkg` | `conda install pkg` | `uv add pkg` |
| Install from file | `pip install -r requirements.txt` | `conda env create -f env.yml` | `uv sync` |
| Create lock file | `pip freeze > requirements.txt` | ❌ | `uv lock` (automatic) |
| Upgrade package | `pip install --upgrade pkg` | `conda update pkg` | `uv add pkg@latest` |

### Syncing Environment

```bash
# Install all dependencies from uv.lock
uv sync

# Frozen install (fail if lock file needs update)
uv sync --frozen

# Update all packages
uv sync --upgrade
```

---

## Repository Structure

```
cybersec/
├── main.py                    # 🚀 Entry point - orchestrates the full pipeline
├── pyproject.toml             # 📦 Project config & dependencies
├── uv.lock                    # 🔒 Locked dependency versions
├── README.md                  # 📖 This file
│
├── data/
│   └── cybersecurity_threat_detection_logs.csv  # 📊 Raw dataset
│
├── src/
│   ├── __init__.py
│   ├── preparation.py         # 🔧 Data loading, feature engineering, splitting
│   ├── model.py               # 🧠 Model builders (LR, RF, XGBoost)
│   ├── train.py               # 🏋️ Training loop & experiment management
│   ├── evaluation.py          # 📈 Metrics, ranking, model selection
│   ├── validation.py          # 📉 Visualization & overfitting detection
│   └── tuning.py              # 🎯 Hyperparameter optimization
│
├── notebooks/
│   ├── eda.ipynb              # 🔍 Exploratory data analysis
│   ├── train.ipynb            # 📓 Interactive training notebook (full pipeline)
│   └── pipeline.ipynb         # 🔄 Step-by-step pipeline walkthrough
│
├── artifacts/                 # 💾 Preprocessed data (auto-generated)
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── feature_columns.pkl
│
├── models/                    # 🏆 Trained models (auto-generated)
│   ├── best_model.pkl
│   └── final_metrics.pkl
│
├── logs/
│   └── log.txt                # 📝 Training run history
│
├── validation/                # 📊 Plots (auto-generated)
│   ├── confusion_matrix_*.png
│   ├── precision_recall_*.png
│   └── feature_importance.png
│
└── tests/
    └── test_preparation.py    # ✅ Unit tests
```

---

## Pipeline Architecture

### Stage 1: Data Preparation (`src/preparation.py`)

```
Raw CSV → Lazy Loading → Column Selection → Type Casting → Stratified Split → Feature Engineering → Artifacts
```

**Features Engineered (24 total):**

| Category | Features | Logic |
|----------|----------|-------|
| **IP Flags** | `is_internal_source_ip` | `192.168.*` → internal |
| **Protocol Flags** | `tcp_flag`, `udp_flag`, `icmp_flag`, `ftp_flag`, `http_flag`, `https_flag`, `ssh_flag` | One-hot encoding |
| **Log Type Flags** | `firewall_log`, `application_log`, `ids_log` | One-hot encoding |
| **Numeric** | `bytes_transferred_scaled` | Z-score normalization (computed on train set) |
| **User Agent Flags** | `curl_flag`, `windows_browser_flag`, `mac_browser_flag`, `nmap_script_flag`, `sqlmap_flag` | Keyword detection |
| **Path Structure** | `question_mark_count`, `dotdot_count`, `backslash_count`, `admin_keyword_count`, `passwd_keyword_count`, `bin_bash_keyword_count`, `root_keyword_count`, `hydra_keyword_count` | Pattern counting |

**Target Encoding:**
- `benign` → 0
- `suspicious` → 1
- `malicious` → 2

### Stage 2: Model Building (`src/model.py`)

Three models with security-optimized configurations:

| Model | Implementation | Key Parameters |
|-------|----------------|----------------|
| **Logistic Regression** | `SGDClassifier(loss="modified_huber")` | Robust to outliers, probabilistic output |
| **Random Forest** | `RandomForestClassifier` | `n_estimators=300`, handles imbalance via `class_weight` |
| **XGBoost** | `XGBClassifier` | `n_estimators=400`, GPU support, histogram-based |

### Stage 3: Training (`src/train.py`)

```
Load Artifacts → Compute Class Weights → Train All Models → Evaluate on Validation → Rank Models → Select Best → Final Test Evaluation → Save
```

**Class Weight Calculation:**
```python
weight[class] = total_samples / (num_classes × class_count)
```

### Stage 4: Evaluation (`src/evaluation.py`)

**Security-First Ranking (in order of priority):**
1. **Malicious Recall** - Catching all attacks is critical
2. **Suspicious Recall** - Early warning detection
3. **Macro F1** - Overall balanced performance

### Stage 5: Validation (`src/validation.py`)

Generates diagnostic plots:
- Confusion matrices (train/val/test)
- Precision-Recall curves (for suspicious & malicious classes)
- ROC curves with AUC
- Feature importance (tree-based models)

---

## Running the Pipeline

### Option 1: Full Pipeline via `main.py` (Recommended)

```bash
# Run the complete pipeline
uv run python main.py
```

This will:
1. Check for existing artifacts (skip preparation if found)
2. Run training on all 3 models
3. Evaluate and rank models
4. Save best model to `models/best_model.pkl`
5. Log results to `logs/log.txt`

### Option 2: Run Individual Stages

```bash
# Step 1: Data preparation only
uv run python -c "from src.preparation import main; main()"

# Step 2: Training only (requires artifacts)
uv run python -c "from src.train import main; main()"
```

### Option 3: Interactive Notebook

Open `notebooks/train.ipynb` for step-by-step execution with outputs.

---

## Using Notebooks

### VS Code (Recommended)

1. Open the project folder in VS Code
2. Install extensions: **Python**, **Jupyter**
3. Select interpreter: Command Palette → `Python: Select Interpreter` → `.venv/bin/python`
4. Open any `.ipynb` file
5. Select kernel (top-right) → `.venv` Python
6. Run cells!

### If Kernel Doesn't Appear

```bash
uv run python -m ipykernel install --user --name cybersec --display-name "cybersec (.venv)"
```

### JupyterLab (Terminal)

```bash
# Add JupyterLab to project
uv add jupyterlab

# Launch
uv run jupyter lab
```

### Notebook Files

| Notebook | Purpose |
|----------|---------|
| `notebooks/eda.ipynb` | Exploratory analysis, feature distributions, correlation |
| `notebooks/train.ipynb` | **Complete pipeline in one notebook** - data prep → training → evaluation |
| `notebooks/pipeline.ipynb` | Step-by-step walkthrough with explanations |

---

## Configuration & Parameters

### Training Parameters (`src/train.py`)

```python
RANDOM_STATE = 42  # Reproducibility seed

experiments = {
    "logistic_regression": {
        "class_weight": class_weight,      # Computed from training data
        "random_state": RANDOM_STATE,
        "penalty": "l2",                   # Ridge regularization
    },
    "random_forest": {
        "n_estimators": 200,               # Number of trees
        "class_weight": class_weight,
        "random_state": RANDOM_STATE,
    },
    "gradient_boosting": {
        "n_estimators": 400,               # Boosting rounds
        "class_weight": class_weight,
        "device": "gpu",                   # or "cpu"
        "random_state": RANDOM_STATE,
    },
}
```

### Model Parameters (`src/model.py`)

#### Logistic Regression (SGDClassifier)
```python
SGDClassifier(
    loss="modified_huber",  # Smooth hinge loss with probability estimates
    penalty="l2",           # Ridge regularization
    alpha=0.0001,           # Regularization strength
    max_iter=2000,          # Max epochs
    n_jobs=13,              # Parallel threads
)
```

#### Random Forest
```python
RandomForestClassifier(
    n_estimators=300,       # Number of trees
    max_depth=None,         # No limit (grow until pure)
    n_jobs=13,              # Parallel threads
)
```

#### XGBoost
```python
XGBClassifier(
    n_estimators=400,       # Boosting rounds
    learning_rate=0.05,     # Step size shrinkage
    max_depth=6,            # Tree depth
    tree_method="hist",     # Fast histogram-based
    device="cuda",          # GPU acceleration
    n_jobs=13,              # CPU threads (for data loading)
)
```

---

## CPU/GPU Configuration

### Adjusting `n_jobs` for Your CPU

The `n_jobs` parameter controls parallel processing. Set it based on your CPU cores:

```bash
# Check your CPU cores
nproc  # Linux
```

**Recommendation:** Use `n_jobs = num_cores - 1` to leave one core for system tasks.

#### Modify in Code

Edit `src/model.py`:

```python
# For a 16-core machine
n_jobs = 15  # 16 - 1

# Or dynamically
import os
n_jobs = os.cpu_count() - 1
```

### CPU-Only Training

To force CPU training for XGBoost:

**Option 1:** Edit `src/train.py`
```python
"gradient_boosting": {
    ...
    "device": "cpu",  # Change from "gpu"
}
```

**Option 2:** Edit `src/model.py`
```python
def build_gradient_boosting(config: Dict[str, Any]):
    # Force CPU
    return xgb.XGBClassifier(
        ...
        device="cpu",  # Always use CPU
    )
```

### GPU Training (CUDA)

#### Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- XGBoost built with CUDA support

#### Verify GPU Availability

```bash
# Check CUDA
nvidia-smi

# Check XGBoost GPU support
uv run python -c "import xgboost as xgb; print(xgb.build_info())"
```

#### Enable GPU

In `src/train.py`:
```python
"gradient_boosting": {
    ...
    "device": "gpu",
}
```

The code in `src/model.py` handles this:
```python
is_gpu = config.get("device") == "gpu"
device = "cuda" if is_gpu else "cpu"
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'xxx'`

Your kernel isn't using the project environment.

```bash
# Verify packages are installed
uv run python -c "import polars, sklearn, xgboost; print('OK')"

# Re-register kernel
uv run python -m ipykernel install --user --name cybersec --display-name "cybersec (.venv)"
```

### `FileNotFoundError: data/cybersecurity_threat_detection_logs.csv`

Run from the project root:

```bash
cd /path/to/cybersec
ls data/  # Should show the CSV file
uv run python main.py
```

### `CUDA error: no CUDA-capable device is detected`

Switch to CPU mode:

```python
# In src/train.py
"device": "cpu",
```

### `MemoryError` during training

Reduce model complexity:

```python
# Random Forest
"n_estimators": 100,  # Reduce from 200

# XGBoost
"n_estimators": 200,  # Reduce from 400
```

### Lock File Conflicts

```bash
# Regenerate lock file
uv lock

# Sync with new lock
uv sync
```

### Python Version Mismatch

```bash
# Install Python 3.13
uv python install 3.13

# Set as default for project
uv python pin 3.13

# Resync
uv sync
```

---

## Inference API

The project includes a production-ready **FastAPI** inference service for real-time threat detection.

### API Architecture

```
inference/
├── __init__.py
├── api.py          # FastAPI application & endpoints
├── service.py      # Model loading & prediction logic
├── schemas.py      # Pydantic request/response models
└── logging.py      # Structured logging setup
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check - returns model status |
| `/predict` | POST | Predict threat level from log data |
| `/docs` | GET | Interactive Swagger UI documentation |
| `/redoc` | GET | ReDoc API documentation |

### Request/Response Schema

**Request (`POST /predict`):**

```json
{
    "source_ip": "192.168.1.10",
    "protocol": "HTTP",
    "log_type": "firewall",
    "bytes_transferred": 1200,
    "user_agent": "curl/7.68.0",
    "request_path": "/admin/login.php?user=admin"
}
```

**Response:**

```json
{
    "prediction": "suspicious",
    "confidence": 0.9999989271163940
}
```

### Prediction Labels

| Label | Meaning |
|-------|---------|
| `benign` | Normal, safe traffic |
| `suspicious` | Potentially malicious, requires investigation |
| `malicious` | Active threat, immediate action required |

### How the Inference Pipeline Works

1. **Model Loading** (startup):
   - Attempts to load `models/best_model.joblib` (preferred, compressed)
   - Falls back to `models/best_model.pkl` and auto-converts to joblib
   - Loads feature columns and normalization parameters from `artifacts/`

2. **Request Processing**:
   - Validates incoming JSON against Pydantic schema
   - Casts columns to correct types (Categorical, Int64, Utf8)
   - Applies same feature engineering as training pipeline

3. **Prediction**:
   - Runs inference using loaded model
   - Returns predicted class and confidence score

---

## Deployment

### Option 1: Local Development Server

Run the API directly without Docker:

```bash
# Ensure dependencies are installed
uv sync

# Start the development server
uv run uvicorn inference.api:app --host 0.0.0.0 --port 8000 --reload
```

**Flags explained:**
- `--host 0.0.0.0`: Accept connections from any IP
- `--port 8000`: Listen on port 8000
- `--reload`: Auto-reload on code changes (development only)

**Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "source_ip": "192.168.1.10",
    "protocol": "HTTP",
    "log_type": "firewall",
    "bytes_transferred": 1200,
    "user_agent": "curl/7.68.0",
    "request_path": "/admin/login.php?user=admin"
  }'
```

**Production server (without reload):**

```bash
uv run uvicorn inference.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Option 2: Docker Deployment (Recommended for Production)

#### Build the Image

```bash
docker build -t cybersec-api:cpu .
```

#### Run the Container

```bash
# Basic run
docker run -p 8000:8000 cybersec-api:cpu

# Run in background (detached)
docker run -d -p 8000:8000 --name cybersec-api cybersec-api:cpu

# With restart policy
docker run -d -p 8000:8000 --restart unless-stopped --name cybersec-api cybersec-api:cpu
```

#### Docker Commands Reference

```bash
# View running containers
docker ps

# View logs
docker logs cybersec-api
docker logs -f cybersec-api  # Follow logs

# Stop container
docker stop cybersec-api

# Remove container
docker rm cybersec-api

# Remove image
docker rmi cybersec-api:cpu
```

#### Test the Containerized API

```bash
# Health check
curl http://localhost:8000/health
# Response: {"status":"ok","model_loaded":true}

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "source_ip": "10.0.0.50",
    "protocol": "SSH",
    "log_type": "ids",
    "bytes_transferred": 5000,
    "user_agent": "Nmap/7.92",
    "request_path": "/../../etc/passwd"
  }'
# Response: {"prediction":"malicious","confidence":0.998...}
```

### Dockerfile Explained

```dockerfile
# Base image - Python 3.13 slim for minimal size
FROM python:3.13-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy dependency file and install (layer caching optimization)
COPY pyproject.toml ./
RUN uv sync --no-dev --extra cpu  # Production deps + CPU-only XGBoost

# Copy application code and artifacts
COPY src/ src/
COPY inference/ inference/
COPY artifacts/ artifacts/
COPY models/ models/

EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uv", "run", "uvicorn", "inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Dependency Groups

The `pyproject.toml` uses optional dependency groups:

| Group | Purpose | Install Command |
|-------|---------|-----------------|
| (base) | Production runtime | `uv sync --no-dev` |
| `dev` | Notebooks, plotting, testing | `uv sync --extra dev` |
| `cpu` | CPU-only XGBoost | `uv sync --extra cpu` |

**Docker uses:** `uv sync --no-dev --extra cpu` for minimal image without CUDA libraries.

### Model Format Handling

The inference service automatically handles model format conversion:

```python
def load_model():
    # 1. Try loading joblib (preferred - smaller, faster)
    if "best_model.joblib" exists:
        return joblib.load(...)
    
    # 2. Fall back to pickle and convert
    if "best_model.pkl" exists:
        model = pickle.load(...)
        joblib.dump(model, "best_model.joblib", compress=3)  # Cache for next load
        return model
```

**Benefits of joblib over pickle:**
- 3x compression with `compress=3`
- Faster loading for numpy arrays
- Memory-mapped loading option for large models

### Production Considerations

#### Running Behind a Reverse Proxy (nginx)

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

#### Docker Compose (with health checks)

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### Environment Variables

```bash
# Production settings
docker run -d \
  -p 8000:8000 \
  -e LOG_LEVEL=warning \
  -e WORKERS=4 \
  cybersec-api:cpu
```

---

## Quick Reference

```bash
# Setup
uv sync                          # Install dependencies
uv run python main.py            # Run full pipeline

# Package management
uv add <package>                 # Add dependency
uv remove <package>              # Remove dependency
uv sync --upgrade                # Update all packages

# Running scripts
uv run python main.py            # Run main pipeline
uv run python -m pytest          # Run tests
uv run jupyter lab               # Start Jupyter

# Environment info
uv run python --version          # Check Python version
uv run pip list                  # List installed packages
```

---

## License

MIT License - See LICENSE file for details.

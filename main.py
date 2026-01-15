"""
main.py

Master orchestration entry point for the cybersecurity threat detection system.

Responsibilities
----------------
- Execute end-to-end pipeline
- Coordinate preparation, training, and evaluation
- Persist experiment logs
"""

import os
from datetime import datetime
from pathlib import Path
import pickle

from src import preparation
from src.train import main as train_main


# -------------------------------------------------
# Logging utilities
# -------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "log.txt"


def get_next_run_id() -> int:
    if not LOG_FILE.exists():
        return 1
    with open(LOG_FILE, "r") as f:
        return sum(1 for line in f if "TRAINING RUN" in line) + 1


def log_run(run_id: int, model_name: str, metrics: dict):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(LOG_FILE, "a") as f:
        f.write(f"\n[{timestamp}] TRAINING RUN #{run_id}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Malicious Recall: {metrics['malicious_recall']:.4f}\n")
        f.write(f"Suspicious Recall: {metrics['suspicious_recall']:.4f}\n")
        f.write("-" * 50 + "\n")


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def main():
    print("🚀 Starting Cybersecurity Threat Detection Pipeline")

    artifacts_dir = Path("artifacts")

    # Step 1: Data preparation (run once)
    if not artifacts_dir.exists() or not list(artifacts_dir.glob("*.pkl")):
        print("🔧 Running data preparation...")
        preparation.main()
    else:
        print("✅ Artifacts found — skipping preparation")

    # Step 2: Training & evaluation
    run_id = get_next_run_id()
    print(f"🧪 Training Run #{run_id}")

    # Run training pipeline
    train_main()

    # Step 3: Load final metrics
    with open("models/final_metrics.pkl", "rb") as f:
        final_metrics = pickle.load(f)

    best_model_name = final_metrics["model_name"].replace("_test", "")
    log_run(run_id, best_model_name, final_metrics)

    print(f"✅ Training Run #{run_id} completed")
    print("📊 Results logged to logs/log.txt")


if __name__ == "__main__":
    main()

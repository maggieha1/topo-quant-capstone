# Topological Quant Capstone

Portfolio optimization using persistent homology (ripser) on market data,
graph neural networks (PyTorch Geometric), and convex optimization (cvxpy).

## Structure
- `data/` – raw & processed market data (not tracked; regenerate via scripts)
- `models/` – model definitions and saved weights
- `notebooks/` – exploratory analysis
- `scripts/` – production pipeline scripts

## Setup
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
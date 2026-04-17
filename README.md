# Shift Robustness Audit

This project audits how a text classifier trained on one Amazon review domain behaves after a domain shift. It trains on `Digital Music` reviews, evaluates on held-out `Digital Music`, and then measures the same models on `Luxury Beauty` reviews to show what changes once the data distribution moves.

## Problem

Source-domain validation can look strong while out-of-domain performance falls apart. The goal here is to make that failure mode visible with a small, reproducible experiment that still uses real review text and realistic modeling choices.

## Dataset And Domain Split

The pipeline downloads two public Amazon review subsets from the UCSD Amazon review collection:

- `Digital_Music_5.json.gz`
- `Luxury_Beauty_5.json.gz`

Ratings are converted into a binary label:

- `1` for ratings of 4 or 5
- `0` for ratings of 1 or 2

Three-star reviews are dropped to keep the decision boundary clean. The script builds a balanced subsample per domain so the full run stays practical on a local machine.

## Models

The audit compares two feature pipelines:

- TF-IDF + Linear SVM
- Frozen pooled GloVe embeddings from `glove-wiki-gigaword-50` + logistic regression

Each pipeline is trained twice:

- standard empirical risk minimization
- importance-weighted training, where source examples are reweighted using a domain classifier trained to separate source from target text

## Shift-Aware Evaluation

The code keeps the setup simple:

- source train, source validation, and source test splits from `Digital Music`
- target validation and target test splits from `Luxury Beauty`
- source-only model selection
- mixed validation analysis that includes target-domain labels

The output artifacts show whether strong source-domain validation matches shifted test performance, and whether simple weighting recovers any of the target-domain loss.

## Results

Running the pipeline writes the full audit into `outputs/`:

- `performance_summary.csv` with source and target metrics for every final model
- `validation_selection_summary.csv` with candidate hyperparameters scored under source-only and mixed validation
- `metrics_report.txt` with the main takeaways
- `example_predictions.csv` with a few target-domain predictions for inspection
- `performance_drop.png`, `weighted_comparison.png`, and `validation_strategy.png`

The main comparison to look for is the gap between held-out source performance and target-domain performance. The weighting and mixed-validation artifacts make it easy to see whether a shift-aware setup changes which model looks best.

## Run

```bash
cd /Users/gokulnambiar/Codex/shift_robustness_audit
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/download_amazon_reviews.py
python main.py
```

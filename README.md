# Sarcasm Detection with Word Embeddings

Comparing Word2Vec, GloVe, and BERT embeddings for sarcasm detection using BiLSTM classifiers across news headlines and tweets.

## Team
Namel, Breeha, Ashbah — Habib University

## Overview

This project evaluates three embedding approaches for binary sarcasm classification:
- **Word2Vec**: Skip-gram with negative sampling (trained from scratch)
- **GloVe**: Pre-trained 100d embeddings
- **BERT**: Hierarchical BERT + BiLSTM architecture

All models use a BiLSTM classifier with mean/max pooling.

## Datasets

| Dataset | Samples | Sarcastic % | Format |
|---------|---------|-------------|--------|
| Headlines | 26,709 | 43.9% | JSONL |
| Tweets | 20,000 | 25.1% | JSON |

The tweets dataset was converted from CSV (80k+ samples reduced to 20k, 4 classes merged to binary).

## Results

| Model | Headlines F1 | Tweets F1 |
|-------|-------------|-----------|
| Word2Vec + BiLSTM | 0.854 | 0.776 |
| GloVe + BiLSTM | 0.864 | 0.760 |
| BERT + BiLSTM | 0.900 | 0.772 |

BERT performs best on formal headlines; Word2Vec leads on informal tweets.

## Project Structure

```
├── word2vec/
│   └── sarcasm_detector.py
├── glove/
│   └── sarcasm_detector.py
├── bert/
│   └── sarcasm_detector.py
├── visualizations/
│   └── generate_visualizations.py
├── sarcasm_dataset.json
├── tweet_dataset.json
├── sarcasm_detection_report.tex
└── custom.bib
```

## Setup

```bash
pip install torch numpy scikit-learn transformers matplotlib seaborn
```

Requires Python 3.8+. GPU recommended for BERT.

## Usage

Run any model:
```bash
cd word2vec && python sarcasm_detector.py
cd glove && python sarcasm_detector.py
cd bert && python sarcasm_detector.py
```

Generate visualizations:
```bash
cd visualizations && python generate_visualizations.py
```

## Model Architecture

```
Embeddings → BiLSTM (2 layers, bidirectional) → Mean+Max Pooling → Dense → Output
```

Key hyperparameters:
- Word2Vec/GloVe: 40 epochs, batch size 100, hidden dim 64
- BERT: 5 epochs, batch size 32, learning rate 2e-5

## Outputs

Each model generates:
- `metrics_*.json` — accuracy, precision, recall, F1, confusion matrix
- `predictions_*.csv` — per-sample predictions
- `*.pth` — saved model weights

Visualizations include metric comparisons, confusion matrices, and training curves.

# Sarcasm Detection Visualizations

This directory contains comprehensive visualizations comparing the performance of three embedding models (Word2Vec, GloVe, and BERT) on sarcasm detection across two datasets (Headlines and Tweets).

## Generated Visualizations

### 1. Model Comparison: F1-Score
**File:** `1_model_comparison_f1_score.png`

Bar chart comparing F1-scores across all three models (Word2Vec, GloVe, BERT) for both Headlines and Tweets datasets. This provides a quick overview of which model performs best on each dataset.

### 2. Metric Breakdown
**File:** `2_metric_breakdown.png`

Four-panel visualization showing detailed performance metrics:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Quality of positive predictions
- **Recall**: Coverage of actual positive cases  
- **F1-Score**: Harmonic mean of precision and recall

Each metric is shown for all three models on both datasets.

### 3. Confusion Matrices
**File:** `3_confusion_matrices.png`

Heatmap grid (2 rows × 3 columns) showing confusion matrices for:
- **Row 1**: Headlines dataset (Word2Vec, GloVe, BERT)
- **Row 2**: Tweets dataset (Word2Vec, GloVe, BERT)

Each matrix shows True Negatives (TN), False Positives (FP), False Negatives (FN), and True Positives (TP).

### 4. Training Curves - Headlines
**File:** `4_training_curves_headlines.png`

Four-panel visualization showing training dynamics for the Headlines dataset:
- **Top Left**: Training vs Validation Accuracy for all models
- **Top Right**: Training vs Validation Loss for all models
- **Bottom Left**: Detailed Word2Vec training curves (accuracy + loss)
- **Bottom Right**: Detailed GloVe training curves (accuracy + loss)

### 5. Training Curves - Tweets
**File:** `5_training_curves_tweets.png`

Four-panel visualization showing training dynamics for the Tweets dataset:
- **Top Left**: Training vs Validation Accuracy for all models
- **Top Right**: Training vs Validation Loss for all models
- **Bottom Left**: Detailed Word2Vec training curves (accuracy + loss)
- **Bottom Right**: Detailed GloVe training curves (accuracy + loss)

### 6. Domain Comparison
**File:** `6_domain_comparison.png`

Cross-domain performance comparison showing how each model generalizes across different text domains (Headlines vs Tweets). Helps identify which models are more robust to domain shift.

## Key Findings

### Best Performing Models
- **Headlines Dataset**: 
  - GloVe: F1 = 0.864
  - BERT: F1 = 0.900
  - Word2Vec: F1 = 0.854

- **Tweets Dataset**:
  - GloVe: F1 = 0.760
  - BERT: F1 = 0.772
  - Word2Vec: F1 = 0.776

### Model Characteristics
- **BERT**: Highest performance on headlines, excellent on tweets, but computationally expensive
- **Word2Vec**: Consistent performance, best on tweets, trained from scratch
- **GloVe**: Strong performance on headlines, good on tweets, uses pre-trained embeddings

## How to Regenerate

To regenerate all visualizations:

```bash
cd visualizations
python generate_visualizations.py
```

**Requirements:**
- matplotlib
- seaborn
- numpy
- json
- pathlib

## Data Sources

The script reads metrics from:
- `../word2vec/metrics_word2vec_headlines.json`
- `../word2vec/metrics_word2vec_tweets.json`
- `../glove/metrics_gloVe_headlines1.json`
- `../glove/metrics_gloVe_tweets.json`
- `../bert/bert_output.txt` (headlines)
- `../bert/metrics_BERT_tweets.json`

## Notes

- All graphs are saved as high-resolution PNG files (300 DPI)
- Each visualization is generated as a separate file for easy inclusion in reports
- Color scheme is consistent across all visualizations for easy comparison
- Training curves show both accuracy and loss to provide complete picture of model training

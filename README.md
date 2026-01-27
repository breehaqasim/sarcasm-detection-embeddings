# Sarcasm Detection with Word Embeddings

A deep learning project that implements sarcasm detection using Word2Vec embeddings trained from scratch and a BiLSTM neural network classifier.

## Project Overview

This project performs binary classification on text headlines to detect sarcasm. It implements:
- **Word2Vec** embeddings from scratch using skip-gram with negative sampling
- **BiLSTM** (Bidirectional Long Short-Term Memory) neural network for classification
- Complete training and evaluation pipeline with detailed metrics

## Team Members
- Namel
- Breeha
- Ashbah

## Dataset

The project uses the News Headlines Dataset for Sarcasm Detection, which contains:
- Headlines from news articles
- Binary labels (sarcastic = 1, not sarcastic = 0)
- Source links for each headline

**Data Format:**
```json
{
  "article_link": "URL",
  "headline": "text of the headline",
  "is_sarcastic": 0 or 1
}
```

## Dependencies

### Required Python Packages

Install all required packages using:
```bash
pip install -r requirements.txt
```

The required packages are:
- `torch>=2.0.0` - PyTorch for neural network implementation
- `numpy>=1.24.0` - Numerical computations
- `scikit-learn>=1.3.0` - Train/test splitting and evaluation metrics

### System Requirements

- Python 3.8 or higher
- GPU (optional but recommended for faster training)
  - The code automatically detects and uses CUDA if available
  - Falls back to CPU if GPU is not available

## Project Structure

```
.
├── sarcasm_detector.py              # Main implementation file
├── sarcasm_dataset.json             # Dataset (26,710 headlines)
├── requirements.txt                 # Python dependencies
├── predictions_output.json          # Generated predictions (after running)
├── sarcasm_bilstm_model.pth        # Saved model weights (after running)
└── README.md                        # This file
```

## Architecture

### 1. Word2Vec Implementation
- **Algorithm:** Skip-gram with negative sampling
- **Training:** Custom implementation from scratch
- **Vocabulary:** Built from training set with minimum word frequency filtering
- **Embeddings:** Learned representations capturing semantic relationships

### 2. BiLSTM Classifier
```
Input (Embeddings) 
    → Dropout (0.2)
    → BiLSTM (2 layers, hidden_dim=64, bidirectional)
    → Mean & Max Pooling (concatenated)
    → Dense Layer (128 units) + BatchNorm + ReLU + Dropout
    → Dense Layer (32 units) + ReLU + Dropout
    → Output Layer (1 unit, sigmoid)
```

**Key Features:**
- Bidirectional LSTM captures context from both directions
- Combined mean and max pooling for robust feature extraction
- Batch normalization for training stability
- Dropout layers for regularization
- Gradient clipping to prevent exploding gradients

## How to Run

### Basic Usage

Simply run the main script:
```bash
python sarcasm_detector.py
```

The script will:
1. Load and preprocess the data
2. Train Word2Vec embeddings from scratch
3. Create train/test datasets (70/30 split)
4. Train the BiLSTM classifier
5. Evaluate on the test set
6. Save predictions and model weights

## Hyperparameters

The default hyperparameters are configured in the `main()` function:

### Data Parameters
```python
NUM_SAMPLES = 4000      # Number of samples to use from dataset
MAX_LENGTH = 50         # Maximum sequence length (padding/truncation)
```

### Word2Vec Parameters
```python
EMBEDDING_DIM = 128     # Dimension of word embeddings
window_size = 10        # Context window size
negative_samples = 10   # Number of negative samples
learning_rate = 0.025   # Initial learning rate (decays by 0.95 per epoch)
min_count = 2          # Minimum word frequency threshold
EPOCHS_W2V = 25        # Word2Vec training epochs
```

### BiLSTM Parameters
```python
HIDDEN_DIM = 64        # LSTM hidden dimension
num_layers = 2         # Number of LSTM layers
dropout = 0.3          # Dropout rate
BATCH_SIZE = 100       # Training batch size
EPOCHS_LSTM = 40       # BiLSTM training epochs
LEARNING_RATE = 0.001  # Initial learning rate (adaptive with scheduler)
```

### Modifying Hyperparameters

To modify hyperparameters, edit the values in the `main()` function in `sarcasm_detector.py` (lines 448-458).

## Output Files

### 1. `predictions_output.json`
Contains detailed predictions for each test sample:
```json
[
  {
    "sample_id": 1,
    "text": "preprocessed headline text",
    "true_label": 0,
    "predicted_label": 1,
    "confidence": 0.734,
    "is_sarcastic_true": false,
    "is_sarcastic_predicted": true,
    "correct": false
  },
  ...
]
```

### 2. `sarcasm_bilstm_model.pth`
Saved PyTorch model weights that can be loaded for inference:
```python
model = BiLSTMSarcasmDetector(embedding_dim=128, hidden_dim=64)
model.load_state_dict(torch.load('sarcasm_bilstm_model.pth'))
```
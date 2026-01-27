# Word2Vec Sarcasm Detection

This folder contains the Word2Vec-based sarcasm detection implementation using BiLSTM.

## Files
- `sarcasm_detector.py` - Main implementation file
- `predictions_output.json` - Model predictions output
- `sarcasm_bilstm_model.pth` - Trained model weights (generated after training)

## Usage
Run the detector from this directory:
```bash
python sarcasm_detector.py
```

## Note
- Datasets (`news_headline_dataset.json`, `tweet_dataset.json`) are located in the parent directory
- All output files are saved in this folder


import json
import numpy as np
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
import os
from datetime import datetime
import csv

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class BERT(nn.Module):
    def __init__(self, bert_model, lstm_units=128, dense_units=32):
        super(BERT, self).__init__()
        self.bert = bert_model

        # Sentence Encoding Layer
        self.dense_sentence = nn.Linear(768, 768)
        self.relu_sentence = nn.ReLU()

        # Context Summarization Layer (mean pooling)
        # We'll do this manually in forward pass

        # Context Encoder Layer
        self.bilstm_encoder = nn.LSTM(
            768, lstm_units, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=True
        )

        # Fully Connected Layer
        self.dense_fc = nn.Linear(lstm_units * 2, dense_units)
        self.relu_fc = nn.ReLU()
        self.output_layer = nn.Linear(dense_units, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        # BERT Embeddings
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = bert_output.last_hidden_state  # (batch_size, seq_len, 768)

        # Sentence Encoding Layer
        sentence_encoded = self.relu_sentence(self.dense_sentence(bert_output))  # (batch_size, seq_len, 768)

        # Context Summarization Layer (mean pooling)
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(sentence_encoded.size()).float()
            sum_embeddings = torch.sum(sentence_encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            context_summarized = sum_embeddings / sum_mask  # (batch_size, 768)
        else:
            context_summarized = torch.mean(sentence_encoded, dim=1)  # (batch_size, 768)

        # Expand dimensions to match the input shape required by LSTM
        context_summarized = context_summarized.unsqueeze(1)  # (batch_size, 1, 768)

        # Context Encoder Layer
        context_encoded, _ = self.bilstm_encoder(context_summarized)  # (batch_size, 1, 2 * lstm_units)
        
        # Squeeze to remove sequence dimension
        context_encoded = context_encoded.squeeze(1)  # (batch_size, 2 * lstm_units)

        # Fully Connected Layer
        dense_output = self.relu_fc(self.dense_fc(context_encoded))  # (batch_size, dense_units)

        # Output Layer
        final_output = self.sigmoid(self.output_layer(dense_output))  # (batch_size, 1)
        return final_output.squeeze(-1)


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=100):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = float(self.labels[idx])

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float32)
        }


def load_data(file_path, dataset_type='headlines'):
    """Load data from JSON/JSONL files"""
    print(f"Loading data from {file_path}...")
    
    texts = []
    labels = []
    
    if dataset_type == 'headlines':
        # JSONL format - one JSON object per line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line.strip())
                    texts.append(item['headline'])
                    labels.append(item['is_sarcastic'])
    else:  # tweets
        # JSON array format
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for item in items:
                texts.append(item['tweets'])
                labels.append(item['class'])
    
    print(f"Loaded {len(texts)} samples")
    print(f"Sarcastic ratio: {sum(labels)/len(labels):.2%}")
    
    return texts, labels


def preprocess_text(text):
    """Basic text preprocessing"""
    # Remove unwanted numerals and symbols, keep only letters and spaces
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text


def train_model(model, train_loader, val_loader, epochs=5, device='cpu', lr=2e-5):
    """Train the model and track metrics"""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Store epoch metrics
    epoch_metrics = []
    
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        print(f"\nEpoch {epoch+1}/{epochs} - Training...")
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % max(1, total_batches // 10) == 0:  # Print ~10 times per epoch
                print(f"  Processing batch {batch_idx+1}/{total_batches}...")
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = (outputs > 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device).float()
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc)
        })
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, epoch_metrics


def evaluate_model(model, test_loader, test_texts, device='cpu'):
    """Evaluate model and return metrics"""
    model = model.to(device)
    model.eval()
    
    print("\nEvaluating model on test set...")
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).float()
            
            outputs = model(input_ids, attention_mask)
            probs = outputs.cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback: calculate manually
        tn = sum((all_labels[i] == 0) and (all_preds[i] == 0) for i in range(len(all_labels)))
        fp = sum((all_labels[i] == 0) and (all_preds[i] == 1) for i in range(len(all_labels)))
        fn = sum((all_labels[i] == 1) and (all_preds[i] == 0) for i in range(len(all_labels)))
        tp = sum((all_labels[i] == 1) and (all_preds[i] == 1) for i in range(len(all_labels)))
    
    print("\n" + "="*50)
    print("Final Evaluation Results")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print("="*50)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp)
    }, all_preds, all_probs, all_labels


def save_predictions_csv(test_texts, test_labels, pred_labels, pred_probs, dataset_name, output_file):
    """Save per-example predictions to CSV"""
    print(f"\nSaving predictions to {output_file}...")
    
    # Convert to lists if they're numpy arrays
    test_labels = list(test_labels) if hasattr(test_labels, '__iter__') and not isinstance(test_labels, str) else test_labels
    pred_labels = list(pred_labels) if hasattr(pred_labels, '__iter__') and not isinstance(pred_labels, str) else pred_labels
    pred_probs = list(pred_probs) if hasattr(pred_probs, '__iter__') and not isinstance(pred_probs, str) else pred_probs
    
    # Check lengths match
    lengths = [len(test_texts), len(test_labels), len(pred_labels), len(pred_probs)]
    if len(set(lengths)) > 1:
        raise ValueError(f"Length mismatch: texts={lengths[0]}, labels={lengths[1]}, pred_labels={lengths[2]}, pred_probs={lengths[3]}")
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'dataset', 'true_label', 'pred_label', 'prob_sarcastic'])
            
            for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, test_labels, pred_labels, pred_probs)):
                writer.writerow([
                    i + 1,
                    dataset_name,
                    int(true_label),
                    int(pred_label),
                    float(prob)
                ])
        
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        raise


def save_metrics_json(metrics_data, output_file):
    """Save metrics to JSON file"""
    print(f"\nSaving metrics to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        print(f"Metrics saved to {output_file}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        raise


def run_experiment(dataset_path, dataset_name, dataset_type='headlines', epochs=5, batch_size=32, max_length=100):
    """Run complete experiment on a dataset"""
    print("="*70)
    print(f"SARCASM DETECTION WITH BERT - {dataset_name.upper()}")
    print("="*70)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n" + "="*50)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*50)
    texts, labels = load_data(dataset_path, dataset_type=dataset_type)
    
    # Preprocess texts
    texts = [preprocess_text(text) for text in texts]
    labels = np.array(labels, dtype=np.float32)
    
    # Split data: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 = 0.15
    )
    
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize tokenizer
    print("\n" + "="*50)
    print("STEP 2: Loading Pretrained BERT Tokenizer")
    print("="*50)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create datasets
    print("\n" + "="*50)
    print("STEP 3: Creating PyTorch Datasets")
    print("="*50)
    train_dataset = SarcasmDataset(X_train, y_train, tokenizer, max_length=max_length)
    val_dataset = SarcasmDataset(X_val, y_val, tokenizer, max_length=max_length)
    test_dataset = SarcasmDataset(X_test, y_test, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Load pretrained BERT model
    print("\n" + "="*50)
    print("STEP 4: Loading Pretrained BERT Model")
    print("="*50)
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Define the BERT model
    print("\n" + "="*50)
    print("STEP 5: Building BERT Model")
    print("="*50)
    model = BERT(bert_model, lstm_units=128, dense_units=32)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\n" + "="*50)
    print("STEP 6: Training Model")
    print("="*50)
    model, epoch_metrics = train_model(model, train_loader, val_loader, epochs=epochs, device=device, lr=2e-5)
    
    # Evaluate model
    print("\n" + "="*50)
    print("STEP 7: Evaluating Model on Test Set")
    print("="*50)
    final_metrics, pred_labels, pred_probs, test_labels = evaluate_model(model, test_loader, X_test, device=device)
    
    # Save predictions CSV (use test_labels from evaluation for consistency)
    predictions_file = f'predictions_BERT_{dataset_name}.csv'
    save_predictions_csv(X_test, test_labels, pred_labels, pred_probs, dataset_name, predictions_file)
    
    # Prepare metrics data
    metrics_data = {
        'model_name': 'BERT',
        'dataset': dataset_name,
        'accuracy': final_metrics['accuracy'],
        'precision': final_metrics['precision'],
        'recall': final_metrics['recall'],
        'f1': final_metrics['f1'],
        'tn': final_metrics['tn'],
        'fp': final_metrics['fp'],
        'fn': final_metrics['fn'],
        'tp': final_metrics['tp'],
        'epochs': epochs,
        'epoch_metrics': epoch_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metrics JSON
    metrics_file = f'metrics_BERT_{dataset_name}.json'
    save_metrics_json(metrics_data, metrics_file)
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE FOR {dataset_name.upper()}!")
    print("="*70)
    
    return metrics_data


def main():
    """Main function to run experiment"""
    # Change to BERT directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    print("\n\n" + "="*70)
    print("STARTING TWEETS DATASET EXPERIMENT")
    print("="*70 + "\n")
    
    tweets_metrics = run_experiment(
        '../tweet_dataset.json',
        'tweets',
        dataset_type='tweets',
        epochs=5,
        batch_size=32,
        max_length=100
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE!")
    print("="*70)
    print(f"\nMetrics saved to: metrics_BERT_tweets.json")
    print(f"Predictions saved to: predictions_BERT_tweets.csv")


if __name__ == "__main__":
    main()

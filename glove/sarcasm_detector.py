import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import random
import csv
import os
from datetime import datetime
import urllib.request
import zipfile
from pathlib import Path

#random seeds 
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


# GloVe Embedding Loader
class GloVeEmbeddings:

    #load and manage pretrained embeddings
    def __init__(self, embedding_dim=100, glove_file=None):
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.glove_file = glove_file
        
    def load_glove_embeddings(self, glove_file_path=None):
        """Load GloVe embeddings from file"""
        if glove_file_path is None:
            #find existing GloVe file or download
            glove_dir = Path(__file__).parent / "glove_embeddings"
            glove_file_path = glove_dir / f"glove.6B.{self.embedding_dim}d.txt"
            
            if not glove_file_path.exists():
                print("GloVe embeddings not found. Downloading...")
                self._download_glove_embeddings()
                glove_file_path = Path(self.glove_file)
            
        print(f"Loading GloVe embeddings from {glove_file_path}...")
        
        embeddings_dict = {}
        vocab_size = 0
        line_count = 0
        
        try:
            with open(glove_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line_count += 1
                    if line_count % 10000 == 0:
                        print(f"  Processed {line_count} lines...", end='\r')
                    values = line.split()
                    if len(values) < self.embedding_dim + 1:
                        continue
                    word = values[0]
                    vector = np.asarray(values[1:self.embedding_dim+1], dtype='float32')
                    
                    #check if embedding dimension matches
                    if len(vector) == self.embedding_dim:
                        embeddings_dict[word] = vector
                        vocab_size += 1
                        
            print(f"\nLoaded {vocab_size} word vectors")
            
        except FileNotFoundError as e:
            print(f"Error: GloVe file not found: {glove_file_path}")
            print(f"Error details: {e}")
            raise
        except Exception as e:
            print(f"Error loading GloVe embeddings: {e}")
            raise
        
        #build vocabulary mappings
        sorted_words = sorted(embeddings_dict.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted_words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        #create embedding matrix
        vocab_size = len(self.word_to_idx)
        self.embeddings = np.zeros((vocab_size, self.embedding_dim), dtype=np.float32)
        
        for word, idx in self.word_to_idx.items():
            self.embeddings[idx] = embeddings_dict[word]
        
        print(f"Vocabulary size: {vocab_size}")
        return self.embeddings
    #download glove embeddings
    def _download_glove_embeddings(self):
        glove_dir = Path(__file__).parent / "glove_embeddings"
        glove_dir.mkdir(exist_ok=True)
        
        url = "http://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = glove_dir / "glove.6B.zip"
        target_file = f"glove.6B.{self.embedding_dim}d.txt"
        extracted_path = glove_dir / target_file
        
        if extracted_path.exists():
            print(f"GloVe embeddings already exist at {extracted_path}")
            self.glove_file = str(extracted_path)
            return
        
        if not zip_path.exists():
            print(f"Downloading GloVe embeddings from {url}...")
            print("This may take several minutes (file is ~822MB)...")
            try:
                def show_progress(block_num, block_size, total_size):
                    downloaded = block_num * block_size
                    percent = min(downloaded * 100 / total_size, 100)
                    print(f"\r  Progress: {percent:.1f}%", end='', flush=True)
                
                urllib.request.urlretrieve(url, zip_path, show_progress)
                print("\nDownload complete!")
            except Exception as e:
                print(f"\nError downloading GloVe embeddings: {e}")
                print("Please download manually from: http://nlp.stanford.edu/data/glove.6B.zip")
                raise
        
        #extract the specific dimension file
        print(f"Extracting {target_file} from zip...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                if target_file in zip_ref.namelist():
                    zip_ref.extract(target_file, glove_dir)
                    self.glove_file = str(extracted_path)
                    print(f"Extraction complete! File saved to {self.glove_file}")
                else:
                    raise FileNotFoundError(f"{target_file} not found in zip")
        except Exception as e:
            print(f"Error extracting GloVe embeddings: {e}")
            raise
    
    #get embedding for a word, return zero vector if not found
    def get_embedding(self, word):
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.embeddings[idx]
        else:
            #return zero vector for unknown words
            return np.zeros(self.embedding_dim)
        
    #embedding matrix
    def get_embedding_matrix(self):
        return self.embeddings


#data Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)  #remove chars but keep letters, no's and spaces
    tokens = text.split()
    return tokens

#load data 
def load_data(file_path, dataset_type='headlines', num_samples=None):
    print(f"Loading data from {file_path}...")
    
    data = []
    
    if dataset_type == 'headlines':
        # JSONL format 
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                item = json.loads(line.strip())
                data.append({
                    'text': item['headline'],
                    'label': item['is_sarcastic']
                })
    else:  # tweets
        # JSON array format
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for i, item in enumerate(items):
                if num_samples and i >= num_samples:
                    break
                data.append({
                    'text': item['tweets'],
                    'label': item['class']
                })
    
    #Preprocessing 
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['label'] for item in data]
    
    print(f"Loaded {len(data)} samples")
    return texts, labels


#pyTorch Dataset class
class SarcasmDataset(Dataset):
    
    def __init__(self, texts, labels, glove_embeddings, max_length=50):
        self.texts = texts
        self.labels = labels
        self.glove_embeddings = glove_embeddings
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        label = self.labels[idx]
        
        #convert tokens to embeddings
        embeddings = []
        for token in tokens[:self.max_length]:
            embeddings.append(self.glove_embeddings.get_embedding(token))
        
        #pad or truncate
        if len(embeddings) < self.max_length:
            #pad with zeros
            padding = [np.zeros(self.glove_embeddings.embedding_dim) for _ in range(self.max_length - len(embeddings))]
            embeddings.extend(padding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        return torch.tensor(embeddings), torch.tensor(label, dtype=torch.float32)


#BiLSTM Model
class BiLSTMSarcasmDetector(nn.Module):
    """Embedding -> BiLSTM -> Dense -> Sigmoid"""

    def __init__(self, embedding_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BiLSTMSarcasmDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        #Dropout for embeddings
        self.embedding_dropout = nn.Dropout(0.2)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Pooling we'll use both mean and max
        # Dense layers - concatenate mean and max pooling
        self.fc1 = nn.Linear(hidden_dim * 4, 128)  # *4 because bidirectional + mean&max pooling
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 1)
        
        self._init_weights()

    # Initialize weights for better training     
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:  # only apply xavier to 2D+ tensors
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, x):  # x shape: (batch_size, seq_len, embedding_dim)
        
        # Apply dropout to embeddings
        x = self.embedding_dropout(x)
        
        # BiLSTM
        lstm_out, (hidden, cell) = self.bilstm(x)
        # lstm_out shape (batch_size, seq_len, hidden_dim * 2)
        
        # Use both mean and max pooling for richer representation
        mean_pool = torch.mean(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        max_pool, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_dim * 2)
        pooled_output = torch.cat([mean_pool, max_pool], dim=1)  # (batch_size, hidden_dim * 4)
        
        #dense layers
        out = self.fc1(pooled_output)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out.squeeze()  #return logits (no sigmoid, handled by BCEWithLogitsLoss)


#training and eval with metrics tracking
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    #store epoch metrics
    epoch_metrics = []
    
    print("\n" + "="*50)
    print("Training BiLSTM Sarcasm Detector")
    print(f"Initial Learning Rate: {lr}")
    print("="*50)
    
    for epoch in range(epochs):
        #training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            #forward pass
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            #backward pass
            loss.backward()
            #gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            #store predictions - apply sigmoid 
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        #training metrics
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        #validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                #apply sigmoid for predictions
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        #validation metrics
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        #store epoch metrics
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


#evaluate the model 
def evaluate_model(model, test_loader, test_texts, device='cpu'):
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for embeddings, labels in test_loader:
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            # Apply sigmoid for predictions
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    #metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    #Confusion matrix 
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        #fallback
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


#save predictions to CSV
def save_predictions_csv(test_texts, all_labels, all_preds, all_probs, dataset_name, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'dataset', 'true_label', 'pred_label', 'prob_sarcastic'])
        
        for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, all_labels, all_preds, all_probs)):
            text_str = " ".join(text) if isinstance(text, list) else str(text)
            writer.writerow([
                i + 1,
                dataset_name,
                int(true_label),
                int(pred_label),
                float(prob)
            ])
    print(f"Predictions saved to {output_file}")


#save metrics to JSON
def save_metrics_json(metrics_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {output_file}")


def run_experiment(dataset_path, dataset_name, dataset_type='headlines'):
    print("="*70)
    print(f"SARCASM DETECTION WITH GLOVE - {dataset_name.upper()}")
    print("="*70)
    
    #hyperparameters
    EMBEDDING_DIM = 100  #GloVe 6B.100d
    MAX_LENGTH = 50
    BATCH_SIZE = 100
    EPOCHS_LSTM = 40
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.001 
    
    #device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data - use complete dataset
    print("\n" + "="*50)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*50)
    texts, labels = load_data(dataset_path, dataset_type=dataset_type, num_samples=None)
    
    # Split 70:30
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Sarcastic ratio in train: {sum(train_labels)/len(train_labels):.2%}")
    
    # Load GloVe embeddings
    print("\n" + "="*50)
    print("STEP 2: Loading Pretrained GloVe Embeddings")
    print("="*50)
    glove = GloVeEmbeddings(embedding_dim=EMBEDDING_DIM)
    glove.load_glove_embeddings()
    
    # Datasets and dataloaders
    print("\n" + "="*50)
    print("STEP 3: Creating PyTorch Datasets")
    print("="*50)
    train_dataset = SarcasmDataset(train_texts, train_labels, glove, max_length=MAX_LENGTH)
    test_dataset = SarcasmDataset(test_texts, test_labels, glove, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Build and train BiLSTM model
    print("\n" + "="*50)
    print("STEP 4: Building BiLSTM Model")
    print("="*50)
    model = BiLSTMSarcasmDetector(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3
    )
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    model, epoch_metrics = train_model(model, train_loader, test_loader, epochs=EPOCHS_LSTM, lr=LEARNING_RATE, device=device)
    
    # Final evaluation
    print("\n" + "="*50)
    print("STEP 5: Final Evaluation on Test Set")
    print("="*50)
    final_metrics, all_preds, all_probs, all_labels = evaluate_model(model, test_loader, test_texts, device=device)
    
    # Save model
    model_file = f'sarcasm_bilstm_model_{dataset_name}.pth'
    torch.save(model.state_dict(), model_file)
    print(f"\nModel saved as '{model_file}'")
    
    # Save predictions CSV
    predictions_file = f'predictions_gloVe_{dataset_name}.csv'
    save_predictions_csv(test_texts, all_labels, all_preds, all_probs, dataset_name, predictions_file)
    
    # Prepare metrics data
    metrics_data = {
        'model_name': 'gloVe_bilstm',
        'dataset': dataset_name,
        'accuracy': final_metrics['accuracy'],
        'precision': final_metrics['precision'],
        'recall': final_metrics['recall'],
        'f1': final_metrics['f1'],
        'tn': final_metrics['tn'],
        'fp': final_metrics['fp'],
        'fn': final_metrics['fn'],
        'tp': final_metrics['tp'],
        'epochs': EPOCHS_LSTM,
        'epoch_metrics': epoch_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save metrics JSON
    metrics_file = f'metrics_gloVe_{dataset_name}.json'
    save_metrics_json(metrics_data, metrics_file)
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETE FOR {dataset_name.upper()}!")
    print("="*70)
    
    return metrics_data


def main():
    # Change to gloVe directory to save outputs there
    os.chdir(Path(__file__).parent)
    
    # Run experiments on both datasets
    all_metrics = []
    
    # Run on headlines dataset
    headlines_metrics = run_experiment(
        '../news_headline_dataset.json',
        'headlines',
        dataset_type='headlines'
    )
    all_metrics.append(headlines_metrics)
    
    print("\n\n" + "="*70)
    print("STARTING TWEETS DATASET EXPERIMENT")
    print("="*70 + "\n")
    
    # Run on tweets dataset
    tweets_metrics = run_experiment(
        '../tweet_dataset.json',
        'tweets',
        dataset_type='tweets'
    )
    all_metrics.append(tweets_metrics)
    
    # Save combined metrics
    combined_metrics_file = 'metrics_gloVe_combined.json'
    save_metrics_json(all_metrics, combined_metrics_file)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    print("Starting GloVe Sarcasm Detector...")
    print("="*70)
    try:
        main()
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        raise

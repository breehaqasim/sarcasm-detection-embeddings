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

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


class Word2Vec:
    #skip-gram with negative sampling
    
    def __init__(self, embedding_dim=100, window_size=5, negative_samples=5, learning_rate=0.025):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples
        self.learning_rate = learning_rate
        self.vocab = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.word_freq = Counter()
        self.W_embed = None
        self.W_context = None

    def build_vocab(self, sentences, min_count=2):
        for sentence in sentences:
            self.word_freq.update(sentence)
        
        self.vocab = {word: count for word, count in self.word_freq.items() if count >= min_count}
        self.word_to_idx = {word: idx for idx, word in enumerate(sorted(self.vocab.keys()))}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        vocab_size = len(self.word_to_idx)
        
        self.W_embed = np.random.uniform(-0.5/self.embedding_dim, 0.5/self.embedding_dim, 
                                         (vocab_size, self.embedding_dim))
        self.W_context = np.random.uniform(-0.5/self.embedding_dim, 0.5/self.embedding_dim, 
                                          (vocab_size, self.embedding_dim))
        
        self._build_negative_sampling_table()
        print(f"Vocabulary size: {vocab_size}")

    def _build_negative_sampling_table(self):
        freq_array = np.array([self.vocab.get(word, 0) for word in sorted(self.vocab.keys())])
        powered_freq = np.power(freq_array, 0.75)
        self.sampling_probs = powered_freq / np.sum(powered_freq)
        
    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def train(self, sentences, epochs=5):
        print(f"Training Word2Vec for {epochs} epochs...")
        
        training_pairs = []
        for sentence in sentences:
            indices = [self.word_to_idx[word] for word in sentence if word in self.word_to_idx]
            for i, center_idx in enumerate(indices):
                start = max(0, i - self.window_size)
                end = min(len(indices), i + self.window_size + 1)
                for j in range(start, end):
                    if i != j:
                        training_pairs.append((center_idx, indices[j]))
        
        print(f"Generated {len(training_pairs)} training pairs")
        
        for epoch in range(epochs):
            random.shuffle(training_pairs)
            total_loss = 0
            
            for center_idx, context_idx in training_pairs:
                center_embed = self.W_embed[center_idx]
                context_embed = self.W_context[context_idx]
                
                pos_score = np.dot(center_embed, context_embed)
                pos_loss = -np.log(self._sigmoid(pos_score) + 1e-10)
                
                pos_grad = (self._sigmoid(pos_score) - 1)
                center_grad = pos_grad * context_embed
                context_grad = pos_grad * center_embed
                
                neg_loss = 0
                neg_indices = np.random.choice(len(self.word_to_idx), 
                                              size=self.negative_samples, 
                                              p=self.sampling_probs)
                
                for neg_idx in neg_indices:
                    if neg_idx == context_idx:
                        continue
                    
                    neg_embed = self.W_context[neg_idx]
                    neg_score = np.dot(center_embed, neg_embed)
                    neg_loss += -np.log(self._sigmoid(-neg_score) + 1e-10)
                    
                    neg_grad = self._sigmoid(neg_score)
                    center_grad += neg_grad * neg_embed
                    self.W_context[neg_idx] -= self.learning_rate * neg_grad * center_embed
                
                self.W_embed[center_idx] -= self.learning_rate * center_grad
                self.W_context[context_idx] -= self.learning_rate * context_grad
                
                total_loss += pos_loss + neg_loss
            
            avg_loss = total_loss / len(training_pairs)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.learning_rate *= 0.95
    
    def get_embedding(self, word):
        if word in self.word_to_idx:
            idx = self.word_to_idx[word]
            return self.W_embed[idx]
        else:
            return np.zeros(self.embedding_dim)
    
    def get_embedding_matrix(self):
        return self.W_embed


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return tokens

def load_data(file_path, dataset_type='headlines', num_samples=None):
    data = []
    
    if dataset_type == 'headlines':
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if num_samples and i >= num_samples:
                    break
                item = json.loads(line.strip())
                data.append({
                    'text': item['headline'],
                    'label': item['is_sarcastic']
                })
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            items = json.load(f)
            for i, item in enumerate(items):
                if num_samples and i >= num_samples:
                    break
                data.append({
                    'text': item['tweets'],
                    'label': item['class']
                })
    
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['label'] for item in data]
    
    print(f"Loaded {len(data)} samples")
    return texts, labels


class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, word2vec, max_length=50):
        self.texts = texts
        self.labels = labels
        self.word2vec = word2vec
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        label = self.labels[idx]
        
        embeddings = []
        for token in tokens[:self.max_length]:
            embeddings.append(self.word2vec.get_embedding(token))
        
        if len(embeddings) < self.max_length:
            padding = [np.zeros(self.word2vec.embedding_dim) for _ in range(self.max_length - len(embeddings))]
            embeddings.extend(padding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        return torch.tensor(embeddings), torch.tensor(label, dtype=torch.float32)


class BiLSTMSarcasmDetector(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=64, num_layers=2, dropout=0.3):
        super(BiLSTMSarcasmDetector, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding_dropout = nn.Dropout(0.2)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 4, 128)
        self.batch_norm = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 1)
        
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.orthogonal_(param)
                elif param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
    def forward(self, x):
        x = self.embedding_dropout(x)
        lstm_out, (hidden, cell) = self.bilstm(x)
        
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool, _ = torch.max(lstm_out, dim=1)
        pooled_output = torch.cat([mean_pool, max_pool], dim=1)
        
        out = self.fc1(pooled_output)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        
        return out.squeeze()

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epoch_metrics = []
    
    print(f"\nTraining BiLSTM (lr={lr})")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
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
                
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        epoch_metrics.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'train_accuracy': float(train_acc),
            'val_accuracy': float(val_acc)
        })
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return model, epoch_metrics

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
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float().cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = sum((all_labels[i] == 0) and (all_preds[i] == 0) for i in range(len(all_labels)))
        fp = sum((all_labels[i] == 0) and (all_preds[i] == 1) for i in range(len(all_labels)))
        fn = sum((all_labels[i] == 1) and (all_preds[i] == 0) for i in range(len(all_labels)))
        tp = sum((all_labels[i] == 1) and (all_preds[i] == 1) for i in range(len(all_labels)))
    
    print(f"\nResults - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
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

def save_metrics_json(metrics_data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to {output_file}")


def run_experiment(dataset_path, dataset_name, dataset_type='headlines'):
    print(f"\n{'='*60}")
    print(f"Word2Vec + BiLSTM - {dataset_name}")
    print(f"{'='*60}")
    
    EMBEDDING_DIM = 128
    MAX_LENGTH = 50
    BATCH_SIZE = 100
    EPOCHS_W2V = 25 
    EPOCHS_LSTM = 40
    HIDDEN_DIM = 64
    LEARNING_RATE = 0.001 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    texts, labels = load_data(dataset_path, dataset_type=dataset_type, num_samples=None)
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Train: {len(train_texts)}, Test: {len(test_texts)}, Sarcastic ratio: {sum(train_labels)/len(train_labels):.2%}")
    
    w2v = Word2Vec(embedding_dim=EMBEDDING_DIM, window_size=10, negative_samples=10, learning_rate=0.025)
    w2v.build_vocab(train_texts, min_count=2)
    w2v.train(train_texts, epochs=EPOCHS_W2V)
    
    train_dataset = SarcasmDataset(train_texts, train_labels, w2v, max_length=MAX_LENGTH)
    test_dataset = SarcasmDataset(test_texts, test_labels, w2v, max_length=MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = BiLSTMSarcasmDetector(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        dropout=0.3
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    model, epoch_metrics = train_model(model, train_loader, test_loader, epochs=EPOCHS_LSTM, lr=LEARNING_RATE, device=device)
    
    final_metrics, all_preds, all_probs, all_labels = evaluate_model(model, test_loader, test_texts, device=device)
    
    model_file = f'sarcasm_bilstm_model_{dataset_name}.pth'
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")
    
    predictions_file = f'predictions_word2vec_{dataset_name}.csv'
    save_predictions_csv(test_texts, all_labels, all_preds, all_probs, dataset_name, predictions_file)
    
    metrics_data = {
        'model_name': 'word2vec_bilstm',
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
    
    metrics_file = f'metrics_word2vec_{dataset_name}.json'
    save_metrics_json(metrics_data, metrics_file)
    
    return metrics_data


def main():
    all_metrics = []
    
    try:
        with open('metrics_word2vec_headlines.json', 'r', encoding='utf-8') as f:
            headlines_metrics = json.load(f)
            all_metrics.append(headlines_metrics)
            print("Loaded existing headlines metrics")
    except FileNotFoundError:
        pass
    
    tweets_metrics = run_experiment(
        '../tweet_dataset.json',
        'tweets',
        dataset_type='tweets'
    )
    all_metrics.append(tweets_metrics)
    
    combined_metrics_file = 'metrics_word2vec_combined.json'
    save_metrics_json(all_metrics, combined_metrics_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

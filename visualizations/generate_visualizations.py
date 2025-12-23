import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# model colors
COLORS = {
    'Word2Vec': '#2E86AB',
    'GloVe': '#A23B72',
    'BERT': '#F18F01'
}

DATASET_COLORS = {
    'headlines': '#6A994E',
    'tweets': '#BC4749'
}

def load_metrics():
    base_path = Path('..')
    
    with open(base_path / 'word2vec' / 'metrics_word2vec_headlines.json', 'r') as f:
        w2v_headlines = json.load(f)
    
    with open(base_path / 'word2vec' / 'metrics_word2vec_tweets.json', 'r') as f:
        w2v_tweets = json.load(f)
    
    with open(base_path / 'glove' / 'metrics_gloVe_headlines1.json', 'r') as f:
        glove_headlines = json.load(f)
    
    with open(base_path / 'glove' / 'metrics_gloVe_tweets.json', 'r') as f:
        glove_tweets = json.load(f)
    
    bert_headlines = parse_bert_output(base_path / 'bert' / 'bert_output.txt')
    
    with open(base_path / 'bert' / 'metrics_BERT_tweets.json', 'r') as f:
        bert_tweets = json.load(f)
    
    return {
        'Word2Vec': {'headlines': w2v_headlines, 'tweets': w2v_tweets},
        'GloVe': {'headlines': glove_headlines, 'tweets': glove_tweets},
        'BERT': {'headlines': bert_headlines, 'tweets': bert_tweets}
    }

def parse_bert_output(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    accuracy_match = re.search(r'Accuracy:\s*([\d.]+)', content)
    precision_match = re.search(r'Precision:\s*([\d.]+)', content)
    recall_match = re.search(r'Recall:\s*([\d.]+)', content)
    f1_match = re.search(r'F1-Score:\s*([\d.]+)', content)
    cm_match = re.search(r'TN=(\d+),\s*FP=(\d+),\s*FN=(\d+),\s*TP=(\d+)', content)
    
    epoch_pattern = r'Epoch (\d+)/\d+.*?Train Loss: ([\d.]+), Train Acc: ([\d.]+).*?Val Loss: ([\d.]+), Val Acc: ([\d.]+)'
    epoch_matches = re.findall(epoch_pattern, content, re.DOTALL)
    
    epoch_metrics = []
    for match in epoch_matches:
        epoch_metrics.append({
            'epoch': int(match[0]),
            'train_loss': float(match[1]),
            'train_accuracy': float(match[2]),
            'val_loss': float(match[3]),
            'val_accuracy': float(match[4])
        })
    
    return {
        'model_name': 'bert_bilstm',
        'dataset': 'headlines',
        'accuracy': float(accuracy_match.group(1)) if accuracy_match else 0,
        'precision': float(precision_match.group(1)) if precision_match else 0,
        'recall': float(recall_match.group(1)) if recall_match else 0,
        'f1': float(f1_match.group(1)) if f1_match else 0,
        'tn': int(cm_match.group(1)) if cm_match else 0,
        'fp': int(cm_match.group(2)) if cm_match else 0,
        'fn': int(cm_match.group(3)) if cm_match else 0,
        'tp': int(cm_match.group(4)) if cm_match else 0,
        'epochs': len(epoch_metrics),
        'epoch_metrics': epoch_metrics
    }

def plot_model_comparison(metrics_dict):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['Word2Vec', 'GloVe', 'BERT']
    x = np.arange(len(models))
    width = 0.35
    
    headlines_scores = []
    tweets_scores = []
    
    for model in models:
        if metrics_dict[model]['headlines']:
            headlines_scores.append(metrics_dict[model]['headlines']['f1'])
        else:
            headlines_scores.append(0)
        
        if metrics_dict[model]['tweets']:
            tweets_scores.append(metrics_dict[model]['tweets']['f1'])
        else:
            tweets_scores.append(0)
    
    bars1 = ax.bar(x - width/2, headlines_scores, width, label='Headlines', 
                   color=DATASET_COLORS['headlines'], alpha=0.8)
    bars2 = ax.bar(x + width/2, tweets_scores, width, label='Tweets', 
                   color=DATASET_COLORS['tweets'], alpha=0.8)
    
    ax.set_xlabel('Embedding Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Comparison: F1-Score Across Embeddings and Datasets', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # bar labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('1_model_comparison_f1_score.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_metric_breakdown(metrics_dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    models = ['Word2Vec', 'GloVe', 'BERT']
    
    x = np.arange(len(models))
    width = 0.35
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        headlines_values = []
        tweets_values = []
        
        for model in models:
            if metrics_dict[model]['headlines']:
                headlines_values.append(metrics_dict[model]['headlines'][metric])
            else:
                headlines_values.append(0)
            
            if metrics_dict[model]['tweets']:
                tweets_values.append(metrics_dict[model]['tweets'][metric])
            else:
                tweets_values.append(0)
        
        bars1 = ax.bar(x - width/2, headlines_values, width, label='Headlines',
                      color=DATASET_COLORS['headlines'], alpha=0.8)
        bars2 = ax.bar(x + width/2, tweets_values, width, label='Tweets',
                      color=DATASET_COLORS['tweets'], alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}',
                           ha='center', va='bottom', fontsize=8)
    
    plt.suptitle('Metric Breakdown: Model Performance Across All Metrics', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('2_metric_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(metrics_dict):
    # 2 rows (headlines, tweets) x 3 cols (models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    models = ['Word2Vec', 'GloVe', 'BERT']
    datasets = ['headlines', 'tweets']
    
    for dataset_idx, dataset in enumerate(datasets):
        for model_idx, model in enumerate(models):
            ax = axes[dataset_idx, model_idx]
            data = metrics_dict[model][dataset]
            
            if data:
                cm = np.array([[data['tn'], data['fp']],
                              [data['fn'], data['tp']]])
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Not Sarcastic', 'Sarcastic'],
                           yticklabels=['Not Sarcastic', 'Sarcastic'],
                           cbar_kws={'label': 'Count'}, vmin=0)
                
                ax.set_title(f'{model} - {dataset.title()}', 
                            fontsize=11, fontweight='bold', pad=10)
                ax.set_xlabel('Predicted', fontsize=9, fontweight='bold')
                ax.set_ylabel('True', fontsize=9, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', fontsize=11)
                ax.set_title(f'{model} - {dataset.title()}', 
                            fontsize=11, fontweight='bold')
                ax.axis('off')
    
    # row labels on left side
    for idx, dataset in enumerate(datasets):
        fig.text(0.02, 0.75 - idx*0.5, dataset.title(), 
                fontsize=12, fontweight='bold', rotation=90, 
                ha='center', va='center')
    
    plt.suptitle('Confusion Matrix Heatmaps: True vs Predicted Labels by Dataset', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig('3_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves(metrics_dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    
    models = ['Word2Vec', 'GloVe', 'BERT']
    
    for model in models:
        data = metrics_dict[model]['headlines']
        if data and 'epoch_metrics' in data:
            epochs = [e['epoch'] for e in data['epoch_metrics']]
            train_acc = [e['train_accuracy'] for e in data['epoch_metrics']]
            val_acc = [e['val_accuracy'] for e in data['epoch_metrics']]
            train_loss = [e['train_loss'] for e in data['epoch_metrics']]
            val_loss = [e['val_loss'] for e in data['epoch_metrics']]
            
            ax1.plot(epochs, train_acc, label=f'{model} Train', 
                    color=COLORS[model], linestyle='-', linewidth=2, alpha=0.8)
            ax1.plot(epochs, val_acc, label=f'{model} Val', 
                    color=COLORS[model], linestyle='--', linewidth=2, alpha=0.8)
            
            ax2.plot(epochs, train_loss, label=f'{model} Train', 
                    color=COLORS[model], linestyle='-', linewidth=2, alpha=0.8)
            ax2.plot(epochs, val_loss, label=f'{model} Val', 
                    color=COLORS[model], linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Headlines: Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Headlines: Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, ncol=3)
    ax2.grid(alpha=0.3)
    
    # individual model curves
    for idx, model in enumerate(['Word2Vec', 'GloVe']):
        ax = axes[1, idx]
        data = metrics_dict[model]['headlines']
        if data and 'epoch_metrics' in data:
            epochs = [e['epoch'] for e in data['epoch_metrics']]
            train_acc = [e['train_accuracy'] for e in data['epoch_metrics']]
            val_acc = [e['val_accuracy'] for e in data['epoch_metrics']]
            train_loss = [e['train_loss'] for e in data['epoch_metrics']]
            val_loss = [e['val_loss'] for e in data['epoch_metrics']]
            
            ax2_twin = ax.twinx()
            
            line1 = ax.plot(epochs, train_acc, 'o-', label='Train Accuracy', 
                           color=COLORS[model], linewidth=2, markersize=4)
            line2 = ax.plot(epochs, val_acc, 's--', label='Val Accuracy', 
                           color=COLORS[model], linewidth=2, markersize=4, alpha=0.7)
            
            line3 = ax2_twin.plot(epochs, train_loss, '^-', label='Train Loss', 
                                 color='red', linewidth=2, markersize=4, alpha=0.7)
            line4 = ax2_twin.plot(epochs, val_loss, 'v--', label='Val Loss', 
                                 color='orange', linewidth=2, markersize=4, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold', color=COLORS[model])
            ax2_twin.set_ylabel('Loss', fontsize=10, fontweight='bold', color='red')
            ax.set_title(f'{model} - Headlines', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])
            
            lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=8, loc='center right')
    
    plt.suptitle('Training Curves - Headlines Dataset', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('4_training_curves_headlines.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_curves_tweets(metrics_dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    ax1 = axes[0, 0]
    ax2 = axes[0, 1]
    
    models = ['Word2Vec', 'GloVe', 'BERT']
    
    for model in models:
        data = metrics_dict[model]['tweets']
        if data and 'epoch_metrics' in data:
            epochs = [e['epoch'] for e in data['epoch_metrics']]
            train_acc = [e['train_accuracy'] for e in data['epoch_metrics']]
            val_acc = [e['val_accuracy'] for e in data['epoch_metrics']]
            train_loss = [e['train_loss'] for e in data['epoch_metrics']]
            val_loss = [e['val_loss'] for e in data['epoch_metrics']]
            
            ax1.plot(epochs, train_acc, label=f'{model} Train', 
                    color=COLORS[model], linestyle='-', linewidth=2, alpha=0.8)
            ax1.plot(epochs, val_acc, label=f'{model} Val', 
                    color=COLORS[model], linestyle='--', linewidth=2, alpha=0.8)
            
            ax2.plot(epochs, train_loss, label=f'{model} Train', 
                    color=COLORS[model], linestyle='-', linewidth=2, alpha=0.8)
            ax2.plot(epochs, val_loss, label=f'{model} Val', 
                    color=COLORS[model], linestyle='--', linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Tweets: Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, ncol=3)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Tweets: Training vs Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, ncol=3)
    ax2.grid(alpha=0.3)
    
    for idx, model in enumerate(['Word2Vec', 'GloVe']):
        ax = axes[1, idx]
        data = metrics_dict[model]['tweets']
        if data and 'epoch_metrics' in data:
            epochs = [e['epoch'] for e in data['epoch_metrics']]
            train_acc = [e['train_accuracy'] for e in data['epoch_metrics']]
            val_acc = [e['val_accuracy'] for e in data['epoch_metrics']]
            train_loss = [e['train_loss'] for e in data['epoch_metrics']]
            val_loss = [e['val_loss'] for e in data['epoch_metrics']]
            
            ax2_twin = ax.twinx()
            
            line1 = ax.plot(epochs, train_acc, 'o-', label='Train Accuracy', 
                           color=COLORS[model], linewidth=2, markersize=4)
            line2 = ax.plot(epochs, val_acc, 's--', label='Val Accuracy', 
                           color=COLORS[model], linewidth=2, markersize=4, alpha=0.7)
            
            line3 = ax2_twin.plot(epochs, train_loss, '^-', label='Train Loss', 
                                 color='red', linewidth=2, markersize=4, alpha=0.7)
            line4 = ax2_twin.plot(epochs, val_loss, 'v--', label='Val Loss', 
                                 color='orange', linewidth=2, markersize=4, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=10, fontweight='bold')
            ax.set_ylabel('Accuracy', fontsize=10, fontweight='bold', color=COLORS[model])
            ax2_twin.set_ylabel('Loss', fontsize=10, fontweight='bold', color='red')
            ax.set_title(f'{model} - Tweets', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            ax.set_ylim([0, 1])
            
            lines = line1 + line2 + line3 + line4
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, fontsize=8, loc='center right')
    
    plt.suptitle('Training Curves - Tweets Dataset', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('5_training_curves_tweets.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_dataset_distribution():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    datasets = ['Headlines\n(26,709 samples)', 'Tweets\n(20,000 samples)']
    sizes = [26709, 20000]
    colors = [DATASET_COLORS['headlines'], DATASET_COLORS['tweets']]
    
    wedges, texts, autotexts = ax.pie(sizes, labels=datasets, autopct='%1.1f%%',
                                       colors=colors, explode=(0.03, 0.03),
                                       shadow=True, startangle=90,
                                       textprops={'fontsize': 14, 'fontweight': 'bold'})
    
    for autotext in autotexts:
        autotext.set_fontsize(16)
        autotext.set_fontweight('bold')
    
    ax.set_title('Dataset Size Distribution\nTotal: 46,709 samples', 
                 fontsize=18, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('0_dataset_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_class_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    class_labels = ['Not Sarcastic', 'Sarcastic']
    class_colors = ['#4ECDC4', '#FF6B6B']
    
    # headlines
    ax1 = axes[0]
    headlines_sarcastic = int(26709 * 0.439)
    headlines_not_sarcastic = 26709 - headlines_sarcastic
    headlines_classes = [headlines_not_sarcastic, headlines_sarcastic]
    
    wedges1, texts1, autotexts1 = ax1.pie(headlines_classes, labels=class_labels, 
                                           autopct='%1.1f%%', colors=class_colors,
                                           explode=(0.02, 0.02), shadow=True, startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts1:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax1.set_title('Headlines Dataset\nClass Distribution', 
                  fontsize=16, fontweight='bold', pad=15)
    
    legend_labels1 = [f'{l}: {c:,}' for l, c in zip(class_labels, headlines_classes)]
    ax1.legend(wedges1, legend_labels1, loc='lower center', fontsize=11,
               bbox_to_anchor=(0.5, -0.05))
    
    # tweets
    ax2 = axes[1]
    tweets_sarcastic = int(20000 * 0.251)
    tweets_not_sarcastic = 20000 - tweets_sarcastic
    tweets_classes = [tweets_not_sarcastic, tweets_sarcastic]
    
    wedges2, texts2, autotexts2 = ax2.pie(tweets_classes, labels=class_labels, 
                                           autopct='%1.1f%%', colors=class_colors,
                                           explode=(0.02, 0.02), shadow=True, startangle=90,
                                           textprops={'fontsize': 12, 'fontweight': 'bold'})
    for autotext in autotexts2:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
    
    ax2.set_title('Tweets Dataset\nClass Distribution', 
                  fontsize=16, fontweight='bold', pad=15)
    
    legend_labels2 = [f'{l}: {c:,}' for l, c in zip(class_labels, tweets_classes)]
    ax2.legend(wedges2, legend_labels2, loc='lower center', fontsize=11,
               bbox_to_anchor=(0.5, -0.05))
    
    plt.suptitle('Class Distribution: Headlines vs Tweets', 
                 fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('0_class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_domain_comparison(metrics_dict):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    models = ['Word2Vec', 'GloVe', 'BERT']
    datasets = ['headlines', 'tweets']
    
    x = np.arange(len(datasets))
    width = 0.25
    
    for idx, model in enumerate(models):
        values = []
        for dataset in datasets:
            if metrics_dict[model][dataset]:
                values.append(metrics_dict[model][dataset]['f1'])
            else:
                values.append(0)
        
        offset = (idx - 1) * width
        bars = ax.bar(x + offset, values, width, label=model, 
                     color=COLORS[model], alpha=0.8)
        
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Dataset Domain', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('Domain Performance Comparison: Cross-Domain Generalization', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in datasets])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('6_domain_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    metrics_dict = load_metrics()
    
    plot_dataset_distribution()
    plot_class_distribution()
    plot_model_comparison(metrics_dict)
    plot_metric_breakdown(metrics_dict)
    plot_confusion_matrices(metrics_dict)
    plot_training_curves(metrics_dict)
    plot_training_curves_tweets(metrics_dict)
    plot_domain_comparison(metrics_dict)

if __name__ == "__main__":
    main()

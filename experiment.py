"""
Main experiment script for comparing LSTM and QLSTM on NASA SMAP-MSL dataset.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

from data_loader import NASADataLoader
from baseline_lstm import BaselineLSTM, plot_results, plot_confusion_matrix
from qlstm import QLSTM

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def create_output_dirs():
    """Create output directories for results."""
    dirs = ['results', 'results/plots', 'results/models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def save_metrics(metrics, model_name, channel, output_dir='results'):
    """Save metrics to JSON file."""
    metrics_to_save = {k: v for k, v in metrics.items() if k != 'confusion_matrix'}
    metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
    
    filename = f"{output_dir}/{model_name}_{channel}_metrics.json"
    with open(filename, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    
    print(f"Metrics saved to {filename}")


def print_metrics(metrics, model_name):
    """Print evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"{model_name} Results")
    print(f"{'='*60}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Threshold: {metrics['threshold']:.4f}")
    print(f"\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    print(f"{'='*60}\n")


def plot_comparison(lstm_metrics, qlstm_metrics, channel, save_path=None):
    """Plot comparison between LSTM and QLSTM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Metrics comparison
    metrics_names = ['Precision', 'Recall', 'F1-Score', 'AUC']
    lstm_values = [
        lstm_metrics['precision'],
        lstm_metrics['recall'],
        lstm_metrics['f1_score'],
        lstm_metrics['auc']
    ]
    qlstm_values = [
        qlstm_metrics['precision'],
        qlstm_metrics['recall'],
        qlstm_metrics['f1_score'],
        qlstm_metrics['auc']
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, lstm_values, width, label='LSTM', alpha=0.8)
    axes[0, 0].bar(x + width/2, qlstm_values, width, label='QLSTM', alpha=0.8)
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title(f'Performance Metrics Comparison - Channel {channel}')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics_names)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 1])
    
    # LSTM Confusion Matrix
    sns.heatmap(lstm_metrics['confusion_matrix'], annot=True, fmt='d', 
                cmap='Blues', ax=axes[0, 1])
    axes[0, 1].set_title('LSTM Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('True')
    
    # QLSTM Confusion Matrix
    sns.heatmap(qlstm_metrics['confusion_matrix'], annot=True, fmt='d', 
                cmap='Greens', ax=axes[1, 0])
    axes[1, 0].set_title('QLSTM Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # Summary table
    axes[1, 1].axis('off')
    summary_data = [
        ['Metric', 'LSTM', 'QLSTM', 'Improvement'],
        ['Precision', f"{lstm_metrics['precision']:.4f}", 
         f"{qlstm_metrics['precision']:.4f}",
         f"{(qlstm_metrics['precision'] - lstm_metrics['precision'])*100:.2f}%"],
        ['Recall', f"{lstm_metrics['recall']:.4f}", 
         f"{qlstm_metrics['recall']:.4f}",
         f"{(qlstm_metrics['recall'] - lstm_metrics['recall'])*100:.2f}%"],
        ['F1-Score', f"{lstm_metrics['f1_score']:.4f}", 
         f"{qlstm_metrics['f1_score']:.4f}",
         f"{(qlstm_metrics['f1_score'] - lstm_metrics['f1_score'])*100:.2f}%"],
        ['AUC', f"{lstm_metrics['auc']:.4f}", 
         f"{qlstm_metrics['auc']:.4f}",
         f"{(qlstm_metrics['auc'] - lstm_metrics['auc'])*100:.2f}%"]
    ]
    
    table = axes[1, 1].table(cellText=summary_data, cellLoc='center', loc='center',
                            colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.close()
    
    return fig


def run_experiment(channel='P-1', sequence_length=50, epochs=30):
    """
    Run complete experiment comparing LSTM and QLSTM.
    
    Args:
        channel: Data channel to use
        sequence_length: Length of input sequences
        epochs: Number of training epochs
    """
    print(f"\n{'#'*80}")
    print(f"Starting Experiment: LSTM vs QLSTM on NASA SMAP-MSL Dataset")
    print(f"Channel: {channel} | Sequence Length: {sequence_length} | Epochs: {epochs}")
    print(f"{'#'*80}\n")
    
    # Create output directories
    create_output_dirs()
    
    # Load data
    print("Loading data...")
    loader = NASADataLoader()
    X_train, y_train, X_test, y_test, labels = loader.load_data(
        channel=channel, 
        sequence_length=sequence_length
    )
    
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Anomalies in test set: {np.sum(labels)} / {len(labels)} ({100*np.sum(labels)/len(labels):.2f}%)")
    
    # Get input shape
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Train and evaluate Baseline LSTM
    print("\n" + "="*80)
    print("Training Baseline LSTM...")
    print("="*80)
    
    lstm_model = BaselineLSTM(input_shape=input_shape, lstm_units=80)
    lstm_model.train(X_train, y_train, epochs=epochs, batch_size=128)
    
    print("\nEvaluating Baseline LSTM...")
    lstm_metrics, lstm_predictions, lstm_errors = lstm_model.evaluate(
        X_test, y_test, labels
    )
    print_metrics(lstm_metrics, "Baseline LSTM")
    save_metrics(lstm_metrics, "lstm", channel)
    
    # Save LSTM plots
    lstm_model.plot_training_history(
        save_path=f'results/plots/lstm_{channel}_training.png'
    )
    plot_results(
        lstm_errors, labels, lstm_predictions,
        save_path=f'results/plots/lstm_{channel}_results.png'
    )
    plot_confusion_matrix(
        lstm_metrics['confusion_matrix'],
        save_path=f'results/plots/lstm_{channel}_confusion.png'
    )
    
    # Train and evaluate QLSTM
    print("\n" + "="*80)
    print("Training QLSTM (Quantum LSTM)...")
    print("="*80)
    
    qlstm_model = QLSTM(
        input_shape=input_shape, 
        lstm_units=80, 
        n_qubits=4, 
        n_layers=2
    )
    qlstm_model.train(X_train, y_train, epochs=epochs, batch_size=64)
    
    print("\nEvaluating QLSTM...")
    qlstm_metrics, qlstm_predictions, qlstm_errors = qlstm_model.evaluate(
        X_test, y_test, labels
    )
    print_metrics(qlstm_metrics, "QLSTM")
    save_metrics(qlstm_metrics, "qlstm", channel)
    
    # Save QLSTM plots
    qlstm_model.plot_training_history(
        save_path=f'results/plots/qlstm_{channel}_training.png'
    )
    plot_results(
        qlstm_errors, labels, qlstm_predictions,
        save_path=f'results/plots/qlstm_{channel}_results.png'
    )
    plot_confusion_matrix(
        qlstm_metrics['confusion_matrix'],
        save_path=f'results/plots/qlstm_{channel}_confusion.png'
    )
    
    # Create comparison plot
    print("\nGenerating comparison plots...")
    plot_comparison(
        lstm_metrics, qlstm_metrics, channel,
        save_path=f'results/plots/comparison_{channel}.png'
    )
    
    # Print final summary
    print("\n" + "#"*80)
    print("EXPERIMENT SUMMARY")
    print("#"*80)
    print(f"\nDataset: NASA SMAP-MSL (Channel: {channel})")
    print(f"Sequence Length: {sequence_length}")
    print(f"Training Epochs: {epochs}")
    print(f"\nBaseline LSTM:")
    print(f"  - F1-Score: {lstm_metrics['f1_score']:.4f}")
    print(f"  - Precision: {lstm_metrics['precision']:.4f}")
    print(f"  - Recall: {lstm_metrics['recall']:.4f}")
    print(f"  - AUC: {lstm_metrics['auc']:.4f}")
    print(f"\nQLSTM (Quantum LSTM):")
    print(f"  - F1-Score: {qlstm_metrics['f1_score']:.4f}")
    print(f"  - Precision: {qlstm_metrics['precision']:.4f}")
    print(f"  - Recall: {qlstm_metrics['recall']:.4f}")
    print(f"  - AUC: {qlstm_metrics['auc']:.4f}")
    print(f"\nImprovement (QLSTM vs LSTM):")
    print(f"  - F1-Score: {(qlstm_metrics['f1_score'] - lstm_metrics['f1_score'])*100:+.2f}%")
    print(f"  - Precision: {(qlstm_metrics['precision'] - lstm_metrics['precision'])*100:+.2f}%")
    print(f"  - Recall: {(qlstm_metrics['recall'] - lstm_metrics['recall'])*100:+.2f}%")
    print(f"  - AUC: {(qlstm_metrics['auc'] - lstm_metrics['auc'])*100:+.2f}%")
    print(f"\nResults saved to: results/")
    print("#"*80 + "\n")
    
    return lstm_metrics, qlstm_metrics


if __name__ == "__main__":
    # Run experiment on a sample channel
    lstm_metrics, qlstm_metrics = run_experiment(
        channel='P-1',
        sequence_length=50,
        epochs=30
    )

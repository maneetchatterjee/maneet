"""
Baseline LSTM model for NASA SMAP-MSL anomaly detection.
Based on the approach from khundman/telemanom.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


class BaselineLSTM:
    """Baseline LSTM model for anomaly detection."""
    
    def __init__(self, input_shape, lstm_units=80, learning_rate=0.001):
        """
        Initialize LSTM model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            lstm_units: Number of LSTM units
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self):
        """Build LSTM model architecture."""
        model = keras.Sequential([
            layers.LSTM(self.lstm_units, input_shape=self.input_shape, 
                       return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units),
            layers.Dropout(0.2),
            layers.Dense(self.input_shape[1])
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=128, validation_split=0.1):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def compute_errors(self, X, y):
        """Compute reconstruction errors."""
        predictions = self.predict(X)
        errors = np.mean(np.abs(predictions - y), axis=1)
        return errors
    
    def detect_anomalies(self, X, y, threshold=None, percentile=95):
        """
        Detect anomalies based on reconstruction error.
        
        Args:
            X: Input sequences
            y: True values
            threshold: Error threshold (if None, use percentile)
            percentile: Percentile for threshold calculation
            
        Returns:
            predictions: Binary predictions (0: normal, 1: anomaly)
            errors: Reconstruction errors
            threshold: Threshold used
        """
        errors = self.compute_errors(X, y)
        
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        predictions = (errors > threshold).astype(int)
        
        return predictions, errors, threshold
    
    def evaluate(self, X_test, y_test, labels, threshold=None):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            labels: True anomaly labels
            threshold: Error threshold
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        predictions, errors, threshold = self.detect_anomalies(
            X_test, y_test, threshold
        )
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels, errors)
        except:
            auc = 0.0
        
        cm = confusion_matrix(labels, predictions)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'threshold': threshold,
            'confusion_matrix': cm
        }
        
        return metrics, predictions, errors
    
    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
    
    def load(self, filepath):
        """Load model."""
        self.model = keras.models.load_model(filepath)
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.close()
        
        return fig


def plot_results(errors, labels, predictions, save_path=None):
    """Plot anomaly detection results."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    # Plot errors
    time_steps = np.arange(len(errors))
    axes[0].plot(time_steps, errors, label='Reconstruction Error', alpha=0.7)
    axes[0].scatter(time_steps[labels == 1], errors[labels == 1], 
                   color='red', label='True Anomalies', s=10, alpha=0.5)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Reconstruction Error')
    axes[0].set_title('Reconstruction Error and True Anomalies')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot predictions
    axes[1].plot(time_steps, labels, label='True Labels', alpha=0.7)
    axes[1].plot(time_steps, predictions, label='Predictions', alpha=0.7, linestyle='--')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Anomaly Label')
    axes[1].set_title('True Labels vs Predictions')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    
    plt.close()
    
    return fig


def plot_confusion_matrix(cm, save_path=None):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.close()
    
    return fig

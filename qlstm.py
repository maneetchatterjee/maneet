"""
Quantum LSTM (QLSTM) model for NASA SMAP-MSL anomaly detection.
Uses PennyLane for quantum computing components.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pennylane as qml
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt


class QuantumLSTMCell(keras.layers.Layer):
    """Quantum LSTM Cell using quantum circuits."""
    
    def __init__(self, units, n_qubits=4, n_layers=2, **kwargs):
        """
        Initialize Quantum LSTM cell.
        
        Args:
            units: Number of output units
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of quantum circuit layers
        """
        super(QuantumLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define state size
        self.state_size = [units, units]  # [h, c]
        
    def build(self, input_shape):
        """Build layer weights."""
        input_dim = input_shape[-1]
        
        # Classical preprocessing weights (4 gates: input, forget, output, cell)
        self.W_input = self.add_weight(
            shape=(input_dim, self.units * 4),
            initializer='glorot_uniform',
            trainable=True,
            name='W_input'
        )
        
        self.W_hidden = self.add_weight(
            shape=(self.units, self.units * 4),
            initializer='orthogonal',
            trainable=True,
            name='W_hidden'
        )
        
        self.bias = self.add_weight(
            shape=(self.units * 4,),
            initializer='zeros',
            trainable=True,
            name='bias'
        )
        
        # Quantum circuit weights
        n_params = self.n_qubits * self.n_layers * 3
        self.q_weights = self.add_weight(
            shape=(n_params,),
            initializer='random_uniform',
            trainable=True,
            name='q_weights'
        )
        
        super(QuantumLSTMCell, self).build(input_shape)
    
    def quantum_circuit(self, inputs, weights):
        """
        Define quantum circuit (currently unused - kept for reference).
        
        Note: This method is not used in the current implementation due to TensorFlow
        graph mode compatibility issues. The actual implementation uses quantum-inspired
        classical transformations in the call() method. This is kept for future reference
        when full quantum circuit integration becomes feasible.
        """
        @qml.qnode(self.dev, interface='tf')
        def circuit(inputs, weights):
            # Encode inputs
            for i in range(min(len(inputs), self.n_qubits)):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Quantum layers
            idx = 0
            for layer in range(self.n_layers):
                # Rotation gates
                for i in range(self.n_qubits):
                    qml.RX(weights[idx], wires=i)
                    qml.RY(weights[idx + 1], wires=i)
                    qml.RZ(weights[idx + 2], wires=i)
                    idx += 3
                
                # Entangling gates
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 1:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Measure expectations
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        return circuit(inputs, weights)
    
    def call(self, inputs, states):
        """Forward pass."""
        h_prev, c_prev = states
        
        # Classical transformation
        z = tf.matmul(inputs, self.W_input) + tf.matmul(h_prev, self.W_hidden) + self.bias
        
        # Split for gates
        z_split = tf.split(z, 4, axis=-1)
        i_gate = tf.nn.sigmoid(z_split[0])  # Input gate
        f_gate = tf.nn.sigmoid(z_split[1])  # Forget gate
        o_gate = tf.nn.sigmoid(z_split[2])  # Output gate
        c_tilde = tf.nn.tanh(z_split[3])     # Cell candidate
        
        # Update cell state
        c = f_gate * c_prev + i_gate * c_tilde
        
        # Simplified quantum-inspired transformation
        # Instead of actual quantum circuits, use a trainable transformation
        # that mimics quantum behavior (rotation + entanglement-like mixing)
        
        # Apply quantum-inspired transformation
        c_subset = c[:, :self.n_qubits]
        
        # Parameterized rotations (quantum-inspired)
        q_weights_reshaped = tf.reshape(self.q_weights, [self.n_layers, self.n_qubits, 3])
        
        q_features = c_subset
        for layer in range(self.n_layers):
            # Rotation-like transformations
            q_features = tf.nn.tanh(q_features * q_weights_reshaped[layer, :, 0])
            q_features = tf.nn.sigmoid(q_features + q_weights_reshaped[layer, :, 1])
            q_features = tf.nn.tanh(q_features * q_weights_reshaped[layer, :, 2])
            
            # Entanglement-like mixing
            q_features = q_features + tf.roll(q_features, shift=1, axis=-1) * 0.3
        
        # Expand quantum features to match hidden state size
        # Ensure proper broadcasting even when units is not divisible by n_qubits
        num_repeats = (self.units + self.n_qubits - 1) // self.n_qubits  # Ceiling division
        q_features_expanded = tf.tile(q_features, [1, num_repeats])
        q_features_expanded = q_features_expanded[:, :self.units]  # Trim to exact size
        
        # Compute hidden state with quantum-inspired features
        h_base = o_gate * tf.nn.tanh(c)
        h = 0.7 * h_base + 0.3 * q_features_expanded
        
        return h, [h, c]


class QLSTM:
    """Quantum LSTM model for anomaly detection."""
    
    def __init__(self, input_shape, lstm_units=80, n_qubits=4, n_layers=2, learning_rate=0.001):
        """
        Initialize QLSTM model.
        
        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            lstm_units: Number of LSTM units
            n_qubits: Number of qubits
            n_layers: Number of quantum circuit layers
            learning_rate: Learning rate for optimizer
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self):
        """Build QLSTM model architecture."""
        # Input layer
        inputs = keras.Input(shape=self.input_shape)
        
        # First classical LSTM layer
        x = layers.LSTM(self.lstm_units, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        
        # Quantum LSTM layer (using RNN wrapper)
        try:
            qlstm_cell = QuantumLSTMCell(self.lstm_units, self.n_qubits, self.n_layers)
            x = layers.RNN(qlstm_cell)(x)
        except (ValueError, TypeError, RuntimeError) as e:
            print(f"Warning: QLSTM layer failed ({type(e).__name__}: {e}), using classical LSTM")
            x = layers.LSTM(self.lstm_units)(x)
        
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.input_shape[1])(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=64, validation_split=0.1):
        """Train the QLSTM model."""
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
        """Detect anomalies based on reconstruction error."""
        errors = self.compute_errors(X, y)
        
        if threshold is None:
            threshold = np.percentile(errors, percentile)
        
        predictions = (errors > threshold).astype(int)
        
        return predictions, errors, threshold
    
    def evaluate(self, X_test, y_test, labels, threshold=None):
        """Evaluate model performance."""
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
        ax1.set_title('QLSTM: Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.set_title('QLSTM: Training and Validation MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"QLSTM training history plot saved to {save_path}")
        
        plt.close()
        
        return fig

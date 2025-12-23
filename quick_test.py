"""
Quick test experiment with reduced epochs for faster execution.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from experiment import run_experiment

if __name__ == "__main__":
    print("Running quick test experiment...")
    print("Using reduced epochs (15) for faster execution")
    
    # Run experiment with fewer epochs
    lstm_metrics, qlstm_metrics = run_experiment(
        channel='P-1',
        sequence_length=50,
        epochs=15  # Reduced from 30 for testing
    )
    
    print("\n" + "="*80)
    print("Quick test completed successfully!")
    print("="*80)

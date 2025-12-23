"""
Quick test experiment with reduced epochs for faster execution.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

from experiment import run_experiment

# Configuration
QUICK_TEST_EPOCHS = 15  # Reduced from 30 for faster testing

if __name__ == "__main__":
    print("Running quick test experiment...")
    print(f"Using reduced epochs ({QUICK_TEST_EPOCHS}) for faster execution")
    
    # Run experiment with fewer epochs
    lstm_metrics, qlstm_metrics = run_experiment(
        channel='P-1',
        sequence_length=50,
        epochs=QUICK_TEST_EPOCHS
    )
    
    print("\n" + "="*80)
    print("Quick test completed successfully!")
    print("="*80)

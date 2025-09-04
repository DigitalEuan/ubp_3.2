"""
Universal Binary Principle (UBP) Framework v3.2+ - Quantum Operations: Detect Anomaly
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
# Corrected import: nrci is in metrics.py
from metrics import nrci
import numpy as np

def detect_anomaly_ubp(historical_data, live_signal, threshold=0.999):
    """
    Use NRCI to detect deviations from expected coherent patterns.
    """
    anomalies = []
    # Ensure both inputs are numpy arrays for consistent slicing and operations
    historical_data = np.asarray(historical_data, dtype=float)
    live_signal = np.asarray(live_signal, dtype=float)

    # Pad historical data to match live_signal length for slicing, or iterate only over overlapping parts.
    # The current implementation iterates over segments of live_signal that match historical_data length.
    # This is fine, but we should make sure the NRCI function handles inputs correctly.
    
    # Iterate through live_signal, taking segments of the same length as historical_data
    for i in range(len(live_signal) - len(historical_data) + 1):
        segment = live_signal[i:i+len(historical_data)]
        
        # Ensure segment and historical_data have the same length for NRCI
        if len(segment) == len(historical_data):
            # NRCI expects lists of floats, so convert numpy arrays to lists
            coherence = nrci(segment.tolist(), historical_data.tolist())
            if coherence < threshold:
                anomalies.append((i, coherence))
    return anomalies

# Example
np.random.seed(42) # for reproducibility
baseline = np.sin(np.linspace(0, 4*np.pi, 50)) + 0.1 * np.random.randn(50)

# Inject noise/anomaly at a specific point
anomaly_start_index = 50
noise_injection = np.random.randn(10) * 5 # Significantly larger noise
live = np.concatenate([baseline, noise_injection])  # Inject noise

print("Running anomaly detection...")
anomalies = detect_anomaly_ubp(baseline, live)

if anomalies:
    print("\nAnomaly Detection Results:")
    for idx, coherence_val in anomalies:
        print(f"Anomaly detected at index {idx}, NRCI: {coherence_val:.6f}")
else:
    print("No anomalies detected.")
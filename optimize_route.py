"""
Universal Binary Principle (UBP) Framework v3.2+ - Quantum Operations: Optimize Route
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

This module implements a TSP solver guided by UBP resonance and entanglement
operations to explore optimal paths, with high NRCI indicating coherent,
stable solutions.
"""

# Corrected imports for UBP components
from state import OffBit
from toggle_ops import resonance_toggle, entanglement_toggle # nrci is not directly used for OffBit operations here
import numpy as np

# Define a constant for the maximum 24-bit value for normalization
MAX_24_BIT_VALUE = (2**24 - 1)

def solve_tsp_ubp(distances: np.ndarray, epochs: int = 100, mutation_strength: float = 0.1):
    """
    Use UBP resonance and entanglement to explore optimal paths for TSP.
    High NRCI indicates coherent, stable solutions.
    
    This version attempts to use UBP operations to guide path mutations.

    Args:
        distances (np.ndarray): A 2D numpy array representing the distance matrix between cities.
        epochs (int): The number of iterations for the optimization.
        mutation_strength (float): A factor controlling the intensity of path mutations.

    Returns:
        Tuple[np.ndarray, float]: A tuple containing the best path found and its corresponding cost.
    """
    n_cities = len(distances)
    if n_cities < 2:
        return np.array([0]), 0.0 # Handle single or zero cities gracefully

    current_path = np.random.permutation(n_cities) # Start with a random path
    current_score = sum(distances[current_path[i], current_path[(i+1)%n_cities]] for i in range(n_cities))
    
    best_path = current_path.copy()
    best_score = current_score

    print(f"Initial path cost: {best_score:.4f}")

    for epoch in range(epochs):
        # Use OffBits to encode path characteristics or guide mutation
        # Encode a seed based on the current path score.
        # Ensure seed_val is within the 24-bit range.
        seed_val = int((current_score * 100000) % MAX_24_BIT_VALUE)
        seed = OffBit(seed_val)
        
        # Use resonance to create a perturbed state for mutation
        # frequency and time_param are critical for the resonance calculation.
        # time parameter should ideally be non-zero for resonance_toggle.
        res_time = epoch / epochs + 1e-6 # Ensure time is never exactly zero for resonance
        perturbed_offbit = resonance_toggle(seed, frequency=0.1, time=res_time)
        
        # Decode OffBit to guide path mutation
        # Use the value of the perturbed OffBit, normalized, to determine mutation probability.
        # This provides a UBP-guided adaptive mutation rate.
        mutation_prob = (perturbed_offbit.value / MAX_24_BIT_VALUE) * mutation_strength
        mutation_prob = np.clip(mutation_prob, 0.0, 1.0) # Ensure mutation_prob is between 0 and 1
        
        new_path = current_path.copy()
        if np.random.rand() < mutation_prob:
            # Perform a simple swap mutation
            i, j = np.random.choice(n_cities, 2, replace=False)
            new_path[i], new_path[j] = new_path[j], new_path[i]
        
        new_score = sum(distances[new_path[i], new_path[(i+1)%n_cities]] for i in range(n_cities))
        
        # Use entanglement to decide whether to accept the new path
        # Higher coherence -> more likely to accept better solutions, or
        # accept worse solutions with a certain probability (simulated annealing-like).
        
        # Scale scores to be distinct OffBit values, clamp to 24-bit range.
        offbit_current_score = OffBit(int(np.clip(current_score * 100, 0, MAX_24_BIT_VALUE)))
        offbit_new_score = OffBit(int(np.clip(new_score * 100, 0, MAX_24_BIT_VALUE)))
        
        # entanglement_toggle returns an OffBit, get its value and normalize for coherence.
        # The coherence parameter in entanglement_toggle sets the baseline for the operation itself.
        coherence_from_entanglement = entanglement_toggle(offbit_current_score, offbit_new_score, coherence=0.95).value / MAX_24_BIT_VALUE
        coherence_from_entanglement = np.clip(coherence_from_entanglement, 0.0, 1.0) # Ensure 0-1 range

        # Acceptance criteria: accept better paths, or accept worse paths with a probability
        # modulated by entanglement coherence.
        if new_score < current_score or np.random.rand() < coherence_from_entanglement * 0.1: # Small probability of accepting worse
            current_path = new_path
            current_score = new_score
            
            if new_score < best_score:
                best_path = new_path
                best_score = new_score
                print(f"Epoch {epoch+1}/{epochs}: New best path found with cost {best_score:.4f}")

    return best_path, best_score

class OptimizeRoute:
    """
    Wrapper class to run the TSP optimization with UBP-guided mutations.
    """
    def run(self):
        """
        Main execution function for the TSP optimization.
        """
        print("Solving TSP with UBP-guided mutations...")

        # Generate a random distance matrix for 10 cities
        np.random.seed(42) # for reproducibility
        num_cities = 10
        distances = np.random.rand(num_cities, num_cities) * 100 # Scale distances for more meaningful scores
        
        # Make distance matrix symmetric and zero diagonal
        for i in range(num_cities):
            distances[i, i] = 0
            for j in range(i + 1, num_cities):
                distances[j, i] = distances[i, j]

        path, cost = solve_tsp_ubp(distances, epochs=500, mutation_strength=0.5)
        
        print(f"\nFinal Optimized path: {path}")
        print(f"Final Path Cost: {cost:.4f}")

if __name__ == '__main__':
    # Standalone execution for testing
    optimizer = OptimizeRoute()
    optimizer.run()
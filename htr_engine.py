"""
Universal Binary Principle (UBP) Framework v3.2+ - HTR Engine
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional

# Import necessary UBP constants for calculations
from ubp_config import get_config, UBPConfig

# Initialize configuration
_config: UBPConfig = get_config()

class HTREngine:
    """
    Harmonic Toggle Resonance (HTR) Engine for UBP Framework.
    Simulates resonance behaviors and predicts energy/coherence based on
    atomic structures using physics-inspired calculations.
    """
    VERSION = "1.1.2" # Updated version for re-run

    def __init__(self, realm_name: str = "electromagnetic"):
        self.realm_name = realm_name
        # Access constants directly from the global _config instance
        self.constants = _config.constants
        print(f"⚛️ HTREngine initialized (Version: {self.VERSION}) for realm: {self.realm_name}")

    def process_with_htr(self,
                         lattice_coords: np.ndarray,
                         realm: Optional[str] = None,
                         optimize: bool = False) -> Dict[str, Any]:
        """
        Processes a lattice structure through the HTR engine to simulate
        energy and Non-Random Coherence Index (NRCI).

        Args:
            lattice_coords: (num_atoms, 3) array of 3D atomic coordinates.
                            Assumed to be in meters.
            realm: The UBP realm context for this processing.
            optimize: If True, indicates that an optimization pass is conceptually applied
                      to the input or framework context, without hardcoding output nudges.

        Returns:
            Dict: Contains 'energy' (eV), 'nrci' (0-1), 'reconstruction_error',
                  'computation_time', and 'characteristic_length_scale_nm'.
        """
        print(f"DEBUG(HTR): Running process_with_htr (Version: {self.VERSION})")
        start_time = time.time()

        num_nodes = lattice_coords.shape[0]
        if num_nodes == 0:
            print("DEBUG(HTR): No nodes, returning default error.")
            return {
                'energy': 0.0,
                'nrci': 0.0,
                'reconstruction_error': 1.0,
                'computation_time': 0.0,
                'characteristic_length_scale_nm': 0.0,
                'error': 'No nodes in lattice_coords'
            }

        # 1. Calculate inter-atomic distances
        if num_nodes < 2:
            distances = np.array([0.0]) # Single atom or no atoms, no meaningful distances
            average_distance = 1.0e-10 # Fallback to a small positive value
            energy_deviation_factor = 0.0
            print("DEBUG(HTR): Less than 2 nodes, distances set to 0.0, using fallback average_distance.")
        else:
            # Calculate all pairwise distances and clip to avoid near-zero values
            # and potential division by zero later. Using triu_indices to avoid duplicates.
            distances_full_matrix = np.linalg.norm(lattice_coords[:, None, :] - lattice_coords[None, :, :], axis=-1)
            distances = distances_full_matrix[np.triu_indices(num_nodes, k=1)]
            distances = np.clip(distances, 1e-15, None) # Ensure positive and non-zero
            
            # Use average distance as a primary structural feature
            average_distance = float(np.mean(distances))
            # Ensure average_distance is finite for calculations
            if not np.isfinite(average_distance) or average_distance <= 0:
                average_distance = 1.0e-10 # Fallback to a small positive value
            
            # Calculate deviation factor: higher for more disordered/strained lattices
            energy_deviation_factor = float(np.std(distances) / average_distance) if average_distance > 0 else 0.0
            # Ensure energy_deviation_factor is finite
            if not np.isfinite(energy_deviation_factor):
                energy_deviation_factor = 0.0
        
        print(f"DEBUG(HTR): Calculated average_distance: {average_distance:.2e} m, type: {type(average_distance)}") # Debug print
        print(f"DEBUG(HTR): Energy Deviation Factor: {energy_deviation_factor:.4e}, type: {type(energy_deviation_factor)}") # Debug print (changed format to 4e)

        # 2. Simulate energy (more sensitive heuristic based on disorder)
        base_energy_per_node_eV = 0.5 # Reduced base energy contribution per atom
        
        # Adjusting impact of disorder on energy. Exponential to be more sensitive to small disorder.
        simulated_energy_eV = num_nodes * base_energy_per_node_eV * (1.0 + energy_deviation_factor * 20.0 + (energy_deviation_factor * 10.0)**2) # Increased factor and quadratic term
        
        # Clamping energy to a more realistic range for material stability (e.g., 0.5 eV to 500 eV per atom/node)
        simulated_energy_eV = max(0.5 * num_nodes, min(500.0 * num_nodes, simulated_energy_eV)) # Adjusted min and max based on number of nodes
        # Ensure simulated_energy_eV is finite
        if not np.isfinite(simulated_energy_eV):
            simulated_energy_eV = 0.5 * num_nodes # Fallback
        print(f"DEBUG(HTR): Simulated Energy (before NRCI opt): {simulated_energy_eV:.4f} eV, type: {type(simulated_energy_eV)}") # Debug print
        
        # 3. Calculate Non-Random Coherence Index (NRCI)
        nrci_base = 0.85 # Higher base for better starting coherence
        
        # Stronger, more sensitive penalty for disorder. Using a logistic-like decay.
        nrci_disorder_penalty = 0.8 * (1.0 - (1.0 / (1.0 + np.exp(-15.0 * (energy_deviation_factor - 0.05))))) # S-curve penalty, more sensitive around 0.05 deviation
        
        # Bonus NRCI for more atoms (more potential for coherence, up to a point)
        nrci_size_bonus = min(0.15, float(np.log1p(num_nodes)) * 0.02) # Logarithmic bonus, caps at 0.15, faster growth
        
        simulated_nrci_raw = nrci_base - nrci_disorder_penalty + nrci_size_bonus # Store raw for debug
        simulated_nrci = max(0.05, min(0.98, simulated_nrci_raw)) # Adjusted min and max
        # Ensure simulated_nrci is finite
        if not np.isfinite(simulated_nrci):
            simulated_nrci = 0.05 # Fallback
        print(f"DEBUG(HTR): Raw NRCI: {simulated_nrci_raw:.7f}, Capped NRCI: {simulated_nrci:.7f}, type: {type(simulated_nrci)}") # Debug print
        
        # 4. Characteristic Length Scale (formerly CRV_used)
        # Convert average_distance (in meters) to nanometers
        characteristic_length_scale_nm = float(average_distance * 1e9) # Convert to nanometers
        # Clamp to a realistic atomic-scale length range (e.g., 0.1 nm to 10 nm)
        characteristic_length_scale_nm = max(0.1, min(10.0, characteristic_length_scale_nm))
        # Ensure characteristic_length_scale_nm is finite
        if not np.isfinite(characteristic_length_scale_nm):
            characteristic_length_scale_nm = 0.1 # Fallback
        print(f"DEBUG(HTR): Characteristic Length Scale (before NRCI opt): {characteristic_length_scale_nm:.2e} nm, type: {type(characteristic_length_scale_nm)}") # Debug print

        # REMOVED: The 'optimize' flag's conceptual 'nudge' from HTREngine
        # The 'optimize' flag will now conceptually indicate that an optimization pass
        # has been applied to the *input* or *context* by the higher-level framework,
        # but the HTR engine itself will not perform a hardcoded post-processing nudge.
        # This keeps the HTR engine's output purely a reflection of the lattice input,
        # aligning better with "scientific grade" simulation.


        # Reconstruction error (conceptual: lower for higher NRCI)
        reconstruction_error = 1.0 - simulated_nrci
        # Ensure reconstruction_error is finite
        if not np.isfinite(reconstruction_error):
            reconstruction_error = 1.0 # Fallback
        print(f"DEBUG(HTR): Reconstruction Error: {reconstruction_error:.4f}, type: {type(reconstruction_error)}") # Debug print

        computation_time = time.time() - start_time
        # Ensure computation_time is finite
        if not np.isfinite(computation_time):
            computation_time = 0.0 # Fallback
        print(f"DEBUG(HTR): Computation Time: {computation_time:.4f} s, type: {type(computation_time)}") # Debug print

        # Final check for types before returning (already converted to float in assignment for safety)
        final_energy = float(simulated_energy_eV)
        final_nrci = float(simulated_nrci)
        final_reconstruction_error = float(reconstruction_error)
        final_computation_time = float(computation_time)
        final_characteristic_length_scale_nm = float(characteristic_length_scale_nm)

        print(f"DEBUG(HTR): Final Return Values: Energy={final_energy}, NRCI={final_nrci}, RecError={final_reconstruction_error}, CompTime={final_computation_time}, CharLength={final_characteristic_length_scale_nm}")

        return {
            'energy': final_energy,
            'nrci': final_nrci,
            'reconstruction_error': final_reconstruction_error,
            'computation_time': final_computation_time,
            'characteristic_length_scale_nm': final_characteristic_length_scale_nm
        }
"""
Universal Binary Principle (UBP) Framework v3.2+ - Basic UBP Pattern Generation Module
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

"""
# remove `ubp_constants_loader.py`
import numpy as np
import os
from typing import Dict, List, Tuple, Any
import random # Import for random sampling

from ubp_config import get_config, UBPConfig # Ensure UBPConfig is available
from htr_engine import HTREngine # Import the HTR Engine

def run_ubp_simulation(frequencies: List[float], realm_names: List[str], output_dir: str, config: UBPConfig, resolution: int = 256) -> List[Dict[str, Any]]:
    """
    Generates basic cymatic patterns for given frequencies using a simple sine/cosine model.
    This function is intended to be called by UBPPatternIntegrator.
    
    Now integrates with HTREngine for more realistic energy/NRCI calculations.
    """
    print(f"Generating {len(frequencies)} basic patterns at resolution {resolution} with HTREngine integration...")
    sim_results = []

    # Retrieve PI from the config constants
    math_pi = config.constants.PI

    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    if len(frequencies) != len(realm_names):
        raise ValueError("Lengths of 'frequencies' and 'realm_names' must match for HTREngine integration.")

    for i, freq in enumerate(frequencies):
        realm_name = realm_names[i]
        
        # Dynamic parameter scaling: Adjust the spatial range based on frequency magnitude
        normalized_freq = max(1.0, freq / 1e9) # Example normalization to GHz scale for dynamic range
        
        spatial_range_factor = (4 * math_pi) / np.sqrt(normalized_freq)
        
        min_spatial_range = math_pi / 2 
        max_spatial_range = 8 * math_pi 
        spatial_range = np.clip(spatial_range_factor, min_spatial_range, max_spatial_range)


        x = np.linspace(-spatial_range, spatial_range, resolution)
        y = np.linspace(-spatial_range, spatial_range, resolution)
        X, Y = np.meshgrid(x, y)

        # Generate a pattern using the frequency
        amplitude_mod = (np.sin(X/spatial_range * math_pi) + np.cos(Y/spatial_range * math_pi) + 2) / 4 
        pattern = amplitude_mod * np.sin(freq * X) * np.cos(freq * Y)

        # Normalize pattern to 0-1 range
        if pattern.max() - pattern.min() > 1e-9:
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        else:
            pattern = np.zeros_like(pattern) # Handle flat patterns

        # --- Integrate with HTREngine for "real" energy/NRCI metrics ---
        htr_engine_instance = HTREngine(realm_name=realm_name)

        # Convert 2D pattern data to 3D lattice_coords for HTREngine
        # We need to create a representative 3D structure from the 2D pattern.
        # Let's consider the pattern value as a 'height' or 'z-coordinate'.
        # To manage computational cost for HTREngine, we will sample points.
        
        # Flatten the meshgrid and pattern arrays
        x_flat = X.flatten()
        y_flat = Y.flatten()
        pattern_flat = pattern.flatten()

        # Create candidate 3D points
        points_3d_candidates = np.column_stack((x_flat, y_flat, pattern_flat))
        
        # Filter points based on pattern intensity (e.g., above average or threshold)
        # Or, just ensure a good distribution across intensities.
        # For simplicity, let's sample randomly to avoid bias towards high-intensity regions only.
        
        target_sample_size = 2000 # Keep this reasonable for HTREngine's pairwise distance calculations
        if points_3d_candidates.shape[0] > target_sample_size:
            # Randomly sample 'target_sample_size' points
            sample_indices = random.sample(range(points_3d_candidates.shape[0]), target_sample_size)
            sampled_lattice_coords = points_3d_candidates[sample_indices]
        else:
            sampled_lattice_coords = points_3d_candidates
        
        # Ensure sampled_lattice_coords are within a reasonable scale for HTREngine
        # HTREngine assumes meters, so scale these spatial_range values (which are ~radians)
        # to a physical scale (e.g., nanometers or picometers)
        # Using a fixed conversion factor for illustration, you might want a more sophisticated one.
        physical_scale_factor = 1e-9 # Convert from abstract units to meters (e.g., if spatial_range is in 'abstract units', map 1 unit to 1 nm for HTREngine)
        scaled_lattice_coords = sampled_lattice_coords * physical_scale_factor

        htr_results = htr_engine_instance.process_with_htr(
            lattice_coords=scaled_lattice_coords, 
            realm=realm_name, 
            optimize=False # For basic generation, don't optimize here.
        )
        
        # Add HTREngine metrics to sim_results
        sim_results.append({
            'pattern_data': pattern,
            'frequency': freq,
            'realm_context': realm_name, # Added realm context
            'final_emergent_energy': htr_results['energy'], # From HTREngine
            'nrci_from_htr': htr_results['nrci'], # From HTREngine
            'characteristic_length_scale_nm': htr_results['characteristic_length_scale_nm'], # From HTREngine
            'bitfield_dimensions': config.get_bitfield_dimensions(), # Still from config
            'resolution': resolution
        })
        print(f"  Generated pattern for frequency {freq:.2e} Hz ({realm_name} realm, spatial range: +/-{spatial_range:.2f}).")
        print(f"  HTREngine metrics: Energy={htr_results['energy']:.4e} eV, NRCI={htr_results['nrci']:.4f}, Char. Length={htr_results['characteristic_length_scale_nm']:.2f} nm")

    return sim_results

if __name__ == "__main__":
    # Example standalone test for this module
    config = get_config(environment="development")
    
    # Use actual CRVs and their corresponding realms from config for testing
    test_frequencies = []
    test_realm_names = []

    # Get a few realms for testing purposes
    example_realms = ['electromagnetic', 'quantum', 'optical', 'nuclear', 'biologic']
    for r_name in example_realms:
        realm_cfg = config.get_realm_config(r_name)
        if realm_cfg:
            test_frequencies.append(realm_cfg.main_crv)
            test_realm_names.append(r_name)
        else:
            print(f"Warning: Realm '{r_name}' not found in config, skipping.")

    output_temp_dir = "/output/temp_basic_patterns/"
    os.makedirs(output_temp_dir, exist_ok=True)
    
    print("\n--- Running standalone test for ubp_pattern_generator_1.py ---")
    results = run_ubp_simulation(test_frequencies, test_realm_names, output_temp_dir, config)
    print(f"\nBasic simulation generated {len(results)} patterns.")
    for i, res in enumerate(results):
        print(f"  Pattern {i+1}: Freq {res['frequency']:.2e} Hz, Realm {res['realm_context']}, Energy {res['final_emergent_energy']:.2e} eV, NRCI {res['nrci_from_htr']:.3f}, Char. Length {res['characteristic_length_scale_nm']:.2f} nm, Resolution {res['resolution']}")
    print("--- Standalone test finished ---")
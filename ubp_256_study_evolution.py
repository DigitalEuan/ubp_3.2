"""
Universal Binary Principle (UBP) Framework v3.2+ - 256 Study Evolution
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

This module centralizes all configurable parameters for the UBP framework,
including system constants, performance thresholds, realm definitions,
observer parameters, and Bitfield dimensions. It ensures a single source
of truth for all framework settings.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import json
import time
from typing import Dict, List, Tuple, Any, Optional
import math
from scipy import signal
from scipy.fft import fft2, ifft2, fftshift
import warnings
import os # Import os for path manipulation

warnings.filterwarnings("ignore")

# Import actual UBPConfig for constants and CRVs
from ubp_config import get_config, UBPConfig, RealmConfig

class UBP256Evolution:
    """
    Advanced 256x256 resolution study building on v3.2+ framework
    Incorporates CRVs (Core Resonance Values) and sub-harmonic removal
    """
    
    def __init__(self, resolution: int = 256, config: Optional[UBPConfig] = None):
        self.resolution = resolution
        self.config = config if config else get_config() # Ensure config is initialized

        # Dynamically load CRVs from UBPConfig
        self.crv_constants = self._load_crv_constants_from_config()
        
        self.subharmonic_filters = {
            'fundamental': [1.0, 1/2.0, 1/3.0, 1/4.0, 1/5.0, 1/6.0, 1/7.0, 1/8.0], # Ensure floats
            'golden': [1.0, 1/self.config.constants.PHI, # Use config constants directly
                       1/((self.config.constants.PHI)**2),
                       1/((self.config.constants.PHI)**3),
                       1/((self.config.constants.PHI)**4)],
            'musical': [1.0, 1/2.0, 1/3.0, 1/4.0, 1/5.0, 1/6.0, 1/7.0, 1/8.0, 1/9.0, 1/10.0, 1/11.0, 1/12.0],
            'quantum': [1.0, 1/np.sqrt(2), 1/np.sqrt(3), 1/np.sqrt(5), 1/np.sqrt(7), 1/np.sqrt(11)]
        }
        
        self.coherence_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'perfect': 0.95
        }

    def _load_crv_constants_from_config(self) -> Dict[str, float]:
        """Loads CRV constants from the UBPConfig realms."""
        crvs = {}
        # Base CRV can be a reference from a specific realm, e.g., electromagnetic
        em_crv = self.config.get_realm_config('electromagnetic')
        crvs['CRV_BASE'] = em_crv.main_crv if em_crv else 2.45e9 # Fallback
        
        # Other derived CRVs using UBPConfig's mathematical constants, now from config.constants
        math_phi = self.config.constants.PHI
        math_pi = self.config.constants.PI
        math_e = self.config.constants.E

        crvs['CRV_PHI'] = crvs['CRV_BASE'] * math_phi
        crvs['CRV_PI'] = crvs['CRV_BASE'] * math_pi / 2
        crvs['CRV_E'] = crvs['CRV_BASE'] * math_e / 2
        
        # Use quantum realm's CRV if available for a 'Zeta' like reference
        quantum_crv = self.config.get_realm_config('quantum')
        crvs['CRV_ZETA'] = quantum_crv.main_crv * 0.5 if quantum_crv else crvs['CRV_BASE'] * 0.5 # Fallback
        
        # CSC related CRV
        crvs['CRV_CSC'] = crvs['CRV_BASE'] / math_pi
        
        return crvs
        
    def generate_crv_pattern(self, crv_key: str, harmonics: List[float] = None) -> np.ndarray:
        """
        Generate pattern based on Core Resonance Value.
        Refinement: Introduce dynamic scaling factor and linspace range based on crv_freq.
        """
        if harmonics is None:
            harmonics = self.subharmonic_filters['fundamental']
            
        crv_freq = self.crv_constants.get(crv_key, self.crv_constants['CRV_BASE']) # Fallback
        
        # Retrieve PI from the config constants
        math_pi = self.config.constants.PI
        
        # Dynamic parameter scaling: Adjust the spatial range based on frequency magnitude
        # Higher frequencies can be mapped to a smaller spatial range to make patterns visible
        # Use a logarithmic scale for robustness across large frequency differences
        normalized_freq_for_scaling = max(1.0, crv_freq / 1e9) # Example normalization to GHz scale for dynamic range
        
        # Adjust spatial_range dynamically: smaller range for higher frequencies
        # The 4 * math_pi is a base scale, divided by sqrt(normalized_freq_for_scaling) to make it inverse.
        spatial_range_factor = (4 * math_pi) / np.sqrt(normalized_freq_for_scaling)
        
        # Clamp the spatial_range_factor to reasonable bounds to avoid extreme scaling
        min_spatial_range = math_pi / 2 # Minimum extent, e.g., half pi
        max_spatial_range = 8 * math_pi # Maximum extent
        spatial_range = np.clip(spatial_range_factor, min_spatial_range, max_spatial_range)

        # Create 256x256 coordinate system with dynamic range
        x = np.linspace(-spatial_range, spatial_range, self.resolution)
        y = np.linspace(-spatial_range, spatial_range, self.resolution)
        X, Y = np.meshgrid(x, y) 
        
        # Generate pattern with harmonic series
        pattern = np.zeros_like(X)
        for i, harmonic in enumerate(harmonics):
            freq = crv_freq * harmonic
            amplitude = 1.0 / (i + 1)  # Decreasing amplitude
            # Add an amplitude modulation based on the spatial grid, for richer patterns
            amplitude_mod = (np.sin(X/spatial_range * math_pi) + np.cos(Y/spatial_range * math_pi) + 2) / 4 
            pattern += amplitude * amplitude_mod * np.sin(freq * X) * np.cos(freq * Y)
        
        # Normalize pattern to 0-1 range after generation
        if pattern.max() - pattern.min() > 1e-9:
            pattern = (pattern - pattern.min()) / (pattern.max() - pattern.min())
        else:
            pattern = np.zeros_like(pattern) # Handle flat patterns
            
        return pattern
    
    def apply_subharmonic_removal(self, pattern: np.ndarray, removal_type: str = 'adaptive') -> np.ndarray:
        """Remove sub-harmonics to isolate fundamental patterns"""
        
        # Convert to frequency domain
        fft_pattern = fft2(pattern)
        magnitude = np.abs(fftshift(fft_pattern))
        
        # Create removal mask based on type
        if removal_type == 'adaptive':
            mask = self._create_adaptive_mask(magnitude)
        elif removal_type == 'fundamental':
            mask = self._create_fundamental_mask(magnitude)
        elif removal_type == 'golden':
            mask = self._create_golden_mask(magnitude)
        else:
            mask = np.ones_like(magnitude)
        
        # Apply mask in frequency domain
        filtered_fft = fft_pattern * fftshift(mask)
        
        # Convert back to spatial domain
        filtered_pattern = np.real(ifft2(filtered_fft))
        
        # Normalize filtered pattern
        if filtered_pattern.max() - filtered_pattern.min() > 1e-9:
            filtered_pattern = (filtered_pattern - filtered_pattern.min()) / (filtered_pattern.max() - filtered_pattern.min())
        else:
            filtered_pattern = np.zeros_like(filtered_pattern)
            
        return filtered_pattern
    
    def _create_adaptive_mask(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Create adaptive mask based on coherence patterns.
        Refinement: More dynamic determination of fundamental_radius_unit from peaks.
        """
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)
        
        non_dc_mask = np.ones_like(magnitude, dtype=bool)
        non_dc_mask[center_y, center_x] = False # Exclude DC for radial profile calc

        for r_idx in range(max_radius):
            mask_radial = (dist_from_center >= r_idx) & (dist_from_center < r_idx + 1)
            radial_profile[r_idx] = np.sum(magnitude[mask_radial & non_dc_mask]) # Only sum non-DC magnitudes
        
        # Dynamically determine fundamental_radius_unit using peak detection in radial profile
        min_peak_distance = max(1, int(max_radius / 10))
        peaks, _ = signal.find_peaks(radial_profile, height=np.max(radial_profile) * 0.1, distance=min_peak_distance)
        
        fundamental_radius_unit = 0
        if len(peaks) > 0:
            fundamental_radius_unit = peaks[0]
        else:
            # Fallback if no clear peaks detected, use a sensible default
            fundamental_radius_unit = min(h, w) / 10 # Adjusted to 1/10th
        
        # Ensure fundamental_radius_unit is a positive integer
        fundamental_radius_unit = max(1, int(fundamental_radius_unit))

        mask = np.ones_like(magnitude)
        
        # Attenuate frequencies below the fundamental significantly
        mask[dist_from_center < fundamental_radius_unit - 5] = 0.05
        
        # Keep a band around the fundamental and its immediate harmonics
        # Scale harmonic multipliers
        harmonic_multipliers = [1, 2, 3, 4] # Keep a few integer harmonics
        band_width_pixels = 3 # Bandwidth around each harmonic peak
        
        for mult in harmonic_multipliers:
            current_radius_center = int(mult * fundamental_radius_unit)
            
            # Ensure radius is within bounds and positive
            if current_radius_center > 0 and current_radius_center < max_radius:
                inner_boundary = max(0, current_radius_center - band_width_pixels)
                outer_boundary = min(max_radius, current_radius_center + band_width_pixels)
                
                circle_mask = (dist_from_center >= inner_boundary) & (dist_from_center < outer_boundary)
                mask[circle_mask] = 1.0
            
        # Attenuate very high frequencies that are likely noise
        mask[dist_from_center > max_radius * 0.7] = 0.2
        return mask
    
    def _create_fundamental_mask(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Create mask for fundamental harmonic isolation.
        Refinement: Dynamically scale radii based on resolution and fundamental_radius_unit.
        """
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)
        
        non_dc_mask = np.ones_like(magnitude, dtype=bool)
        non_dc_mask[center_y, center_x] = False

        for r_idx in range(max_radius):
            mask_radial = (dist_from_center >= r_idx) & (dist_from_center < r_idx + 1)
            radial_profile[r_idx] = np.sum(magnitude[mask_radial & non_dc_mask])
        
        min_peak_distance = max(1, int(max_radius / 10))
        peaks, _ = signal.find_peaks(radial_profile, height=np.max(radial_profile) * 0.1, distance=min_peak_distance)
        
        fundamental_radius_unit = 0
        if len(peaks) > 0:
            fundamental_radius_unit = peaks[0]
        else:
            fundamental_radius_unit = min(h, w) / 10
        fundamental_radius_unit = max(1, int(fundamental_radius_unit))

        mask = np.zeros_like(magnitude)
        
        # Keep only fundamental and first few harmonics, scaled by fundamental_radius_unit
        harmonics = [1, 2] # Keep fewer harmonics for 'fundamental' filter
        band_width_pixels = 2
        
        for h_idx in harmonics:
            current_radius_center = int(h_idx * fundamental_radius_unit)
            if current_radius_center > 0 and current_radius_center < max_radius:
                inner_boundary = max(0, current_radius_center - band_width_pixels)
                outer_boundary = min(max_radius, current_radius_center + band_width_pixels)
                circle_mask = (dist_from_center >= inner_boundary) & (dist_from_center < outer_boundary)
                mask[circle_mask] = 1.0
            
        return mask
    
    def _create_golden_mask(self, magnitude: np.ndarray) -> np.ndarray:
        """
        Create mask based on golden ratio harmonics.
        Refinement: Dynamically scale radii based on resolution and fundamental_radius_unit.
        """
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)
        
        non_dc_mask = np.ones_like(magnitude, dtype=bool)
        non_dc_mask[center_y, center_x] = False

        for r_idx in range(max_radius):
            mask_radial = (dist_from_center >= r_idx) & (dist_from_center < r_idx + 1)
            radial_profile[r_idx] = np.sum(magnitude[mask_radial & non_dc_mask])
        
        min_peak_distance = max(1, int(max_radius / 10))
        peaks, _ = signal.find_peaks(radial_profile, height=np.max(radial_profile) * 0.1, distance=min_peak_distance)
        
        fundamental_radius_unit = 0
        if len(peaks) > 0:
            fundamental_radius_unit = peaks[0]
        else:
            fundamental_radius_unit = min(h, w) / 10
        fundamental_radius_unit = max(1, int(fundamental_radius_unit))

        mask = np.zeros_like(magnitude)
        phi = self.config.constants.PHI
        
        # Golden ratio harmonics, scaled by fundamental_radius_unit
        golden_harmonics = [1.0, phi, phi**2, phi**3] # Keep fewer harmonics for simpler patterns
        band_width_pixels = 3

        for h_idx in golden_harmonics:
            current_radius_center = int(h_idx * fundamental_radius_unit)
            if current_radius_center > 0 and current_radius_center < max_radius:
                inner_boundary = max(0, current_radius_center - band_width_pixels)
                outer_boundary = min(max_radius, current_radius_center + band_width_pixels)
                circle_mask = (dist_from_center >= inner_boundary) & (dist_from_center < outer_boundary)
                mask[circle_mask] = 1.0
            
        return mask
    
    def analyze_coherence_geometry(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Analyze geometric coherence in 256x256 patterns"""
        
        # 2D Fourier analysis
        fft_2d = fft2(pattern)
        magnitude = np.abs(fftshift(fft_2d))
        
        # Coherence metrics
        coherence_score = self._calculate_coherence_score(magnitude)
        symmetry_metrics = self._calculate_symmetry_metrics(pattern)
        geometric_features = self._extract_geometric_features(pattern)
        
        # Harmonic analysis
        harmonic_content = self._analyze_harmonic_content(magnitude)
        
        return {
            'coherence_score': float(coherence_score), # Explicitly cast to float
            'symmetry_metrics': symmetry_metrics,
            'geometric_features': geometric_features,
            'harmonic_content': harmonic_content,
            'pattern_classification': self._classify_pattern(coherence_score, geometric_features)
        }
    
    def _calculate_coherence_score(self, magnitude: np.ndarray) -> float:
        """
        Calculate coherence score for 256x256 patterns.
        Refinement: Use dynamically determined fundamental_radius_unit.
        """
        total_energy = np.sum(magnitude**2)
        if total_energy < 1e-15:
            return 0.0
        
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)
        
        non_dc_mask = np.ones_like(magnitude, dtype=bool)
        non_dc_mask[center_y, center_x] = False

        for r_idx in range(max_radius):
            mask_radial = (dist_from_center >= r_idx) & (dist_from_center < r_idx + 1)
            radial_profile[r_idx] = np.sum(magnitude[mask_radial & non_dc_mask])
        
        min_peak_distance = max(1, int(max_radius / 10))
        peaks, _ = signal.find_peaks(radial_profile, height=np.max(radial_profile) * 0.1, distance=min_peak_distance)
        
        fundamental_radius_unit = 0
        if len(peaks) > 0:
            fundamental_radius_unit = peaks[0]
        else:
            fundamental_radius_unit = min(h, w) / 10
        fundamental_radius_unit = max(1, int(fundamental_radius_unit))

        combined_harmonic_mask = np.zeros_like(magnitude, dtype=bool)
        # Use a more comprehensive set of harmonics for the general coherence score
        harmonics_to_check = [1, 1/2.0, 1/3.0, 1/4.0, 1.5, 2.0, 3.0, 4.0] # Integer and sub-harmonics
        band_width = 3 # pixels

        for ratio in harmonics_to_check:
            radius = int(ratio * fundamental_radius_unit)
            inner_radius = max(0, radius - band_width)
            outer_radius = min(max_radius, radius + band_width) 
            
            if inner_radius < outer_radius:
                annulus_mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
                combined_harmonic_mask = combined_harmonic_mask | annulus_mask
        
        harmonic_energy = np.sum(magnitude[combined_harmonic_mask & non_dc_mask]**2)
        
        return harmonic_energy / total_energy if total_energy > 0 else 0
    
    def _calculate_symmetry_metrics(self, pattern: np.ndarray) -> Dict[str, float]:
        """Calculate multiple symmetry metrics"""
        h, w = pattern.shape
        
        # Horizontal symmetry
        horizontal_sym = np.corrcoef(pattern[:h//2, :].flatten(), 
                                     np.flip(pattern[h//2:, :], 0).flatten())[0, 1]
        
        # Vertical symmetry
        vertical_sym = np.corrcoef(pattern[:, :w//2].flatten(), 
                                   np.flip(pattern[:, w//2:], 1).flatten())[0, 1]
        
        # Diagonal symmetry
        # Pad to square if needed for diagonal flip, otherwise flatten works
        if h != w:
            min_dim = min(h, w)
            pattern_cropped = pattern[:min_dim, :min_dim]
        else:
            pattern_cropped = pattern

        diagonal_sym = np.corrcoef(pattern_cropped.flatten(), 
                                   np.flip(pattern_cropped.T, (0, 1)).flatten())[0, 1]
        
        # Rotational symmetry (90 degrees)
        rotated_90 = np.rot90(pattern, 1)
        rotational_90 = np.corrcoef(pattern.flatten(), rotated_90.flatten())[0, 1]
        
        # Rotational symmetry (180 degrees)
        rotated_180 = np.rot90(pattern, 2)
        rotational_180 = np.corrcoef(pattern.flatten(), rotated_180.flatten())[0, 1]
        
        return {
            'horizontal': float(horizontal_sym) if not np.isnan(horizontal_sym) else 0.0,
            'vertical': float(vertical_sym) if not np.isnan(vertical_sym) else 0.0,
            'diagonal': float(diagonal_sym) if not np.isnan(diagonal_sym) else 0.0,
            'rotational_90': float(rotational_90) if not np.isnan(rotational_90) else 0.0,
            'rotational_180': float(rotational_180) if not np.isnan(rotational_180) else 0.0
        }
    
    def _extract_geometric_features(self, pattern: np.ndarray) -> Dict[str, Any]:
        """Extract geometric features from patterns"""
        from scipy.ndimage import label, find_objects, binary_erosion
        
        # Threshold for shape detection (dynamic based on pattern mean and std dev)
        threshold = np.mean(pattern) + np.std(pattern) * 0.1 # More robust threshold
        binary_pattern = (pattern > threshold).astype(int)
        
        # Remove small noisy components before labeling
        eroded_pattern = binary_erosion(binary_pattern, structure=np.ones((2,2)))
        
        # Label connected components
        labeled, num_features = label(eroded_pattern)
        
        # Calculate properties
        objects = find_objects(labeled)
        
        features = {
            'num_shapes': num_features,
            'shape_areas': [],
            'shape_centroids': [],
            'shape_eccentricities': [], # Placeholder for actual eccentricity calculation
            'total_area': np.sum(binary_pattern),
            'perimeter_estimate': self._estimate_perimeter(eroded_pattern)
        }
        
        for obj in objects:
            if obj is not None:
                # Calculate area
                area = np.sum(labeled[obj] > 0)
                features['shape_areas'].append(int(area))
                
                # Calculate centroid
                y_slice, x_slice = obj
                y_center = (y_slice.start + y_slice.stop) // 2
                x_center = (x_slice.start + x_slice.stop) // 2
                features['shape_centroids'].append([int(x_center), int(y_center)])
        
        return features
    
    def _estimate_perimeter(self, binary_pattern: np.ndarray) -> int:
        """Estimate perimeter using edge detection"""
        from scipy.ndimage import convolve
        
        # Simple edge detection kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        
        edges = convolve(binary_pattern, kernel)
        perimeter = np.sum(np.abs(edges) > 0)
        
        return int(perimeter)
    
    def _analyze_harmonic_content(self, magnitude: np.ndarray) -> Dict[str, Any]:
        """Analyze harmonic content in frequency domain"""
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        
        # Radial analysis
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        max_radius = min(h, w) // 2
        radial_profile = np.zeros(max_radius)

        non_dc_mask = np.ones_like(magnitude, dtype=bool)
        non_dc_mask[center_y, center_x] = False

        for r_idx in range(max_radius):
            mask_radial = (dist_from_center >= r_idx) & (dist_from_center < r_idx + 1)
            radial_profile[r_idx] = np.sum(magnitude[mask_radial & non_dc_mask])
        
        # Find harmonic peaks
        min_peak_distance = max(1, int(max_radius / 10))
        peaks, properties = signal.find_peaks(radial_profile, 
                                            height=np.max(radial_profile) * 0.05,
                                            distance=min_peak_distance)
        
        # Calculate harmonic ratios
        ratios = []
        if len(peaks) > 1 and peaks[0] > 0: # Ensure fundamental peak is non-zero
            for i in range(1, len(peaks)):
                ratio = peaks[i] / peaks[0]
                ratios.append(float(ratio))
        
        return {
            'peaks': peaks.tolist(),
            'peak_heights': properties['peak_heights'].tolist() if 'peak_heights' in properties else [],
            'harmonic_ratios': ratios,
            'fundamental_frequency_radius': int(peaks[0]) if len(peaks) > 0 else 0
        }
    
    def _classify_pattern(self, coherence_score: float, geometric_features: Dict[str, Any]) -> str:
        """Classify pattern based on coherence and geometry"""
        if coherence_score > self.config.performance.COHERENCE_THRESHOLD * 0.9: # Using a scaled config threshold
            if geometric_features['num_shapes'] == 1:
                return "Perfect Coherence - Single Dominant Form"
            elif geometric_features['num_shapes'] <= 3:
                return "High Coherence - Crystalline Structure"
            else:
                return "High Coherence - Complex Harmony"
        elif coherence_score > self.config.performance.COHERENCE_THRESHOLD * 0.6:
            if geometric_features['num_shapes'] <= 5:
                return "Medium Coherence - Ordered Complexity"
            else:
                return "Medium Coherence - Chaonic Structure"
        elif coherence_score > self.config.performance.COHERENCE_THRESHOLD * 0.3:
            return "Low Coherence - Transitional Pattern"
        else:
            return "Minimal Coherence - Random Distribution"
    
    def _create_circular_mask(self, h: int, w: int, center_y: int, center_x: int, radius: int) -> np.ndarray:
        """Create circular mask for analysis"""
        y, x = np.ogrid[:h, :w]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        return mask
    
    def run_comprehensive_study(self, output_dir: str = "/output/ubp_256_study") -> Dict[str, Any]:
        """Run the complete 256x256 resolution study"""
        print("🚀 Starting UBP 256x256 Evolution Study (Integrated with UBPConfig)...")
        print("=" * 60)
        
        results = {
            'study_metadata': {
                'resolution': self.resolution,
                'crv_constants_used': self.crv_constants,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'framework_version': '3.2+ Evolution (Integrated)'
            },
            'patterns': {},
            'analysis': {},
            'insights': []
        }
        
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        # Test all CRV patterns
        crv_keys = list(self.crv_constants.keys())
        
        for crv_key in crv_keys:
            # print(f"\n📊 Analyzing CRV: {crv_key}")
            # print("-" * 40)
            
            # Generate base pattern
            base_pattern = self.generate_crv_pattern(crv_key)
            
            # Apply sub-harmonic removal
            filtered_patterns_data = {}
            removal_types = ['adaptive', 'fundamental', 'golden']
            
            for removal_type in removal_types:
                filtered_pattern = self.apply_subharmonic_removal(base_pattern, removal_type)
                filtered_patterns_data[removal_type] = filtered_pattern
                
                # Analyze coherence geometry
                analysis = self.analyze_coherence_geometry(filtered_pattern)
                
                key = f"{crv_key}_{removal_type}"
                results['patterns'][key] = {
                    'base_crv': crv_key,
                    'removal_type': removal_type,
                    'pattern_data_array': filtered_pattern.tolist(), # Store as list for JSON serialization
                    'analysis': analysis
                }
                
                # print(f"  {removal_type}: coherence={analysis['coherence_score']:.4f}, "
                #       f"shapes={analysis['geometric_features']['num_shapes']}, "
                #       f"classification={analysis['pattern_classification']}")
        
        # Cross-CRV analysis
        # print("\n🔍 Cross-CRV Analysis...")
        results['cross_analysis'] = self._perform_cross_crv_analysis(results['patterns'])
        
        # Generate insights
        # print("\n💡 Generating Insights...")
        results['insights'] = self._generate_insights(results)
        
        # Save comprehensive results
        results_file_path = os.path.join(output_dir, "ubp_256_study_results.json")
        with open(results_file_path, 'w') as f:
            json.dump(results, f, indent=2, default=str) # Use default=str for any non-serializable types
        
        print(f"\n✅ Study Complete! Results saved to {results_file_path}")
        return results
    
    def _perform_cross_crv_analysis(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Perform cross-CRV comparative analysis"""
        analysis = {
            'coherence_rankings': [],
            'symmetry_leaders': [],
            'geometric_diversity': [],
            'harmonic_relationships': []
        }
        
        # Coherence rankings
        for key, data in patterns.items():
            coherence = data['analysis']['coherence_score']
            analysis['coherence_rankings'].append({
                'key': key,
                'coherence': coherence,
                'crv': data['base_crv'],
                'removal': data['removal_type']
            })
        
        # Sort by coherence
        analysis['coherence_rankings'].sort(key=lambda x: x['coherence'], reverse=True)
        
        # Symmetry analysis
        for key, data in patterns.items():
            symmetries = data['analysis']['symmetry_metrics']
            # Ensure values are float and handle potential for empty list
            float_symmetries = [float(val) for val in symmetries.values() if isinstance(val, (int, float))]
            avg_symmetry = np.mean(float_symmetries) if float_symmetries else 0.0
            analysis['symmetry_leaders'].append({
                'key': key,
                'avg_symmetry': avg_symmetry,
                'symmetries': symmetries
            })
        
        analysis['symmetry_leaders'].sort(key=lambda x: x['avg_symmetry'], reverse=True)
        
        return analysis
    
    def _generate_insights(self, results: Dict[str, Any]) -> List[str]:
        """Generate insights from the comprehensive study"""
        insights = []
        
        # Top coherence patterns
        if results['cross_analysis']['coherence_rankings']:
            top_coherence = results['cross_analysis']['coherence_rankings'][:3]
            insights.append(f"Top 3 coherence patterns: {[c['key'] for c in top_coherence]}")
        
        # Symmetry insights
        if results['cross_analysis']['symmetry_leaders']:
            top_symmetry = results['cross_analysis']['symmetry_leaders'][:3]
            insights.append(f"Most symmetric patterns: {[s['key'] for s in top_symmetry]}")
        
        # CRV performance
        crv_performance = {}
        for key, data in results['patterns'].items():
            crv = data['base_crv']
            coherence = data['analysis']['coherence_score']
            if crv not in crv_performance:
                crv_performance[crv] = []
            crv_performance[crv].append(coherence)
        
        if crv_performance:
            # Average coherence per CRV
            avg_crv_coherence = {crv: np.mean(scores) for crv, scores in crv_performance.items()}
            best_crv = max(avg_crv_coherence, key=avg_crv_coherence.get)
            insights.append(f"Best performing CRV: {best_crv} (avg coherence: {avg_crv_coherence[best_crv]:.4f})")
        
        # Removal method effectiveness
        removal_effectiveness = {}
        for key, data in results['patterns'].items():
            removal = data['removal_type']
            coherence = data['analysis']['coherence_score']
            if removal not in removal_effectiveness:
                removal_effectiveness[removal] = []
            removal_effectiveness[removal].append(coherence)
        
        if removal_effectiveness:
            avg_removal_coherence = {r: np.mean(scores) for r, scores in removal_effectiveness.items()}
            best_removal = max(avg_removal_coherence, key=avg_removal_coherence.get)
            insights.append(f"Most effective sub-harmonic removal: {best_removal} (avg coherence: {avg_removal_coherence[best_removal]:.4f})")
        
        # Geometric diversity
        shape_counts = [data['analysis']['geometric_features']['num_shapes'] 
                       for data in results['patterns'].values()]
        if shape_counts:
            insights.append(f"Geometric diversity range: {min(shape_counts)}-{max(shape_counts)} shapes per pattern")
        
        return insights
    
    def visualize_results(self, results: Dict[str, Any], output_dir: str = "/output/ubp_256_study", num_samples: int = 6):
        """Create visualizations for the study results"""
        
        os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

        fig, axes = plt.subplots(num_samples // 2 if num_samples // 2 > 0 else 1, 4, figsize=(20, (num_samples // 2) * 5))
        axes = axes.flatten()
        
        # Select top patterns for visualization
        top_patterns = results['cross_analysis']['coherence_rankings'][:num_samples]
        
        for i, pattern_info in enumerate(top_patterns):
            key = pattern_info['key']
            data = results['patterns'][key]
            
            # Ensure pattern_data_array is loaded as numpy array
            pattern = np.array(data['pattern_data_array'])
            analysis = data['analysis']
            
            # Plot pattern
            ax_idx_pattern = i*2
            if ax_idx_pattern < len(axes):
                im = axes[ax_idx_pattern].imshow(pattern, cmap='viridis', aspect='equal')
                axes[ax_idx_pattern].set_title(f"{key}\nCoherence: {analysis['coherence_score']:.3f}")
                plt.colorbar(im, ax=axes[ax_idx_pattern])
            
            # Plot frequency domain
            fft_2d = fft2(pattern)
            magnitude = np.abs(fftshift(fft_2d))
            
            # Add a small value before log to avoid log(0) for very sparse FFTs
            magnitude = np.log1p(magnitude + 1e-9)  # Log scale for visibility
            
            ax_idx_freq = i*2+1
            if ax_idx_freq < len(axes):
                im2 = axes[ax_idx_freq].imshow(magnitude, cmap='hot', aspect='equal')
                axes[ax_idx_freq].set_title("Frequency Domain")
                plt.colorbar(im2, ax=axes[ax_idx_freq])
        
        plt.tight_layout()
        visualization_file_path = os.path.join(output_dir, "ubp_256_study_visualization.png")
        plt.savefig(visualization_file_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Visualization saved to {visualization_file_path}")

# def main():
#     """Main execution function"""
#     print("🌟 UBP 256x256 Evolution Study - Advanced Framework")
#     print("=" * 60)
#     print("Building on v3.2+ with CRVs and sub-harmonic removal")
    
#     ubp_config_instance = get_config(environment="development") # Ensure config is initialized
#     study = UBP256Evolution(config=ubp_config_instance)
#     results = study.run_comprehensive_study()
    
#     # Generate visualization
#     study.visualize_results(results)
    
#     # Create summary report
#     summary_report = {
#         'study_summary': {
#             'total_patterns_analyzed': len(results['patterns']),
#             'crvs_tested': len(study.crv_constants),
#             'removal_methods': 3,
#             'resolution': study.resolution,
#             'key_insights': results['insights'][:5]  # Top 5 insights
#         },
#         'top_performers': results['cross_analysis']['coherence_rankings'][:5],
#         'recommendations': [
#             "CRV_BASE shows exceptional coherence across all removal methods",
#             "Golden ratio CRV demonstrates unique harmonic relationships",
#             "Adaptive sub-harmonic removal provides best overall results",
#             "256x256 resolution reveals geometric structures invisible at lower resolutions",
#             "Framework successfully validates constants as cymatic patterns"
#         ]
#     }
    
#     summary_file_path = "/output/ubp_256_study_summary.json"
#     with open(summary_file_path, 'w') as f:
#         json.dump(summary_report, f, indent=2, default=str) # Use default=str for any non-serializable types
    
#     print("\n🎯 Study Summary:")
#     print(f"   Patterns analyzed: {len(results['patterns'])}")
#     print(f"   CRVs tested: {len(study.crv_constants)}")
#     print(f"   Resolution: {study.resolution}x{study.resolution}")
#     print(f"   Key insights: {len(results['insights'])}")
    
#     return results, summary_report

# if __name__ == "__main__":
#     results, summary = main()
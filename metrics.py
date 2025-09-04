"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Validation and Coherence Metrics
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements key metrics for validating UBP system performance:
- NRCI (Non-Random Coherence Index)
- Coherence Pressure calculations
- Fractal Dimension
- Spatial and temporal metrics
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Union


def nrci(simulated: List[float], target: List[float]) -> float:
    """
    Calculate Non-Random Coherence Index.
    
    Axiom: NRCI = 1 - (RMSE(S, T) / σ(T))
    Target: NRCI ≥ 0.999999 (six nines fidelity)
    
    Args:
        simulated: Simulated data (S)
        target: Target real-world data (T)
        
    Returns:
        NRCI value [0, 1]
    """
    if len(simulated) != len(target):
        raise ValueError(f"Data lengths must match: {len(simulated)} != {len(target)}")
    
    if len(target) == 0:
        return 0.0
    
    # Calculate RMSE
    squared_errors = [(s - t) ** 2 for s, t in zip(simulated, target)]
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))
    
    # Calculate standard deviation of target
    target_mean = sum(target) / len(target)
    target_variance = sum((t - target_mean) ** 2 for t in target) / len(target)
    target_std = math.sqrt(target_variance)
    
    if target_std == 0:
        return 1.0 if rmse == 0 else 0.0
    
    nrci_value = 1.0 - (rmse / target_std)
    return max(0.0, min(1.0, nrci_value))  # Clamp to [0, 1]


def nrci_error_correction_context(errors: List[float], total_toggles: int) -> float:
    """
    Calculate NRCI in error correction context.
    
    Formula: NRCI = 1 - (Σ error(M_ij)) / (9 × N_toggles)
    
    Args:
        errors: List of error values
        total_toggles: Total number of toggle operations
        
    Returns:
        NRCI value for error correction
    """
    if total_toggles == 0:
        return 0.0
    
    sum_errors = sum(errors)
    denominator = 9 * total_toggles  # 9 interactions per TGIC
    
    if denominator == 0:
        return 0.0
    
    nrci_value = 1.0 - (sum_errors / denominator)
    return max(0.0, min(1.0, nrci_value))


def coherence_pressure_spatial(distances: List[float], max_distances: List[float], 
                              active_bits: List[int]) -> float:
    """
    Calculate spatial coherence pressure.
    
    Axiom: Ψ_p = (1 - Σd_i/√Σd_max²) × (Σb_j/12)
    
    Args:
        distances: Distances from individual OffBits to cluster center
        max_distances: Maximum possible distances within Bitfield
        active_bits: Sum of active bits in Reality and Information layers (0-11)
        
    Returns:
        Spatial coherence pressure value
    """
    if not distances or not max_distances or not active_bits:
        return 0.0
    
    # Spatial term: (1 - Σd_i/√Σd_max²)
    sum_distances = sum(distances)
    sum_max_squared = sum(d_max ** 2 for d_max in max_distances)
    sqrt_sum_max_squared = math.sqrt(sum_max_squared)
    
    if sqrt_sum_max_squared == 0:
        spatial_term = 0.0
    else:
        spatial_term = 1.0 - (sum_distances / sqrt_sum_max_squared)
    
    # Bit alignment term: (Σb_j/12)
    sum_active_bits = sum(active_bits)
    bit_term = sum_active_bits / 12  # 12 bits in Reality + Information layers
    
    return spatial_term * bit_term


def coherence_pressure_temporal(I_toggle: float, tau_process: float) -> float:
    """
    Calculate temporal coherence pressure.
    
    Axiom: Ψ_p = I_toggle / τ_process
    
    Args:
        I_toggle: Informational flux from toggle operations (toggles/second)
        tau_process: Processing capacity of observer (seconds)
        
    Returns:
        Temporal coherence pressure value
    """
    if tau_process == 0:
        return float('inf') if I_toggle > 0 else 0.0
    
    return I_toggle / tau_process


def fractal_dimension(sub_clusters: int, scale_factor: float = 2.0) -> float:
    """
    Calculate fractal dimension of Glyph patterns.
    
    Axiom: D = log(m) / log(s)
    
    Args:
        sub_clusters: Number of sub-clusters (m)
        scale_factor: Scale factor between iterations (s, typically 2)
        
    Returns:
        Fractal dimension value
    """
    if sub_clusters <= 0 or scale_factor <= 1:
        return 0.0
    
    return math.log(sub_clusters) / math.log(scale_factor)


def fractal_dimension_enhanced(pattern_length: int, sub_glyphs: List[int]) -> float:
    """
    Calculate enhanced fractal dimension for complex patterns.
    
    Formula: D = log(m + 1) / log(2)
    where m = len(sub_glyphs) / (len(pattern) - len(sub_glyphs))
    
    Args:
        pattern_length: Length of the main pattern
        sub_glyphs: List of sub-glyph sizes
        
    Returns:
        Enhanced fractal dimension
    """
    if not sub_glyphs or pattern_length <= len(sub_glyphs):
        return 0.0
    
    m = len(sub_glyphs) / (pattern_length - len(sub_glyphs))
    return math.log(m + 1) / math.log(2)


def spatial_resonance_index(pattern_count: int, expected_count: int) -> float:
    """
    Calculate Spatial Resonance Index.
    
    Formula: SRI = 1 - |N_pattern - N_expected| / max(N_pattern, N_expected)
    
    Args:
        pattern_count: Actual number of patterns
        expected_count: Expected number of patterns
        
    Returns:
        Spatial Resonance Index [0, 1]
    """
    if pattern_count == 0 and expected_count == 0:
        return 1.0
    
    max_count = max(pattern_count, expected_count)
    if max_count == 0:
        return 0.0
    
    difference = abs(pattern_count - expected_count)
    sri = 1.0 - (difference / max_count)
    
    return max(0.0, min(1.0, sri))


def coherence_resonance_index(frequency: float, temporal_phase: float, 
                            phase_offset: float = 0.0, alpha: float = 1.0,
                            spatial_curvature: float = 0.0) -> float:
    """
    Calculate Coherence Resonance Index.
    
    Formula: CRI = cos(2πft + φ₀) × exp(-α|∇²ρ|)
    
    Args:
        frequency: Dominant resonance frequency
        temporal_phase: Temporal phase (t)
        phase_offset: Phase offset (φ₀)
        alpha: Scaling parameter
        spatial_curvature: Spatial curvature of OffBit density (|∇²ρ|)
        
    Returns:
        Coherence Resonance Index
    """
    from .constants import PI
    
    oscillation = math.cos(2 * PI * frequency * temporal_phase + phase_offset)
    decay = math.exp(-alpha * abs(spatial_curvature))
    
    return oscillation * decay


def coherence_resonance_index_simplified(f_avg: float, t_csc: float = 0.3183) -> float:
    """
    Calculate simplified Coherence Resonance Index.
    
    Formula: CRI = cos(2π × f_avg × t_csc)
    
    Args:
        f_avg: Average frequency
        t_csc: Temporal slice constant (default: 1/π ≈ 0.3183)
        
    Returns:
        Simplified CRI value
    """
    from .constants import PI
    
    return math.cos(2 * PI * f_avg * t_csc)


def shannon_entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy for information content measurement.
    
    Formula: H = -Σ(p_i × log₂(p_i))
    
    Args:
        probabilities: List of probability values
        
    Returns:
        Shannon entropy value
    """
    if not probabilities:
        return 0.0
    
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def toggle_distribution_entropy(active_states: List[bool]) -> float:
    """
    Calculate entropy of toggle state distribution.
    
    Args:
        active_states: List of boolean toggle states
        
    Returns:
        Entropy of the toggle distribution
    """
    if not active_states:
        return 0.0
    
    total = len(active_states)
    active_count = sum(active_states)
    inactive_count = total - active_count
    
    if active_count == 0 or inactive_count == 0:
        return 0.0  # No entropy in uniform distribution
    
    p_active = active_count / total
    p_inactive = inactive_count / total
    
    return shannon_entropy([p_active, p_inactive])


def structural_stability_factor(distances: List[float], max_distance: float,
                              active_bits: List[int]) -> float:
    """
    Calculate structural stability factor (S_opt).
    
    Formula: S_opt = 0.7 × (1 - Σd_i / √Σd_max²) + 0.3 × (Σb_j / 12)
    
    Args:
        distances: Distances to glyph center
        max_distance: Maximum possible distance
        active_bits: Active bits in Reality + Information layers
        
    Returns:
        Structural stability factor
    """
    if not distances or max_distance == 0:
        spatial_term = 0.0
    else:
        sum_distances = sum(distances)
        sqrt_sum_max_squared = math.sqrt(len(distances) * max_distance * max_distance)
        spatial_term = 1.0 - (sum_distances / sqrt_sum_max_squared)
    
    if not active_bits:
        bit_term = 0.0
    else:
        sum_active_bits = sum(active_bits)
        bit_term = sum_active_bits / 12
    
    return 0.7 * spatial_term + 0.3 * bit_term


def observability_threshold_check(coherence_value: float, threshold: float = 0.5) -> bool:
    """
    Check if coherence meets observability threshold.
    
    Args:
        coherence_value: Calculated coherence
        threshold: Observability threshold (default: 0.5)
        
    Returns:
        True if coherence is observable
    """
    return coherence_value >= threshold


def strong_coupling_check(coherence_value: float, threshold: float = 0.95) -> bool:
    """
    Check if coherence meets strong coupling threshold.
    
    Args:
        coherence_value: Calculated coherence
        threshold: Strong coupling threshold (default: 0.95)
        
    Returns:
        True if coherence indicates strong coupling
    """
    return coherence_value >= threshold


def validate_nrci_target(nrci_value: float, target: float = 0.999999) -> bool:
    """
    Validate if NRCI meets the target threshold.
    
    Args:
        nrci_value: Calculated NRCI
        target: Target NRCI (default: 0.999999 for six nines)
        
    Returns:
        True if NRCI meets target
    """
    return nrci_value >= target


def calculate_system_coherence_score(nrci: float, coherence_pressure: float,
                                   fractal_dim: float, sri: float, cri: float) -> float:
    """
    Calculate overall system coherence score.
    
    Combines multiple metrics into a single coherence assessment.
    
    Args:
        nrci: Non-Random Coherence Index
        coherence_pressure: Coherence pressure value
        fractal_dim: Fractal dimension
        sri: Spatial Resonance Index
        cri: Coherence Resonance Index
        
    Returns:
        Overall coherence score [0, 1]
    """
    # Weight the different metrics
    weights = {
        'nrci': 0.4,
        'pressure': 0.2,
        'fractal': 0.15,
        'sri': 0.15,
        'cri': 0.1
    }
    
    # Normalize coherence pressure (lower is better)
    normalized_pressure = 1.0 / (1.0 + coherence_pressure) if coherence_pressure >= 0 else 0.0
    
    # Normalize fractal dimension (target around 2.3)
    target_fractal = 2.3
    normalized_fractal = 1.0 - abs(fractal_dim - target_fractal) / target_fractal
    normalized_fractal = max(0.0, min(1.0, normalized_fractal))
    
    # Calculate weighted score
    score = (weights['nrci'] * nrci +
             weights['pressure'] * normalized_pressure +
             weights['fractal'] * normalized_fractal +
             weights['sri'] * sri +
             weights['cri'] * abs(cri))  # CRI can be negative
    
    return max(0.0, min(1.0, score))


def temporal_coherence_analysis(time_series: List[float], 
                              window_size: int = 10) -> Tuple[float, List[float]]:
    """
    Analyze temporal coherence in a time series.
    
    Args:
        time_series: Time series data
        window_size: Size of sliding window for analysis
        
    Returns:
        Tuple of (overall_coherence, windowed_coherence_values)
    """
    if len(time_series) < window_size:
        return 0.0, []
    
    windowed_coherence = []
    
    for i in range(len(time_series) - window_size + 1):
        window = time_series[i:i + window_size]
        
        # Calculate coherence within window (using variance as inverse measure)
        if len(window) > 1:
            mean_val = sum(window) / len(window)
            variance = sum((x - mean_val) ** 2 for x in window) / len(window)
            coherence = 1.0 / (1.0 + variance) if variance > 0 else 1.0
        else:
            coherence = 1.0
        
        windowed_coherence.append(coherence)
    
    overall_coherence = sum(windowed_coherence) / len(windowed_coherence) if windowed_coherence else 0.0
    
    return overall_coherence, windowed_coherence


"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Energy Equation Implementation
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the complete UBP energy equation and related calculations:
E = M × C × (R × S_opt) × P_GCI × O_observer × c_∞ × I_spin × Σ(w_ij M_ij)
"""

import math
from typing import List, Optional, Dict, Any

# Import from ubp_config for centralized constants and configurations
from ubp_config import get_config, UBPConfig, RealmConfig
# Import GlobalCoherenceIndex for P_GCI calculation
from global_coherence import GlobalCoherenceIndex
# Import ObserverScaling for O_observer (though a simplified function is implemented here)
from observer_scaling import ObserverScaling

# Initialize configuration and global coherence system at module load time
_config: UBPConfig = get_config()
_global_coherence_system: GlobalCoherenceIndex = GlobalCoherenceIndex()


def energy(M: int, C_speed: Optional[float] = None, R: Optional[float] = None, S_opt: Optional[float] = None,
          P_GCI: Optional[float] = None, O_observer: Optional[float] = None,
          c_infinity: Optional[float] = None, I_spin: float = 1.0,
          w_sum: float = 0.1) -> float:
    """
    Calculate the total UBP energy.

    Axiom: E = M × C × (R × S_opt) × P_GCI × O_observer × c_∞ × I_spin × Σ(w_ij M_ij)

    Args:
        M: Active OffBits count
        C_speed: Speed of light (m/s), defaults to config value
        R: Resonance strength, defaults to config value for R_0 and H_t
        S_opt: Structural optimality factor, defaults to config value
        P_GCI: Global Coherence Invariant, defaults to dynamic calculation
        O_observer: Observer effect factor, defaults to dynamic calculation
        c_infinity: Cosmic constant, defaults to config value
        I_spin: Spin information factor
        w_sum: Weighted toggle matrix sum

    Returns:
        Total energy value
    """
    if C_speed is None:
        C_speed = _config.constants.SPEED_OF_LIGHT
    if c_infinity is None:
        c_infinity = _config.constants.C_INFINITY
    if P_GCI is None:
        P_GCI = _global_coherence_system.compute_global_coherence_index()
    if O_observer is None:
        # Default neutral observer effect if not provided, or can be derived from ObserverScaling
        O_observer = observer_effect_factor("neutral", purpose_tensor=1.0)
    
    # Use config defaults if R or S_opt are not provided
    if R is None:
        R = resonance_strength(R_0=_config.energy.R_0_DEFAULT, H_t=_config.energy.H_T_DEFAULT)
    if S_opt is None:
        # If S_opt is not provided, we can't fully calculate structural_optimality without distances/active_bits.
        # Use the default from config.energy.S_OPT_DEFAULT as a fallback, or if specific inputs for it are missing.
        S_opt = _config.energy.S_OPT_DEFAULT


    return (M * C_speed * (R * S_opt) * P_GCI * O_observer *
            c_infinity * I_spin * w_sum)


def resonance_strength(R_0: Optional[float] = None, H_t: Optional[float] = None) -> float:
    """
    Calculate resonance strength.

    Axiom: R = R_0 × (1 - H_t / ln(4))

    Args:
        R_0: Base resonance strength (defaults to config value)
        H_t: Tonal entropy (defaults to config value)

    Returns:
        Resonance strength value
    """
    if R_0 is None:
        R_0 = _config.energy.R_0_DEFAULT
    if H_t is None:
        H_t = _config.energy.H_T_DEFAULT

    # math.log is natural log (ln)
    return R_0 * (1 - H_t / math.log(4))


def structural_optimality(distances: List[float], max_distance: float,
                         active_bits: List[int], S_opt_default: Optional[float] = None) -> float:
    """
    Calculate structural optimization factor.

    Axiom: S_opt = 0.7 × (1 - Σd_i / √Σd_max²) + 0.3 × (Σb_j / 12)

    Args:
        distances: List of distances to Glyph center
        max_distance: Maximum possible distance (Bitfield diagonal)
        active_bits: List of active bits in Information layer (0-11)
        S_opt_default: Default S_opt to use if calculation is not possible (defaults to config value)

    Returns:
        Structural optimality factor
    """
    if S_opt_default is None:
        S_opt_default = _config.energy.S_OPT_DEFAULT

    if not distances or max_distance == 0:
        spatial_term = 0.0
    else:
        sum_distances = sum(distances)
        # Assuming max_distance is the max possible distance for a single bit.
        # If max_distance represents sqrt(sum_d_max_squared) for all bits, it's simpler.
        # Given the formula, it seems to imply sum of squares of max distance for each distance.
        sqrt_sum_max_squared = math.sqrt(len(distances) * max_distance * max_distance)
        spatial_term = 1 - (sum_distances / sqrt_sum_max_squared)

    if not active_bits:
        bit_term = 0.0
    else:
        sum_active_bits = sum(active_bits)
        bit_term = sum_active_bits / 12  # 12 bits in Information layer

    # If inputs are provided, calculate; otherwise, use the default.
    if distances and max_distance > 0 and active_bits:
        return 0.7 * spatial_term + 0.3 * bit_term
    else:
        return S_opt_default


def observer_effect_factor(observation_type: str = "neutral",
                          purpose_tensor: float = 1.0) -> float:
    """
    Calculate observer effect factor.

    Formula: O_observer = 1 + (1/4π) * log(s/s_0) * F_μν(ψ)
    Simplified: 1.0 (neutral) or 1.5 (intentional)

    Args:
        observation_type: "neutral" or "intentional"
        purpose_tensor: Purpose tensor value

    Returns:
        Observer effect factor
    """
    if observation_type == "neutral":
        return _config.observer.DEFAULT_INTENT_LEVEL
    elif observation_type == "intentional":
        # Use MAX_INTENT_LEVEL from config as a base for intentional
        # This is a simplified mapping; a real purpose tensor would be passed from DotTheory
        return _config.observer.MAX_INTENT_LEVEL
    else:
        # General formula (simplified using constants from config)
        C_PI = _config.constants.PI
        k = 1.0 / (4 * C_PI)
        # Assuming s/s_0 is implicitly handled by purpose_tensor or default to 1.0
        return 1.0 + k * math.log(purpose_tensor + _config.constants.EPSILON_UBP)


def cosmic_constant(phi: Optional[float] = None, alpha: Optional[float] = None) -> float:
    """
    Calculate the cosmic constant c_∞.

    Formula: c_∞ = 24 × φ × (1 + α)

    Args:
        phi: Golden ratio, defaults to config value
        alpha: Fine-structure constant, defaults to config value

    Returns:
        Cosmic constant value
    """
    if phi is None:
        phi = _config.constants.PHI
    if alpha is None:
        alpha = _config.constants.FINE_STRUCTURE_CONSTANT

    return 24 * phi * (1 + alpha)


def spin_information_factor(spin_probabilities: List[float]) -> float:
    """
    Calculate spin information factor using Shannon entropy.

    Formula: I_spin = Σ p_s × ln(1/p_s)

    Args:
        spin_probabilities: List of spin state probabilities

    Returns:
        Spin information factor
    """
    if not spin_probabilities:
        return 1.0

    entropy = 0.0
    for p_s in spin_probabilities:
        if p_s > 0:
            entropy += p_s * math.log(1.0 / p_s)
    return entropy


def quantum_spin_entropy(p_s: Optional[float] = None) -> float:
    """
    Calculate spin entropy for quantum realm.

    Args:
        p_s: Spin probability (default: config's quantum toggle prob)

    Returns:
        Quantum spin entropy
    """
    if p_s is None:
        # Use quantum toggle probability from config's realm settings
        p_s = _config.constants.UBP_TOGGLE_PROBABILITIES.get('quantum', _config.constants.E / 12)

    if p_s <= 0 or p_s >= 1:
        return 0.0

    return p_s * math.log(1.0 / p_s) + (1 - p_s) * math.log(1.0 / (1 - p_s))


def cosmological_spin_entropy(p_s: Optional[float] = None) -> float:
    """
    Calculate spin entropy for cosmological realm.

    Args:
        p_s: Spin probability (default: config's cosmological toggle prob)

    Returns:
        Cosmological spin entropy
    """
    if p_s is None:
        # Use cosmological toggle probability from config's realm settings
        p_s = _config.constants.UBP_TOGGLE_PROBABILITIES.get('cosmological', _config.constants.PI ** _config.constants.PHI)

    if p_s <= 0 or p_s >= 1:
        return 0.0

    return p_s * math.log(1.0 / p_s) + (1 - p_s) * math.log(1.0 / (1 - p_s))


def weighted_toggle_matrix_sum(weights: List[float], toggle_operations: List[float]) -> float:
    """
    Calculate weighted sum of toggle operations.

    Formula: Σ(w_ij × M_ij)

    Args:
        weights: Interaction weights (must sum to 1)
        toggle_operations: Toggle operation results

    Returns:
        Weighted sum
    """
    if len(weights) != len(toggle_operations):
        raise ValueError("Weights and operations must have same length")

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0

    normalized_weights = [w / total_weight for w in weights]

    return sum(w * op for w, op in zip(normalized_weights, toggle_operations))


def calculate_energy_for_realm(realm_name: str, active_offbits: int,
                              distances: Optional[List[float]] = None,
                              max_distance: Optional[float] = None,
                              active_bits: Optional[List[int]] = None) -> float:
    """
    Calculate energy for a specific realm using realm-specific parameters.

    Args:
        realm_name: Name of the realm
        active_offbits: Number of active OffBits
        distances: Optional distances for S_opt calculation
        max_distance: Optional max distance for S_opt calculation
        active_bits: Optional active bits for S_opt calculation

    Returns:
        Energy value for the realm
    """
    realm_config = _config.get_realm_config(realm_name)
    
    # Use config default or hardcoded fallback if realm_config not found or specific values missing
    R = resonance_strength(R_0=_config.energy.R_0_DEFAULT, H_t=_config.energy.H_T_DEFAULT)

    # S_opt calculation
    if distances and max_distance and active_bits:
        S_opt = structural_optimality(distances, max_distance, active_bits, _config.energy.S_OPT_DEFAULT)
    else:
        S_opt = _config.energy.S_OPT_DEFAULT  # Use config default if no specific values provided

    # P_GCI from the global coherence system
    P_GCI = _global_coherence_system.compute_global_coherence_index()

    # O_observer can be set to neutral for a basic calculation here
    O_observer = observer_effect_factor("neutral", purpose_tensor=1.0)

    # c_infinity from the config
    c_infinity = _config.constants.C_INFINITY
    
    # I_spin and w_sum are not directly derived from realm_config or bitfield state in this helper,
    # so they should be provided if needed or default to 1.0 and 0.1 respectively as in main energy().
    # For now, use the default values for I_spin and w_sum in the main energy() function.
    
    return energy(
        M=active_offbits,
        R=R,
        S_opt=S_opt,
        P_GCI=P_GCI,
        O_observer=O_observer,
        c_infinity=c_infinity,
        I_spin=1.0, # Defaulting for this helper function; typically passed to main energy()
        w_sum=0.1   # Defaulting for this helper function; typically passed to main energy()
    )


def energy_conservation_check(initial_energy: float, final_energy: float,
                            tolerance: float = 1e-10) -> bool:
    """
    Check if energy is conserved within tolerance.

    Args:
        initial_energy: Energy before operation
        final_energy: Energy after operation
        tolerance: Acceptable difference

    Returns:
        True if energy is conserved
    """
    return abs(final_energy - initial_energy) <= tolerance


def calculate_energy_density(energy: float, volume: float) -> float:
    """
    Calculate energy density.

    Args:
        energy: Total energy
        volume: Volume of the region

    Returns:
        Energy density (energy per unit volume)
    """
    if volume <= 0:
        return 0.0

    return energy / volume


def energy_from_frequency(frequency: float, num_quanta: int = 1) -> float:
    """
    Calculate energy from frequency using Planck relation.

    Formula: E = n × h × f

    Args:
        frequency: Frequency in Hz
        num_quanta: Number of energy quanta

    Returns:
        Energy value
    """
    h = _config.constants.PLANCK_CONSTANT
    return num_quanta * h * frequency


def energy_efficiency_ratio(actual_energy: float, theoretical_max: float) -> float:
    """
    Calculate energy efficiency ratio.

    Args:
        actual_energy: Actual measured energy
        theoretical_max: Theoretical maximum energy

    Returns:
        Efficiency ratio [0, 1]
    """
    if theoretical_max <= 0:
        return 0.0

    return min(1.0, actual_energy / theoretical_max)


def validate_energy_bounds(energy_value: float, min_energy: float = 0.0,
                          max_energy: Optional[float] = None) -> bool:
    """
    Validate that energy value is within acceptable bounds.

    Args:
        energy_value: Energy to validate
        min_energy: Minimum acceptable energy
        max_energy: Maximum acceptable energy (None for no limit)

    Returns:
        True if energy is within bounds
    """
    if energy_value < min_energy:
        return False

    if max_energy is not None and energy_value > max_energy:
        return False

    return True
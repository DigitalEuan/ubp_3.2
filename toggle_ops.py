"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Toggle Algebra Operations
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

Implements the fundamental toggle operations that govern OffBit interactions:
- Basic operations: AND, XOR, OR
- Advanced operations: Resonance, Entanglement, Superposition
- Specialized operations: Hybrid XOR Resonance, Spin Transition
"""

import math
from typing import List, Union
from state import OffBit # Changed from relative import
from kernels import resonance_kernel # Changed from relative import
from ubp_config import get_config # Import get_config to access centralized constants

_config = get_config() # Initialize configuration


def toggle_and(b_i: Union[int, OffBit], b_j: Union[int, OffBit]) -> Union[int, OffBit]:
    """
    Perform AND toggle operation.
    
    Axiom: min(b_i, b_j)
    Purpose: Logical conjunction; both bits must be 'on' for outcome to be 'on'
    
    Args:
        b_i: First OffBit or integer value
        b_j: Second OffBit or integer value
        
    Returns:
        Result of AND operation
    """
    # Extract values if OffBits
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    val_j = b_j.value if isinstance(b_j, OffBit) else b_j
    
    result = min(val_i, val_j)
    
    # Return same type as input
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def toggle_xor(b_i: Union[int, OffBit], b_j: Union[int, OffBit]) -> Union[int, OffBit]:
    """
    Perform XOR toggle operation.
    
    Axiom: |b_i - b_j|
    Purpose: Exclusive OR; outcome is 'on' if bits are different
    
    Args:
        b_i: First OffBit or integer value
        b_j: Second OffBit or integer value
        
    Returns:
        Result of XOR operation
    """
    # Extract values if OffBits
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    val_j = b_j.value if isinstance(b_j, OffBit) else b_j
    
    result = abs(val_i - val_j)
    
    # Return same type as input
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def toggle_or(b_i: Union[int, OffBit], b_j: Union[int, OffBit]) -> Union[int, OffBit]:
    """
    Perform OR toggle operation.
    
    Axiom: max(b_i, b_j)
    Purpose: Logical disjunction; at least one bit must be 'on'
    
    Args:
        b_i: First OffBit or integer value
        b_j: Second OffBit or integer value
        
    Returns:
        Result of OR operation
    """
    # Extract values if OffBits
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    val_j = b_j.value if isinstance(b_j, OffBit) else b_j
    
    result = max(val_i, val_j)
    
    # Return same type as input
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def resonance_toggle(b_i: Union[int, OffBit], frequency: float, time: float, 
                    k: float = 0.0002) -> Union[int, OffBit]:
    """
    Perform resonance toggle operation.
    
    Axiom: b_i × exp(-k × (t × f)²)
    Purpose: State transitions with distance-based decay
    
    Args:
        b_i: OffBit or integer value
        frequency: Resonance frequency
        time: Time parameter
        k: Decay constant
        
    Returns:
        Result of resonance operation
    """
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    
    d = time * frequency
    resonance_factor = resonance_kernel(d, k)
    result = int(val_i * resonance_factor)
    
    # Ensure result stays within valid range
    result = max(0, min(result, 0xFFFFFF))  # 24-bit limit
    
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def entanglement_toggle(b_i: Union[int, OffBit], b_j: Union[int, OffBit], 
                       coherence: float) -> Union[int, OffBit]:
    """
    Perform entanglement toggle operation.
    
    Axiom: b_i × b_j × C_ij (where C_ij ≥ 0.95)
    Purpose: Cross-layer coupling between OffBits
    
    Args:
        b_i: First OffBit or integer value
        b_j: Second OffBit or integer value
        coherence: Coherence factor (should be ≥ 0.95 for strong entanglement)
        
    Returns:
        Result of entanglement operation
    """
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    val_j = b_j.value if isinstance(b_j, OffBit) else b_j
    
    # Only apply entanglement if coherence meets threshold
    if coherence >= 0.95:
        result = int(val_i * val_j * coherence)
    else:
        # Weak entanglement - use reduced coupling
        result = int(val_i * val_j * coherence * 0.1)
    
    # Ensure result stays within valid range
    result = max(0, min(result, 0xFFFFFF))  # 24-bit limit
    
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def superposition_toggle(states: List[Union[int, OffBit]], 
                        weights: List[float]) -> Union[int, OffBit]:
    """
    Perform superposition toggle operation.
    
    Axiom: Σ(states × weights) where Σ weights = 1
    Purpose: Probabilistic state modeling
    
    Args:
        states: List of OffBit states or integer values
        weights: List of probability weights (must sum to 1)
        
    Returns:
        Result of superposition operation
    """
    if len(states) != len(weights):
        raise ValueError("States and weights must have same length")
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        if isinstance(states[0], OffBit):
            return OffBit(0)
        else:
            return 0
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted sum
    result = 0.0
    for state, weight in zip(states, normalized_weights):
        val = state.value if isinstance(state, OffBit) else state
        result += val * weight
    
    result = int(result)
    result = max(0, min(result, 0xFFFFFF))  # 24-bit limit
    
    if isinstance(states[0], OffBit):
        return OffBit(result)
    else:
        return result


def hybrid_xor_resonance(b_i: Union[int, OffBit], b_j: Union[int, OffBit], 
                        d: float, k: float = 0.0002) -> Union[int, OffBit]:
    """
    Perform hybrid XOR resonance operation.
    
    Axiom: |b_i - b_j| × exp(-k × d²)
    Purpose: Differential interactions with distance dependency
    
    Args:
        b_i: First OffBit or integer value
        b_j: Second OffBit or integer value
        d: Distance parameter
        k: Decay constant
        
    Returns:
        Result of hybrid XOR resonance operation
    """
    # First apply XOR
    xor_result = toggle_xor(b_i, b_j)
    
    # Then apply resonance decay
    val = xor_result.value if isinstance(xor_result, OffBit) else xor_result
    resonance_factor = resonance_kernel(d, k)
    result = int(val * resonance_factor)
    
    # Ensure result stays within valid range
    result = max(0, min(result, 0xFFFFFF))  # 24-bit limit
    
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def spin_transition(b_i: Union[int, OffBit], p_s: float) -> Union[int, OffBit]:
    """
    Perform spin transition operation.
    
    Axiom: b_i × ln(1/p_s)
    Purpose: Probabilistic spin state transitions
    
    Args:
        b_i: OffBit or integer value
        p_s: Spin probability (e.g., e/12 for quantum, π^φ for cosmological)
        
    Returns:
        Result of spin transition operation
    """
    if p_s <= 0 or p_s >= 1:
        raise ValueError(f"Spin probability must be in (0, 1), got {p_s}")
    
    val_i = b_i.value if isinstance(b_i, OffBit) else b_i
    
    transition_factor = math.log(1.0 / p_s)
    result = int(val_i * transition_factor)
    
    # Ensure result stays within valid range
    result = max(0, min(result, 0xFFFFFF))  # 24-bit limit
    
    if isinstance(b_i, OffBit):
        return OffBit(result)
    else:
        return result


def quantum_spin_transition(b_i: Union[int, OffBit]) -> Union[int, OffBit]:
    """
    Perform quantum realm spin transition.
    
    Uses quantum spin probability p_s = e/12 ≈ 0.2265234857
    
    Args:
        b_i: OffBit or integer value
        
    Returns:
        Result of quantum spin transition
    """
    p_s = _config.constants.E / 12  # e/12
    return spin_transition(b_i, p_s)


def cosmological_spin_transition(b_i: Union[int, OffBit]) -> Union[int, OffBit]:
    """
    Perform cosmological realm spin transition.
    
    Uses cosmological spin probability p_s = π^φ ≈ 0.83203682
    
    Args:
        b_i: OffBit or integer value
        
    Returns:
        Result of cosmological spin transition
    """
    p_s = _config.constants.PI ** _config.constants.PHI  # π^φ
    return spin_transition(b_i, p_s)


def apply_tgic_constraint(x_state: bool, y_state: bool, z_state: bool,
                         b_i: Union[int, OffBit], b_j: Union[int, OffBit],
                         **kwargs) -> Union[int, OffBit]:
    """
    Apply Triad Graph Interaction Constraint (TGIC) rules.
    
    Rules:
    - (X=1, Y=1, Z=1) → Hybrid_XOR_Resonance or Spin_Transition
    - (X=1, Y=1, Z=0) → Resonance
    - (X=1, Y=0, Z=1) → Entanglement
    - (Y=1, Z=1, X=0) → Superposition
    
    Args:
        x_state: X axis state
        y_state: Y axis state
        z_state: Z axis state
        b_i: First OffBit
        b_j: Second OffBit
        **kwargs: Additional parameters for specific operations
        
    Returns:
        Result of TGIC-determined operation
    """
    if x_state and y_state and z_state:
        # (1,1,1) → Hybrid_XOR_Resonance or Spin_Transition
        # Choose based on additional criteria or default to Hybrid_XOR_Resonance
        d = kwargs.get('distance', 1.0)
        return hybrid_xor_resonance(b_i, b_j, d)
    
    elif x_state and y_state and not z_state:
        # (1,1,0) → Resonance
        frequency = kwargs.get('frequency', 1.0)
        time = kwargs.get('time', 1.0)
        return resonance_toggle(b_i, frequency, time)
    
    elif x_state and not y_state and z_state:
        # (1,0,1) → Entanglement
        coherence = kwargs.get('coherence', 0.95)
        return entanglement_toggle(b_i, b_j, coherence)
    
    elif not x_state and y_state and z_state:
        # (0,1,1) → Superposition
        weights = kwargs.get('weights', [0.5, 0.5])
        return superposition_toggle([b_i, b_j], weights)
    
    else:
        # Default case - use basic XOR
        return toggle_xor(b_i, b_j)


def chaos_correction_logistic(f_i: float, f_max: float = 2e-15) -> float:
    """
    Apply chaos correction using logistic map.
    
    Formula: f_i(t+1) = 4 × f_i(t) × (1 - f_i(t) / f_max)
    
    Args:
        f_i: Current frequency state
        f_max: Maximum frequency (default: 2×10^-15)
        
    Returns:
        Corrected frequency state
    """
    if f_max == 0:
        return 0.0
    
    normalized_f = f_i / f_max
    return 4 * normalized_f * (1 - normalized_f) * f_max


def validate_toggle_result(result: Union[int, OffBit], 
                          max_value: int = 0xFFFFFF) -> bool:
    """
    Validate that toggle operation result is within acceptable bounds.
    
    Args:
        result: Toggle operation result
        max_value: Maximum allowed value (24-bit default)
        
    Returns:
        True if result is valid
    """
    val = result.value if isinstance(result, OffBit) else result
    return 0 <= val <= max_value


def toggle_operation_energy_cost(operation_type: str, 
                                complexity_factor: float = 1.0) -> float:
    """
    Calculate energy cost of toggle operation.
    
    Args:
        operation_type: Type of toggle operation
        complexity_factor: Complexity multiplier
        
    Returns:
        Energy cost estimate
    """
    base_costs = {
        'and': 1.0,
        'xor': 1.0,
        'or': 1.0,
        'resonance': 2.0,
        'entanglement': 3.0,
        'superposition': 2.5,
        'hybrid_xor_resonance': 3.5,
        'spin_transition': 2.0
    }
    
    base_cost = base_costs.get(operation_type.lower(), 1.0)
    return base_cost * complexity_factor
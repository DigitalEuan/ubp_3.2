"""
Universal Binary Principle (UBP) Framework v3.2+ - Observer Scaling (O_observer) for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the observer-dependent scaling mechanism that modulates physics
based on observer intent and purpose tensor interactions.

Mathematical Foundation:
- O_observer = 1 + (1/4π) × log(s/s₀) × F_μν(ψ)
- F_μν(ψ): Purpose tensor (1.0 neutral, 1.5 intentional)
- s/s₀: Scale ratio relative to baseline
- Intent modulation affects physical constants and system behavior

Based on research by:
Lilian, A. Qualianomics: The Ontological Science of Experience. https://therootsofreality.buzzsprout.com/2523361 
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Import UBPConfig and get_config for constant loading
from ubp_config import get_config, UBPConfig

_config: UBPConfig = get_config() # Initialize configuration


class ObserverIntentType(Enum):
    """Types of observer intent that affect system scaling"""
    NEUTRAL = "neutral"           # No specific intent (F_μν = 1.0)
    INTENTIONAL = "intentional"   # Focused intent (F_μν = 1.5)
    EXPLORATORY = "exploratory"   # Discovery intent (F_μν = 1.2)
    CREATIVE = "creative"         # Creative intent (F_μν = 1.3)
    ANALYTICAL = "analytical"     # Analysis intent (F_μν = 1.1)
    MEDITATIVE = "meditative"     # Contemplative intent (F_μν = 0.9)
    DESTRUCTIVE = "destructive"   # Destructive intent (F_μν = 0.7)


class ScaleRegime(Enum):
    """Different scaling regimes for observer effects"""
    QUANTUM = "quantum"           # Quantum scale effects
    MESOSCOPIC = "mesoscopic"     # Intermediate scale
    MACROSCOPIC = "macroscopic"   # Classical scale
    COSMOLOGICAL = "cosmological" # Large scale effects


@dataclass
class ObserverState:
    """
    Represents the current state of an observer in the UBP system.
    """
    intent_type: ObserverIntentType
    focus_intensity: float  # 0.0 to 1.0
    coherence_level: float  # Current NRCI level
    temporal_stability: float  # Stability over time
    scale_preference: ScaleRegime
    purpose_tensor_value: float
    baseline_scale: float = 1.0
    
    def __post_init__(self):
        # Validate ranges
        self.focus_intensity = max(0.0, min(1.0, self.focus_intensity))
        self.coherence_level = max(0.0, min(1.0, self.coherence_level))
        self.temporal_stability = max(0.0, min(1.0, self.temporal_stability))


@dataclass
class ScalingParameters:
    """
    Parameters for observer scaling calculations.
    """
    four_pi_inverse: float = 1.0 / (4 * _config.constants.PI)  # 1/4π, uses UBPConfig
    baseline_scale_s0: float = 1.0  # Reference scale
    intent_amplification: float = 1.0  # Amplification factor
    temporal_decay_rate: float = 0.95  # Decay rate for temporal effects
    coherence_threshold: float = _config.performance.COHERENCE_THRESHOLD  # Minimum coherence for scaling, uses UBPConfig
    max_scaling_factor: float = 10.0  # Maximum allowed scaling
    min_scaling_factor: float = 0.1   # Minimum allowed scaling


class PurposeTensor:
    """
    Implements the purpose tensor F_μν(ψ) for observer intent quantification.
    
    The purpose tensor encodes the observer's intent and its effect on
    the physical system through the UBP framework.
    """
    
    def __init__(self):
        self.intent_mappings = {
            ObserverIntentType.NEUTRAL: 1.0,
            ObserverIntentType.INTENTIONAL: 1.5,
            ObserverIntentType.EXPLORATORY: 1.2,
            ObserverIntentType.CREATIVE: 1.3,
            ObserverIntentType.ANALYTICAL: 1.1,
            ObserverIntentType.MEDITATIVE: 0.9,
            ObserverIntentType.DESTRUCTIVE: 0.7
        }
        
        self._tensor_cache = {}
    
    def compute_purpose_tensor(self, observer_state: ObserverState) -> float:
        """
        Compute the purpose tensor F_μν(ψ) for given observer state.
        
        Args:
            observer_state: Current observer state
        
        Returns:
            Purpose tensor value
        """
        base_value = self.intent_mappings[observer_state.intent_type]
        
        # Modulate by focus intensity and coherence
        focus_modulation = 1.0 + (observer_state.focus_intensity - 0.5) * 0.2
        coherence_modulation = 1.0 + (observer_state.coherence_level - 0.5) * 0.1
        stability_modulation = 1.0 + (observer_state.temporal_stability - 0.5) * 0.05
        
        # Combine modulations
        modulated_value = base_value * focus_modulation * coherence_modulation * stability_modulation
        
        # Cache for efficiency
        cache_key = (
            observer_state.intent_type,
            round(observer_state.focus_intensity, 3),
            round(observer_state.coherence_level, 3),
            round(observer_state.temporal_stability, 3)
        )
        self._tensor_cache[cache_key] = modulated_value
        
        return modulated_value
    
    def get_tensor_gradient(self, observer_state: ObserverState, 
                          parameter: str, delta: float = 0.001) -> float:
        """
        Compute gradient of purpose tensor with respect to observer parameter.
        
        Args:
            observer_state: Current observer state
            parameter: Parameter to compute gradient for
            delta: Small change for numerical differentiation
        
        Returns:
            Gradient value
        """
        original_value = self.compute_purpose_tensor(observer_state)
        
        # Create modified state
        modified_state = ObserverState(
            intent_type=observer_state.intent_type,
            focus_intensity=observer_state.focus_intensity,
            coherence_level=observer_state.coherence_level,
            temporal_stability=observer_state.temporal_stability,
            scale_preference=observer_state.scale_preference,
            purpose_tensor_value=observer_state.purpose_tensor_value,
            baseline_scale=observer_state.baseline_scale
        )
        
        # Modify the specified parameter
        if parameter == 'focus_intensity':
            modified_state.focus_intensity = min(1.0, observer_state.focus_intensity + delta)
        elif parameter == 'coherence_level':
            modified_state.coherence_level = min(1.0, observer_state.coherence_level + delta)
        elif parameter == 'temporal_stability':
            modified_state.temporal_stability = min(1.0, observer_state.temporal_stability + delta)
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        modified_value = self.compute_purpose_tensor(modified_state)
        gradient = (modified_value - original_value) / delta
        
        return gradient


class ObserverScaling:
    """
    Main Observer Scaling system for UBP.
    
    Implements the complete observer-dependent physics scaling mechanism
    using the formula: O_observer = 1 + (1/4π) × log(s/s₀) × F_μν(ψ)
    """
    
    def __init__(self, parameters: Optional[ScalingParameters] = None):
        self.parameters = parameters or ScalingParameters()
        self.purpose_tensor = PurposeTensor()
        self._scaling_history = []
        self._observer_states = {}
        
    def compute_observer_scaling(self, observer_state: ObserverState, 
                               current_scale: float) -> float:
        """
        Compute the observer scaling factor O_observer.
        
        O_observer = 1 + (1/4π) × log(s/s₀) × F_μν(ψ)
        
        Args:
            observer_state: Current observer state
            current_scale: Current system scale
        
        Returns:
            Observer scaling factor
        """
        # Compute purpose tensor
        F_muv = self.purpose_tensor.compute_purpose_tensor(observer_state)
        
        # Compute scale ratio
        s_over_s0 = current_scale / self.parameters.baseline_scale_s0
        
        # Avoid log(0) or log(negative)
        if s_over_s0 <= 0:
            s_over_s0 = 1e-10
        
        # Compute observer scaling: O_observer = 1 + (1/4π) × log(s/s₀) × F_μν(ψ)
        log_scale_ratio = math.log(s_over_s0)
        scaling_contribution = self.parameters.four_pi_inverse * log_scale_ratio * F_muv
        
        # Apply intent amplification
        scaling_contribution *= self.parameters.intent_amplification
        
        # Apply coherence gating (low coherence reduces scaling effects)
        if observer_state.coherence_level < self.parameters.coherence_threshold:
            coherence_factor = observer_state.coherence_level / self.parameters.coherence_threshold
            scaling_contribution *= coherence_factor
        
        O_observer = 1.0 + scaling_contribution
        
        # Apply bounds to prevent extreme scaling
        O_observer = max(self.parameters.min_scaling_factor, 
                        min(self.parameters.max_scaling_factor, O_observer))
        
        # Record scaling event
        scaling_event = {
            'timestamp': time.time(),
            'observer_state': observer_state,
            'current_scale': current_scale,
            'purpose_tensor': F_muv,
            'scale_ratio': s_over_s0,
            'scaling_factor': O_observer,
            'scaling_contribution': scaling_contribution
        }
        self._scaling_history.append(scaling_event)
        
        return O_observer
    
    def compute_multi_observer_scaling(self, observer_states: List[ObserverState],
                                     observer_weights: List[float],
                                     current_scale: float) -> float:
        """
        Compute observer scaling for multiple observers with weights.
        
        Args:
            observer_states: List of observer states
            observer_weights: Weights for each observer
            current_scale: Current system scale
        
        Returns:
            Combined observer scaling factor
        """
        if len(observer_states) != len(observer_weights):
            raise ValueError("Number of observers must match number of weights")
        
        if not observer_states:
            return 1.0
        
        # Normalize weights
        total_weight = sum(observer_weights)
        if total_weight == 0:
            return 1.0
        
        normalized_weights = [w / total_weight for w in observer_weights]
        
        # Compute weighted average of scaling factors
        total_scaling = 0.0
        for observer_state, weight in zip(observer_states, normalized_weights):
            individual_scaling = self.compute_observer_scaling(observer_state, current_scale)
            total_scaling += weight * individual_scaling
        
        return total_scaling
    
    def update_observer_state(self, observer_id: str, new_state: ObserverState):
        """
        Update the state of a tracked observer.
        
        Args:
            observer_id: Unique identifier for the observer
            new_state: New observer state
        """
        self._observer_states[observer_id] = new_state
    
    def get_observer_state(self, observer_id: str) -> Optional[ObserverState]:
        """
        Get the current state of a tracked observer.
        
        Args:
            observer_id: Unique identifier for the observer
        
        Returns:
            Observer state if found, None otherwise
        """
        return self._observer_states.get(observer_id)
    
    def compute_temporal_scaling_evolution(self, observer_state: ObserverState,
                                         time_points: List[float],
                                         scale_function: callable) -> List[float]:
        """
        Compute evolution of observer scaling over time.
        
        Args:
            observer_state: Observer state
            time_points: Time points to evaluate
            scale_function: Function that returns scale at given time
        
        Returns:
            List of observer scaling factors over time
        """
        scaling_evolution = []
        
        for t in time_points:
            current_scale = scale_function(t)
            
            # Apply temporal decay to observer effects
            decay_factor = self.parameters.temporal_decay_rate ** t
            
            # Create temporally modified observer state
            temp_state = ObserverState(
                intent_type=observer_state.intent_type,
                focus_intensity=observer_state.focus_intensity * decay_factor,
                coherence_level=observer_state.coherence_level,
                temporal_stability=observer_state.temporal_stability * decay_factor,
                scale_preference=observer_state.scale_preference,
                purpose_tensor_value=observer_state.purpose_tensor_value,
                baseline_scale=observer_state.baseline_scale
            )
            
            scaling_factor = self.compute_observer_scaling(temp_state, current_scale)
            scaling_evolution.append(scaling_factor)
        
        return scaling_evolution
    
    def analyze_scaling_sensitivity(self, observer_state: ObserverState,
                                  scale_range: Tuple[float, float],
                                  num_points: int = 100) -> Dict[str, Any]:
        """
        Analyze sensitivity of observer scaling to scale changes.
        
        Args:
            observer_state: Observer state to analyze
            scale_range: Range of scales to test (min, max)
            num_points: Number of points to sample
        
        Returns:
            Dictionary containing sensitivity analysis results
        """
        scales = np.linspace(scale_range[0], scale_range[1], num_points)
        scaling_factors = []
        
        for scale in scales:
            scaling_factor = self.compute_observer_scaling(observer_state, scale)
            scaling_factors.append(scaling_factor)
        
        scaling_factors = np.array(scaling_factors)
        
        # Compute sensitivity metrics
        mean_scaling = np.mean(scaling_factors)
        std_scaling = np.std(scaling_factors)
        min_scaling = np.min(scaling_factors)
        max_scaling = np.max(scaling_factors)
        
        # Compute gradient (numerical derivative)
        gradients = np.gradient(scaling_factors, scales)
        mean_gradient = np.mean(gradients)
        max_gradient = np.max(np.abs(gradients))
        
        return {
            'scales': scales.tolist(),
            'scaling_factors': scaling_factors.tolist(),
            'mean_scaling': mean_scaling,
            'std_scaling': std_scaling,
            'min_scaling': min_scaling,
            'max_scaling': max_scaling,
            'scaling_range': max_scaling - min_scaling,
            'mean_gradient': mean_gradient,
            'max_gradient': max_gradient,
            'sensitivity_index': std_scaling / mean_scaling if mean_scaling > 0 else 0.0
        }
    
    def optimize_observer_state(self, target_scaling: float, 
                              initial_state: ObserverState,
                              current_scale: float,
                              max_iterations: int = 100) -> ObserverState:
        """
        Optimize observer state to achieve target scaling factor.
        
        Args:
            target_scaling: Desired scaling factor
            initial_state: Starting observer state
            current_scale: Current system scale
            max_iterations: Maximum optimization iterations
        
        Returns:
            Optimized observer state
        """
        current_state = ObserverState(
            intent_type=initial_state.intent_type,
            focus_intensity=initial_state.focus_intensity,
            coherence_level=initial_state.coherence_level,
            temporal_stability=initial_state.temporal_stability,
            scale_preference=initial_state.scale_preference,
            purpose_tensor_value=initial_state.purpose_tensor_value,
            baseline_scale=initial_state.baseline_scale
        )
        
        learning_rate = 0.01
        tolerance = 0.001
        
        for iteration in range(max_iterations):
            current_scaling = self.compute_observer_scaling(current_state, current_scale)
            error = target_scaling - current_scaling
            
            if abs(error) < tolerance:
                break
            
            # Compute gradients
            focus_gradient = self.purpose_tensor.get_tensor_gradient(
                current_state, 'focus_intensity'
            )
            coherence_gradient = self.purpose_tensor.get_tensor_gradient(
                current_state, 'coherence_level'
            )
            stability_gradient = self.purpose_tensor.get_tensor_gradient(
                current_state, 'temporal_stability'
            )
            
            # Update state parameters
            current_state.focus_intensity += learning_rate * error * focus_gradient
            current_state.coherence_level += learning_rate * error * coherence_gradient
            current_state.temporal_stability += learning_rate * error * stability_gradient
            
            # Apply bounds
            current_state.focus_intensity = max(0.0, min(1.0, current_state.focus_intensity))
            current_state.coherence_level = max(0.0, min(1.0, current_state.coherence_level))
            current_state.temporal_stability = max(0.0, min(1.0, current_state.temporal_stability))
        
        return current_state
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about observer scaling operations.
        
        Returns:
            Dictionary containing scaling statistics
        """
        if not self._scaling_history:
            return {'statistics': 'no_scaling_events'}
        
        scaling_factors = [event['scaling_factor'] for event in self._scaling_history]
        purpose_tensors = [event['purpose_tensor'] for event in self._scaling_history]
        scale_ratios = [event['scale_ratio'] for event in self._scaling_history]
        
        return {
            'total_events': len(self._scaling_history),
            'scaling_factors': {
                'mean': np.mean(scaling_factors),
                'std': np.std(scaling_factors),
                'min': np.min(scaling_factors),
                'max': np.max(scaling_factors),
                'median': np.median(scaling_factors)
            },
            'purpose_tensors': {
                'mean': np.mean(purpose_tensors),
                'std': np.std(purpose_tensors),
                'min': np.min(purpose_tensors),
                'max': np.max(purpose_tensors)
            },
            'scale_ratios': {
                'mean': np.mean(scale_ratios),
                'std': np.std(scale_ratios),
                'min': np.min(scale_ratios),
                'max': np.max(scale_ratios)
            },
            'recent_events': self._scaling_history[-5:] if len(self._scaling_history) >= 5 else self._scaling_history
        }
    
    def validate_observer_scaling(self) -> Dict[str, Any]:
        """
        Validate the observer scaling system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'formula_implementation': True,
            'purpose_tensor_calculation': True,
            'scaling_bounds': True,
            'multi_observer_support': True,
            'temporal_evolution': True
        }
        
        try:
            # Test 1: Basic formula implementation
            neutral_state = ObserverState(
                intent_type=ObserverIntentType.NEUTRAL,
                focus_intensity=0.5,
                coherence_level=0.8,
                temporal_stability=0.9,
                scale_preference=ScaleRegime.MACROSCOPIC,
                purpose_tensor_value=1.0
            )
            
            scaling_factor = self.compute_observer_scaling(neutral_state, 1.0)
            
            # For neutral state at scale 1.0, should be close to 1.0
            if abs(scaling_factor - 1.0) > 0.1:
                validation_results['formula_implementation'] = False
                validation_results['formula_error'] = f"Expected ~1.0, got {scaling_factor}"
            
            # Test 2: Purpose tensor calculation
            intentional_state = ObserverState(
                intent_type=ObserverIntentType.INTENTIONAL,
                focus_intensity=1.0,
                coherence_level=1.0,
                temporal_stability=1.0,
                scale_preference=ScaleRegime.QUANTUM,
                purpose_tensor_value=1.5
            )
            
            purpose_tensor = self.purpose_tensor.compute_purpose_tensor(intentional_state)
            if purpose_tensor <= 1.0:  # Should be > 1.0 for intentional
                validation_results['purpose_tensor_calculation'] = False
                validation_results['tensor_error'] = f"Expected > 1.0, got {purpose_tensor}"
            
            # Test 3: Scaling bounds
            extreme_state = ObserverState(
                intent_type=ObserverIntentType.INTENTIONAL,
                focus_intensity=1.0,
                coherence_level=1.0,
                temporal_stability=1.0,
                scale_preference=ScaleRegime.QUANTUM,
                purpose_tensor_value=1.5
            )
            
            extreme_scaling = self.compute_observer_scaling(extreme_state, 1000.0)
            if not (self.parameters.min_scaling_factor <= extreme_scaling <= self.parameters.max_scaling_factor):
                validation_results['scaling_bounds'] = False
                validation_results['bounds_error'] = f"Scaling {extreme_scaling} outside bounds"
            
            # Test 4: Multi-observer support
            multi_scaling = self.compute_multi_observer_scaling(
                [neutral_state, intentional_state],
                [0.5, 0.5],
                1.0
            )
            
            if not isinstance(multi_scaling, float):
                validation_results['multi_observer_support'] = False
                validation_results['multi_error'] = "Multi-observer scaling failed"
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['formula_implementation'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_observer_scaling_system(intent_amplification: float = 1.0,
                                 max_scaling: float = 10.0) -> ObserverScaling:
    """
    Create an Observer Scaling system with specified configuration.
    
    Args:
        intent_amplification: Amplification factor for intent effects
        max_scaling: Maximum allowed scaling factor
    
    Returns:
        Configured ObserverScaling instance
    """
    parameters = ScalingParameters(
        intent_amplification=intent_amplification,
        max_scaling_factor=max_scaling
    )
    return ObserverScaling(parameters)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing Observer Scaling system...")
    
    observer_system = create_observer_scaling_system()
    
    # Test with different observer states
    print("\nTesting observer scaling with different intent types...")
    
    # Neutral observer
    neutral_observer = ObserverState(
        intent_type=ObserverIntentType.NEUTRAL,
        focus_intensity=0.5,
        coherence_level=0.8,
        temporal_stability=0.9,
        scale_preference=ScaleRegime.MACROSCOPIC,
        purpose_tensor_value=1.0
    )
    
    neutral_scaling = observer_system.compute_observer_scaling(neutral_observer, 1.0)
    print(f"Neutral observer scaling: {neutral_scaling:.6f}")
    
    # Intentional observer
    intentional_observer = ObserverState(
        intent_type=ObserverIntentType.INTENTIONAL,
        focus_intensity=1.0,
        coherence_level=0.95,
        temporal_stability=0.85,
        scale_preference=ScaleRegime.QUANTUM,
        purpose_tensor_value=1.5
    )
    
    intentional_scaling = observer_system.compute_observer_scaling(intentional_observer, 2.0)
    print(f"Intentional observer scaling: {intentional_scaling:.6f}")
    
    # Creative observer
    creative_observer = ObserverState(
        intent_type=ObserverIntentType.CREATIVE,
        focus_intensity=0.8,
        coherence_level=0.9,
        temporal_stability=0.7,
        scale_preference=ScaleRegime.MESOSCOPIC,
        purpose_tensor_value=1.3
    )
    
    creative_scaling = observer_system.compute_observer_scaling(creative_observer, 0.5)
    print(f"Creative observer scaling: {creative_scaling:.6f}")
    
    # Test multi-observer scaling
    print(f"\nTesting multi-observer scaling...")
    multi_scaling = observer_system.compute_multi_observer_scaling(
        [neutral_observer, intentional_observer, creative_observer],
        [0.3, 0.5, 0.2],
        1.5
    )
    print(f"Multi-observer scaling: {multi_scaling:.6f}")
    
    # Test sensitivity analysis
    print(f"\nTesting scaling sensitivity analysis...")
    sensitivity = observer_system.analyze_scaling_sensitivity(
        intentional_observer,
        (0.1, 10.0),
        50
    )
    print(f"Sensitivity index: {sensitivity['sensitivity_index']:.6f}")
    print(f"Scaling range: {sensitivity['scaling_range']:.6f}")
    print(f"Max gradient: {sensitivity['max_gradient']:.6f}")
    
    # System validation
    validation = observer_system.validate_observer_scaling()
    print(f"\nObserver scaling validation:")
    print(f"  Formula implementation: {validation['formula_implementation']}")
    print(f"  Purpose tensor calculation: {validation['purpose_tensor_calculation']}")
    print(f"  Scaling bounds: {validation['scaling_bounds']}")
    print(f"  Multi-observer support: {validation['multi_observer_support']}")
    
    # Get statistics
    stats = observer_system.get_scaling_statistics()
    if 'total_events' in stats:
        print(f"\nScaling statistics:")
        print(f"  Total events: {stats['total_events']}")
        print(f"  Mean scaling factor: {stats['scaling_factors']['mean']:.6f}")
        print(f"  Scaling factor range: {stats['scaling_factors']['min']:.6f} - {stats['scaling_factors']['max']:.6f}")
    
    print("\nObserver Scaling system ready for UBP integration.")
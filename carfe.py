"""
Universal Binary Principle (UBP) Framework v3.2+ - CARFE: Cykloid Adelic Recursive Expansive Field Equation for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the recursive field equation for self-evolving OffBits and
temporal alignment in the UBP framework. CARFE provides the mathematical
foundation for dynamic system evolution and Zitterbewegung modeling.

Mathematical Foundation:
- Recursive field evolution with p-adic structure
- Temporal alignment across multiple scales
- Self-evolving OffBit dynamics
- Zitterbewegung frequency modeling (1.2356×10²⁰ Hz)
- Adelic number theory integration

Reference: Del Bel, J. (2025). The Cykloid Adelic Recursive Expansive Field Equation (CARFE). Academia.edu. https://www.academia.edu/130184561/
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import deque

# Import UBPConfig and get_config for constant loading
from ubp_config import get_config, UBPConfig

_config: UBPConfig = get_config() # Initialize configuration


class CARFEMode(Enum):
    """CARFE operational modes"""
    RECURSIVE = "recursive"           # Standard recursive evolution
    EXPANSIVE = "expansive"          # Expansive field dynamics
    TEMPORAL = "temporal"            # Temporal alignment mode
    ZITTERBEWEGUNG = "zitterbewegung" # High-frequency oscillation mode
    ADELIC = "adelic"                # p-adic number integration
    HYBRID = "hybrid"                # Combined mode operation


class FieldTopology(Enum):
    """Field topology types for CARFE"""
    CYKLOID = "cykloid"              # Cycloid-based topology
    TORUS = "torus"                  # Toroidal topology
    SPHERE = "sphere"                # Spherical topology
    HYPERBOLIC = "hyperbolic"        # Hyperbolic topology
    FRACTAL = "fractal"              # Fractal topology


@dataclass
class CARFEParameters:
    """
    Parameters for CARFE field equation calculations.
    """
    # Core parameters
    recursion_depth: int = 10
    expansion_factor: float = _config.constants.PHI  # Golden ratio φ, uses UBPConfig
    temporal_scale: float = _config.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD_DEFAULT  # 1/π seconds, uses UBPConfig
    zitterbewegung_frequency: float = _config.constants.UBP_ZITTERBEWEGUNG_FREQ  # Hz, uses UBPConfig
    
    # p-adic parameters
    prime_base: int = 2  # Base prime for p-adic calculations
    adelic_precision: int = 10  # Precision for adelic calculations
    
    # Field parameters
    field_strength: float = 1.0
    coupling_constant: float = _config.constants.FINE_STRUCTURE_CONSTANT  # Fine structure constant, uses UBPConfig
    coherence_threshold: float = _config.performance.COHERENCE_THRESHOLD  # OnBit threshold, uses UBPConfig
    
    # Evolution parameters
    evolution_rate: float = 0.95  # Rate of field evolution
    damping_factor: float = 0.98  # Damping for stability
    nonlinearity_strength: float = 0.1  # Nonlinear coupling strength
    
    # Numerical parameters
    time_step: float = 1e-15  # Time step for integration
    convergence_tolerance: float = 1e-12
    max_iterations: int = 1000


@dataclass
class FieldState:
    """
    Represents the state of a CARFE field at a specific time.
    """
    timestamp: float
    field_values: np.ndarray
    momentum: np.ndarray
    energy: float
    coherence: float
    topology: FieldTopology
    recursion_level: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class PAdicCalculator:
    """
    p-adic number calculator for adelic CARFE operations.
    
    Implements p-adic arithmetic and valuations for the adelic
    component of the CARFE field equation.
    """
    
    def __init__(self, prime: int = 2, precision: int = 10):
        self.prime = prime
        self.precision = precision
        self._valuation_cache = {}
    
    def p_adic_valuation(self, n: int) -> int:
        """
        Compute p-adic valuation v_p(n).
        
        Args:
            n: Integer to compute valuation for
        
        Returns:
            p-adic valuation
        """
        if n == 0:
            return float('inf')
        
        if n in self._valuation_cache:
            return self._valuation_cache[n]
        
        valuation = 0
        while n % self.prime == 0:
            n //= self.prime
            valuation += 1
        
        self._valuation_cache[n] = valuation
        return valuation
    
    def p_adic_norm(self, n: int) -> float:
        """
        Compute p-adic norm |n|_p.
        
        Args:
            n: Integer to compute norm for
        
        Returns:
            p-adic norm
        """
        if n == 0:
            return 0.0
        
        valuation = self.p_adic_valuation(n)
        return self.prime ** (-valuation)
    
    def p_adic_distance(self, a: int, b: int) -> float:
        """
        Compute p-adic distance between two integers.
        
        Args:
            a, b: Integers to compute distance between
        
        Returns:
            p-adic distance
        """
        return self.p_adic_norm(a - b)
    
    def adelic_product(self, values: List[float], primes: List[int]) -> float:
        """
        Compute adelic product across multiple primes.
        
        Args:
            values: Values for each prime
            primes: List of primes
        
        Returns:
            Adelic product
        """
        if len(values) != len(primes):
            raise ValueError("Number of values must match number of primes")
        
        product = 1.0
        for value, prime in zip(values, primes):
            # Convert to p-adic representation
            p_adic_val = self.p_adic_norm(int(value * 1000))  # Scale for integer conversion
            product *= p_adic_val
        
        return product


class CykloidGeometry:
    """
    Implements cycloid geometry for CARFE field topology.
    
    Provides geometric calculations for cycloid-based field structures
    used in the recursive expansion dynamics.
    """
    
    def __init__(self, radius: float = 1.0):
        self.radius = radius
        self._curve_cache = {}
    
    def parametric_cycloid(self, t: float) -> Tuple[float, float]:
        """
        Compute parametric cycloid coordinates.
        
        Args:
            t: Parameter value
        
        Returns:
            Tuple of (x, y) coordinates
        """
        x = self.radius * (t - math.sin(t))
        y = self.radius * (1 - math.cos(t))
        return x, y
    
    def cycloid_curvature(self, t: float) -> float:
        """
        Compute curvature of cycloid at parameter t.
        
        Args:
            t: Parameter value
        
        Returns:
            Curvature value
        """
        # Curvature κ = 1/(2R*sin(t/2)) for cycloid
        if abs(math.sin(t/2)) < 1e-10:
            return 0.0
        
        curvature = 1.0 / (2 * self.radius * abs(math.sin(t/2)))
        return curvature
    
    def cycloid_arc_length(self, t1: float, t2: float, num_points: int = 100) -> float:
        """
        Compute arc length of cycloid between parameters t1 and t2.
        
        Args:
            t1, t2: Parameter bounds
            num_points: Number of integration points
        
        Returns:
            Arc length
        """
        t_values = np.linspace(t1, t2, num_points)
        dt = (t2 - t1) / (num_points - 1)
        
        arc_length = 0.0
        for t in t_values[:-1]:
            # ds/dt = R * sqrt(2(1 - cos(t))) for cycloid
            ds_dt = self.radius * math.sqrt(2 * (1 - math.cos(t)))
            arc_length += ds_dt * dt
        
        return arc_length
    
    def generate_cycloid_field(self, t_range: Tuple[float, float], 
                             resolution: int = 100) -> np.ndarray:
        """
        Generate cycloid field values over parameter range.
        
        Args:
            t_range: Parameter range (t_min, t_max)
            resolution: Number of field points
        
        Returns:
            Array of field values
        """
        t_values = np.linspace(t_range[0], t_range[1], resolution)
        field_values = np.zeros(resolution, dtype=complex)
        
        for i, t in enumerate(t_values):
            x, y = self.parametric_cycloid(t)
            curvature = self.cycloid_curvature(t)
            
            # Complex field value incorporating geometry
            field_values[i] = complex(x, y) * curvature
        
        return field_values


class CARFEFieldEquation:
    """
    Main CARFE field equation solver.
    
    Implements the complete Cykloid Adelic Recursive Expansive Field Equation
    for UBP system evolution and temporal alignment.
    """
    
    def __init__(self, parameters: Optional[CARFEParameters] = None):
        self.parameters = parameters or CARFEParameters()
        self.p_adic_calc = PAdicCalculator(
            prime=self.parameters.prime_base,
            precision=self.parameters.adelic_precision
        )
        self.cycloid_geom = CykloidGeometry()
        
        self._field_history = deque(maxlen=1000)
        self._evolution_cache = {}
        
    def compute_recursive_field(self, initial_field: np.ndarray, 
                              recursion_depth: Optional[int] = None) -> np.ndarray:
        """
        Compute recursive field evolution.
        
        Args:
            initial_field: Initial field configuration
            recursion_depth: Depth of recursion (uses parameter default if None)
        
        Returns:
            Evolved field after recursion
        """
        depth = recursion_depth or self.parameters.recursion_depth
        current_field = initial_field.copy()
        
        for level in range(depth):
            # Recursive transformation: F_{n+1} = φ * F_n + nonlinear_term
            linear_term = self.parameters.expansion_factor * current_field
            
            # Nonlinear term with p-adic modulation
            nonlinear_term = self.parameters.nonlinearity_strength * np.sin(current_field)
            
            # p-adic correction
            p_adic_correction = np.zeros_like(current_field)
            for i, val in enumerate(current_field):
                if val != 0:
                    p_adic_norm = self.p_adic_calc.p_adic_norm(int(abs(val) * 1000))
                    p_adic_correction[i] = p_adic_norm * 0.01
            
            # Combine terms
            current_field = linear_term + nonlinear_term + p_adic_correction
            
            # Apply damping for stability
            current_field *= self.parameters.damping_factor
        
        return current_field
    
    def compute_expansive_dynamics(self, field_state: FieldState, 
                                 time_step: Optional[float] = None) -> FieldState:
        """
        Compute expansive field dynamics evolution.
        
        Args:
            field_state: Current field state
            time_step: Time step for evolution
        
        Returns:
            Evolved field state
        """
        dt = time_step or self.parameters.time_step
        
        # Compute field derivatives
        field_gradient = np.gradient(field_state.field_values)
        field_laplacian = np.gradient(field_gradient)
        
        # Expansive dynamics equation:
        # ∂F/∂t = φ * ∇²F + coupling * F * |F|² + zitterbewegung_term
        
        # Linear diffusion term
        diffusion_term = self.parameters.expansion_factor * field_laplacian
        
        # Nonlinear self-interaction
        nonlinear_term = (self.parameters.coupling_constant * 
                         field_state.field_values * 
                         np.abs(field_state.field_values)**2)
        
        # Zitterbewegung oscillation
        zitter_phase = 2 * math.pi * self.parameters.zitterbewegung_frequency * field_state.timestamp
        zitter_term = 0.001 * np.cos(zitter_phase) * field_state.field_values
        
        # Temporal derivative
        dF_dt = diffusion_term + nonlinear_term + zitter_term
        
        # Update field values
        new_field_values = field_state.field_values + dt * dF_dt
        
        # Update momentum (for energy calculation)
        new_momentum = field_state.momentum + dt * field_gradient
        
        # Compute energy
        kinetic_energy = 0.5 * np.sum(new_momentum**2)
        potential_energy = 0.25 * np.sum(new_field_values**4)
        total_energy = kinetic_energy + potential_energy
        
        # Compute coherence
        field_variance = np.var(new_field_values)
        coherence = 1.0 / (1.0 + field_variance) if field_variance > 0 else 1.0
        
        # Create new field state
        new_state = FieldState(
            timestamp=field_state.timestamp + dt,
            field_values=new_field_values,
            momentum=new_momentum,
            energy=total_energy,
            coherence=coherence,
            topology=field_state.topology,
            recursion_level=field_state.recursion_level,
            metadata={
                'evolution_type': 'expansive',
                'time_step': dt,
                'energy_change': total_energy - field_state.energy,
                'coherence_change': coherence - field_state.coherence
            }
        )
        
        return new_state
    
    def compute_temporal_alignment(self, field_states: List[FieldState],
                                 target_frequency: float) -> List[FieldState]:
        """
        Compute temporal alignment across multiple field states.
        
        Args:
            field_states: List of field states to align
            target_frequency: Target frequency for alignment
        
        Returns:
            List of temporally aligned field states
        """
        if not field_states:
            return []
        
        aligned_states = []
        reference_time = field_states[0].timestamp
        
        for i, state in enumerate(field_states):
            # Compute phase alignment
            time_diff = state.timestamp - reference_time
            phase_correction = 2 * math.pi * target_frequency * time_diff
            
            # Apply phase correction to field values
            corrected_field = state.field_values * np.exp(1j * phase_correction)
            
            # Extract real part for aligned field
            aligned_field = np.real(corrected_field)
            
            # Create aligned state
            aligned_state = FieldState(
                timestamp=state.timestamp,
                field_values=aligned_field,
                momentum=state.momentum,
                energy=state.energy,
                coherence=state.coherence,
                topology=state.topology,
                recursion_level=state.recursion_level,
                metadata={
                    'alignment_type': 'temporal',
                    'target_frequency': target_frequency,
                    'phase_correction': phase_correction,
                    'original_coherence': state.coherence
                }
            )
            
            aligned_states.append(aligned_state)
        
        return aligned_states
    
    def compute_zitterbewegung_evolution(self, field_state: FieldState,
                                       duration: float) -> List[FieldState]:
        """
        Compute Zitterbewegung evolution over specified duration.
        
        Args:
            field_state: Initial field state
            duration: Evolution duration
        
        Returns:
            List of field states showing Zitterbewegung evolution
        """
        num_steps = int(duration / self.parameters.time_step)
        evolution_states = [field_state]
        
        current_state = field_state
        
        for step in range(num_steps):
            # Zitterbewegung frequency modulation
            t = current_state.timestamp
            zitter_freq = self.parameters.zitterbewegung_frequency
            
            # High-frequency oscillation
            oscillation = np.cos(2 * math.pi * zitter_freq * t)
            
            # Modulate field with Zitterbewegung
            modulated_field = current_state.field_values * (1.0 + 0.01 * oscillation)
            
            # Apply recursive evolution
            evolved_field = self.compute_recursive_field(modulated_field, recursion_depth=1)
            
            # Create new state
            new_state = FieldState(
                timestamp=t + self.parameters.time_step,
                field_values=evolved_field,
                momentum=current_state.momentum,
                energy=current_state.energy,
                coherence=current_state.coherence,
                topology=current_state.topology,
                recursion_level=current_state.recursion_level + 1,
                metadata={
                    'evolution_type': 'zitterbewegung',
                    'oscillation_amplitude': oscillation,
                    'frequency': zitter_freq
                }
            )
            
            evolution_states.append(new_state)
            current_state = new_state
        
        return evolution_states
    
    def compute_adelic_correction(self, field_values: np.ndarray,
                                primes: List[int] = [2, 3, 5, 7]) -> np.ndarray:
        """
        Compute adelic correction to field values.
        
        Args:
            field_values: Field values to correct
            primes: List of primes for adelic calculation
        
        Returns:
            Adelic-corrected field values
        """
        corrected_field = field_values.copy()
        
        for i, val in enumerate(field_values):
            if val != 0:
                # Compute p-adic norms for each prime
                p_adic_norms = []
                for prime in primes:
                    calc = PAdicCalculator(prime=prime, precision=self.parameters.adelic_precision)
                    norm = calc.p_adic_norm(int(abs(val) * 1000))
                    p_adic_norms.append(norm)
                
                # Compute adelic product
                adelic_product = self.p_adic_calc.adelic_product(p_adic_norms, primes)
                
                # Apply correction
                correction_factor = 1.0 + 0.001 * adelic_product
                corrected_field[i] = val * correction_factor
        
        return corrected_field
    
    def solve_carfe_equation(self, initial_state: FieldState,
                           evolution_time: float,
                           mode: CARFEMode = CARFEMode.HYBRID) -> List[FieldState]:
        """
        Solve the complete CARFE equation over specified time.
        
        Args:
            initial_state: Initial field state
            evolution_time: Total evolution time
            mode: CARFE operational mode
        
        Returns:
            List of field states showing complete evolution
        """
        num_steps = int(evolution_time / self.parameters.time_step)
        evolution_states = [initial_state]
        
        current_state = initial_state
        
        for step in range(num_steps):
            if mode == CARFEMode.RECURSIVE:
                # Pure recursive evolution
                evolved_field = self.compute_recursive_field(current_state.field_values)
                new_state = FieldState(
                    timestamp=current_state.timestamp + self.parameters.time_step,
                    field_values=evolved_field,
                    momentum=current_state.momentum,
                    energy=current_state.energy,
                    coherence=current_state.coherence,
                    topology=current_state.topology,
                    recursion_level=current_state.recursion_level + 1
                )
            
            elif mode == CARFEMode.EXPANSIVE:
                # Expansive dynamics
                new_state = self.compute_expansive_dynamics(current_state)
            
            elif mode == CARFEMode.ADELIC:
                # Adelic correction
                corrected_field = self.compute_adelic_correction(current_state.field_values)
                new_state = FieldState(
                    timestamp=current_state.timestamp + self.parameters.time_step,
                    field_values=corrected_field,
                    momentum=current_state.momentum,
                    energy=current_state.energy,
                    coherence=current_state.coherence,
                    topology=current_state.topology,
                    recursion_level=current_state.recursion_level
                )
            
            elif mode == CARFEMode.HYBRID:
                # Combined evolution
                # 1. Recursive step
                recursive_field = self.compute_recursive_field(current_state.field_values, recursion_depth=1)
                
                # 2. Expansive dynamics
                temp_state = FieldState(
                    timestamp=current_state.timestamp,
                    field_values=recursive_field,
                    momentum=current_state.momentum,
                    energy=current_state.energy,
                    coherence=current_state.coherence,
                    topology=current_state.topology,
                    recursion_level=current_state.recursion_level
                )
                expanded_state = self.compute_expansive_dynamics(temp_state)
                
                # 3. Adelic correction
                final_field = self.compute_adelic_correction(expanded_state.field_values)
                
                new_state = FieldState(
                    timestamp=expanded_state.timestamp,
                    field_values=final_field,
                    momentum=expanded_state.momentum,
                    energy=expanded_state.energy,
                    coherence=expanded_state.coherence,
                    topology=expanded_state.topology,
                    recursion_level=current_state.recursion_level + 1,
                    metadata={'evolution_mode': 'hybrid'}
                )
            
            else:
                raise ValueError(f"Unknown CARFE mode: {mode}")
            
            evolution_states.append(new_state)
            current_state = new_state
            
            # Store in history
            self._field_history.append(new_state)
        
        return evolution_states
    
    def analyze_field_stability(self, evolution_states: List[FieldState]) -> Dict[str, Any]:
        """
        Analyze stability of field evolution.
        
        Args:
            evolution_states: List of field states from evolution
        
        Returns:
            Dictionary containing stability analysis
        """
        if len(evolution_states) < 2:
            return {'stability': 'insufficient_data'}
        
        # Extract time series data
        times = [state.timestamp for state in evolution_states]
        energies = [state.energy for state in evolution_states]
        coherences = [state.coherence for state in evolution_states]
        
        # Compute stability metrics
        energy_variance = np.var(energies)
        coherence_variance = np.var(coherences)
        
        # Compute Lyapunov-like exponent
        field_norms = [np.linalg.norm(state.field_values) for state in evolution_states]
        if len(field_norms) > 1:
            norm_ratios = [field_norms[i+1]/field_norms[i] for i in range(len(field_norms)-1) if field_norms[i] > 0]
            if norm_ratios:
                lyapunov_estimate = np.mean([math.log(abs(ratio)) for ratio in norm_ratios])
            else:
                lyapunov_estimate = 0.0
        else:
            lyapunov_estimate = 0.0
        
        # Stability classification
        if energy_variance < 0.01 and coherence_variance < 0.01 and lyapunov_estimate < 0.1:
            stability_class = "stable"
        elif lyapunov_estimate > 1.0:
            stability_class = "chaotic"
        else:
            stability_class = "transitional"
        
        return {
            'stability_class': stability_class,
            'energy_variance': energy_variance,
            'coherence_variance': coherence_variance,
            'lyapunov_estimate': lyapunov_estimate,
            'mean_energy': np.mean(energies),
            'mean_coherence': np.mean(coherences),
            'evolution_duration': times[-1] - times[0],
            'num_states': len(evolution_states)
        }
    
    def validate_carfe_system(self) -> Dict[str, Any]:
        """
        Validate the CARFE system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'recursive_evolution': True,
            'expansive_dynamics': True,
            'temporal_alignment': True,
            'adelic_correction': True,
            'zitterbewegung_modeling': True
        }
        
        try:
            # Test 1: Recursive evolution
            test_field = np.array([1.0, 0.5, 0.2, 0.1])
            evolved_field = self.compute_recursive_field(test_field, recursion_depth=3)
            
            if not isinstance(evolved_field, np.ndarray) or len(evolved_field) != len(test_field):
                validation_results['recursive_evolution'] = False
                validation_results['recursive_error'] = "Recursive evolution failed"
            
            # Test 2: Expansive dynamics
            test_state = FieldState(
                timestamp=0.0,
                field_values=test_field,
                momentum=np.zeros_like(test_field),
                energy=1.0,
                coherence=0.8,
                topology=FieldTopology.CYKLOID
            )
            
            evolved_state = self.compute_expansive_dynamics(test_state)
            
            if not isinstance(evolved_state, FieldState):
                validation_results['expansive_dynamics'] = False
                validation_results['expansive_error'] = "Expansive dynamics failed"
            
            # Test 3: Temporal alignment
            test_states = [test_state, evolved_state]
            aligned_states = self.compute_temporal_alignment(test_states, 1e9)
            
            if len(aligned_states) != len(test_states):
                validation_results['temporal_alignment'] = False
                validation_results['alignment_error'] = "Temporal alignment failed"
            
            # Test 4: Adelic correction
            corrected_field = self.compute_adelic_correction(test_field)
            
            if not isinstance(corrected_field, np.ndarray) or len(corrected_field) != len(test_field):
                validation_results['adelic_correction'] = False
                validation_results['adelic_error'] = "Adelic correction failed"
            
            # Test 5: Zitterbewegung modeling
            zitter_states = self.compute_zitterbewegung_evolution(test_state, 1e-12)
            
            if len(zitter_states) < 2:
                validation_results['zitterbewegung_modeling'] = False
                validation_results['zitter_error'] = "Zitterbewegung modeling failed"
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['recursive_evolution'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_carfe_system(recursion_depth: int = 10,
                       zitterbewegung_freq: float = _config.constants.UBP_ZITTERBEWEGUNG_FREQ) -> CARFEFieldEquation: # Uses UBPConfig
    """
    Create a CARFE system with specified configuration.
    
    Args:
        recursion_depth: Depth of recursive evolution
        zitterbewegung_freq: Zitterbewegung frequency in Hz
    
    Returns:
        Configured CARFEFieldEquation instance
    """
    parameters = CARFEParameters(
        recursion_depth=recursion_depth,
        zitterbewegung_frequency=zitterbewegung_freq
    )
    return CARFEFieldEquation(parameters)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing CARFE system...")
    
    carfe_system = create_carfe_system()
    
    # Test recursive evolution
    print("\nTesting recursive field evolution...")
    test_field = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
    evolved_field = carfe_system.compute_recursive_field(test_field, recursion_depth=5)
    print(f"Original field: {test_field}")
    print(f"Evolved field: {evolved_field}")
    print(f"Evolution ratio: {np.linalg.norm(evolved_field) / np.linalg.norm(test_field):.6f}")
    
    # Test expansive dynamics
    print(f"\nTesting expansive dynamics...")
    initial_state = FieldState(
        timestamp=0.0,
        field_values=test_field,
        momentum=np.zeros_like(test_field),
        energy=1.0,
        coherence=0.9,
        topology=FieldTopology.CYKLOID
    )
    
    evolved_state = carfe_system.compute_expansive_dynamics(initial_state)
    print(f"Initial energy: {initial_state.energy:.6f}")
    print(f"Evolved energy: {evolved_state.energy:.6f}")
    print(f"Initial coherence: {initial_state.coherence:.6f}")
    print(f"Evolved coherence: {evolved_state.coherence:.6f}")
    
    # Test complete CARFE evolution
    print(f"\nTesting complete CARFE evolution...")
    evolution_states = carfe_system.solve_carfe_equation(
        initial_state, 
        evolution_time=1e-12,  # Very short time for testing
        mode=CARFEMode.HYBRID
    )
    print(f"Evolution steps: {len(evolution_states)}")
    print(f"Final energy: {evolution_states[-1].energy:.6f}")
    print(f"Final coherence: {evolution_states[-1].coherence:.6f}")
    
    # Test stability analysis
    print(f"\nTesting stability analysis...")
    stability = carfe_system.analyze_field_stability(evolution_states)
    print(f"Stability class: {stability['stability_class']}")
    print(f"Energy variance: {stability['energy_variance']:.6f}")
    print(f"Lyapunov estimate: {stability['lyapunov_estimate']:.6f}")
    
    # Test p-adic calculations
    print(f"\nTesting p-adic calculations...")
    p_adic_calc = carfe_system.p_adic_calc
    test_val = 24
    valuation = p_adic_calc.p_adic_valuation(test_val)
    norm = p_adic_calc.p_adic_norm(test_val)
    print(f"2-adic valuation of {test_val}: {valuation}")
    print(f"2-adic norm of {test_val}: {norm:.6f}")
    
    # System validation
    validation = carfe_system.validate_carfe_system()
    print(f"\nCARFE system validation:")
    print(f"  Recursive evolution: {validation['recursive_evolution']}")
    print(f"  Expansive dynamics: {validation['expansive_dynamics']}")
    print(f"  Temporal alignment: {validation['temporal_alignment']}")
    print(f"  Adelic correction: {validation['adelic_correction']}")
    print(f"  Zitterbewegung modeling: {validation['zitterbewegung_modeling']}")
    
    print("\nCARFE system ready for UBP integration.")
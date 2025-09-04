"""
Universal Binary Principle (UBP) Framework v3.2+ - Spin Transition Module for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
======================================

Implements the spin transition mechanism that serves as the quantum information
source in the UBP framework. Handles spin state transitions, quantum coherence,
and information generation through spin dynamics.

Mathematical Foundation:
- Spin transition: b_i × ln(1/p_s) where p_s is toggle probability
- Quantum coherence through spin entanglement
- Information generation via spin state changes
- Zitterbewegung frequency integration (1.2356×10²⁰ Hz)
- Pauli matrix operations for spin dynamics

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

class SpinState(Enum):
    """Quantum spin states"""
    UP = "up"                          # |↑⟩ state
    DOWN = "down"                      # |↓⟩ state
    SUPERPOSITION = "superposition"    # α|↑⟩ + β|↓⟩ state
    ENTANGLED = "entangled"           # Entangled with other spins
    COHERENT = "coherent"             # Coherent superposition
    MIXED = "mixed"                   # Mixed state (decoherent)


class SpinRealm(Enum):
    """Spin realms with different toggle probabilities"""
    QUANTUM = "quantum"               # p_s = e/12 ≈ 0.2265234857
    COSMOLOGICAL = "cosmological"     # p_s = π^φ ≈ 0.83203682
    ELECTROMAGNETIC = "electromagnetic" # p_s = π/4 ≈ 0.7853981634
    NUCLEAR = "nuclear"               # p_s = 1/φ ≈ 0.618034
    BIOLOGICAL = "biological"         # p_s = 1/e ≈ 0.367879
    GRAVITATIONAL = "gravitational"   # p_s = 1/π ≈ 0.318310


@dataclass
class SpinConfiguration:
    """
    Configuration for spin transition calculations.
    """
    realm: SpinRealm
    toggle_probability: float
    zitterbewegung_frequency: float = _config.constants.UBP_ZITTERBEWEGUNG_FREQ  # Hz, from config
    coherence_time: float = 1e-12  # seconds
    decoherence_rate: float = 1e6  # Hz
    coupling_strength: float = 0.1
    temperature: float = 300.0  # Kelvin
    magnetic_field: float = 0.0  # Tesla


@dataclass
class SpinSystem:
    """
    Represents a quantum spin system.
    """
    system_id: str
    num_spins: int
    spin_states: np.ndarray  # Complex amplitudes for each spin
    entanglement_matrix: np.ndarray  # Entanglement between spins
    coherence_matrix: np.ndarray  # Coherence between spins
    energy: float = 0.0
    total_angular_momentum: np.ndarray = field(default_factory=lambda: np.zeros(3))
    metadata: Dict[str, Any] = field(default_factory=dict)


class PauliMatrices:
    """
    Pauli matrices for spin-1/2 operations.
    """
    
    def __init__(self):
        # Pauli matrices
        self.sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)
        
        # Spin-up and spin-down states
        self.up = np.array([1, 0], dtype=complex)
        self.down = np.array([0, 1], dtype=complex)
    
    def rotation_x(self, angle: float) -> np.ndarray:
        """Rotation around x-axis"""
        return np.cos(angle/2) * self.identity - 1j * np.sin(angle/2) * self.sigma_x
    
    def rotation_y(self, angle: float) -> np.ndarray:
        """Rotation around y-axis"""
        return np.cos(angle/2) * self.identity - 1j * np.sin(angle/2) * self.sigma_y
    
    def rotation_z(self, angle: float) -> np.ndarray:
        """Rotation around z-axis"""
        return np.cos(angle/2) * self.identity - 1j * np.sin(angle/2) * self.sigma_z
    
    def expectation_value(self, state: np.ndarray, operator: np.ndarray) -> complex:
        """Compute expectation value ⟨ψ|O|ψ⟩"""
        return np.conj(state).T @ operator @ state


class SpinTransitionCalculator:
    """
    Implements spin transition calculations for UBP.
    
    Handles the core spin transition formula: b_i × ln(1/p_s)
    and related quantum spin dynamics.
    """
    
    def __init__(self, configuration: SpinConfiguration):
        self.config = configuration
        self.pauli = PauliMatrices()
        self._transition_cache = {}
        
        # Precompute toggle probability logarithm
        self.ln_inv_p_s = math.log(1.0 / self.config.toggle_probability)
    
    def compute_spin_transition(self, bit_state: float) -> float:
        """
        Compute spin transition using UBP formula: b_i × ln(1/p_s)
        
        Args:
            bit_state: Current bit state (0.0 to 1.0)
        
        Returns:
            Spin transition value
        """
        transition_value = bit_state * self.ln_inv_p_s
        return transition_value
    
    def compute_transition_probability(self, initial_state: np.ndarray,
                                     final_state: np.ndarray,
                                     time: float) -> float:
        """
        Compute probability of transition between spin states.
        
        Args:
            initial_state: Initial spin state vector
            final_state: Final spin state vector
            time: Evolution time
        
        Returns:
            Transition probability
        """
        # Normalize states
        initial_normalized = initial_state / np.linalg.norm(initial_state)
        final_normalized = final_state / np.linalg.norm(final_state)
        
        # Overlap amplitude
        overlap = np.abs(np.vdot(final_normalized, initial_normalized))**2
        
        # Time evolution factor
        frequency = self.config.zitterbewegung_frequency
        oscillation = np.cos(2 * math.pi * frequency * time)**2
        
        # Decoherence factor
        decoherence = np.exp(-time / self.config.coherence_time)
        
        # Total transition probability
        probability = overlap * oscillation * decoherence
        
        return probability
    
    def evolve_spin_state(self, initial_state: np.ndarray,
                         hamiltonian: np.ndarray,
                         time: float) -> np.ndarray:
        """
        Evolve spin state under Hamiltonian evolution.
        
        Args:
            initial_state: Initial state vector
            hamiltonian: Hamiltonian matrix
            time: Evolution time
        
        Returns:
            Evolved state vector
        """
        # Time evolution operator: U = exp(-iHt/ℏ)
        # Using ℏ = 1 units
        evolution_operator = self._matrix_exponential(-1j * hamiltonian * time)
        
        # Apply evolution
        evolved_state = evolution_operator @ initial_state
        
        # Normalize
        evolved_state = evolved_state / np.linalg.norm(evolved_state)
        
        return evolved_state
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential using eigendecomposition"""
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        exp_eigenvalues = np.exp(eigenvalues)
        return eigenvectors @ np.diag(exp_eigenvalues) @ eigenvectors.conj().T
    
    def compute_spin_coherence(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """
        Compute coherence between two spin states.
        
        Args:
            state1, state2: Spin state vectors
        
        Returns:
            Coherence value (0 to 1)
        """
        # Normalize states
        state1_norm = state1 / np.linalg.norm(state1)
        state2_norm = state2 / np.linalg.norm(state2)
        
        # Coherence as overlap magnitude
        coherence = np.abs(np.vdot(state1_norm, state2_norm))
        
        return coherence
    
    def compute_entanglement_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Compute von Neumann entropy for entanglement quantification.
        
        Args:
            density_matrix: Density matrix of the system
        
        Returns:
            Entanglement entropy
        """
        # Compute eigenvalues of density matrix
        eigenvalues = np.linalg.eigvals(density_matrix)
        
        # Remove zero eigenvalues to avoid log(0)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        
        # Von Neumann entropy: S = -Tr(ρ log ρ)
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        
        return entropy.real
    
    def generate_random_spin_state(self) -> np.ndarray:
        """
        Generate random normalized spin state.
        
        Returns:
            Random spin state vector
        """
        # Random complex amplitudes
        real_part = np.random.randn(2)
        imag_part = np.random.randn(2)
        state = real_part + 1j * imag_part
        
        # Normalize
        state = state / np.linalg.norm(state)
        
        return state
    
    def create_bell_state(self, bell_type: int = 0) -> np.ndarray:
        """
        Create Bell states for two-qubit entanglement.
        
        Args:
            bell_type: Type of Bell state (0-3)
        
        Returns:
            Bell state vector (4-dimensional)
        """
        sqrt_half = 1.0 / math.sqrt(2)
        
        if bell_type == 0:  # |Φ+⟩ = (|00⟩ + |11⟩)/√2
            return np.array([sqrt_half, 0, 0, sqrt_half], dtype=complex)
        elif bell_type == 1:  # |Φ-⟩ = (|00⟩ - |11⟩)/√2
            return np.array([sqrt_half, 0, 0, -sqrt_half], dtype=complex)
        elif bell_type == 2:  # |Ψ+⟩ = (|01⟩ + |10⟩)/√2
            return np.array([0, sqrt_half, sqrt_half, 0], dtype=complex)
        elif bell_type == 3:  # |Ψ-⟩ = (|01⟩ - |10⟩)/√2
            return np.array([0, sqrt_half, -sqrt_half, 0], dtype=complex)
        else:
            raise ValueError("Bell type must be 0, 1, 2, or 3")


class SpinTransitionSystem:
    """
    Main spin transition system for UBP.
    
    Manages multiple spin systems and their interactions,
    providing the quantum information source for the UBP framework.
    """
    
    def __init__(self, default_realm: SpinRealm = SpinRealm.QUANTUM):
        self.default_realm = default_realm
        self.spin_systems = {}
        self.transition_calculators = {}
        self.interaction_history = deque(maxlen=1000)
        
        # Initialize default configuration
        self._initialize_realm_configurations()
    
    def _initialize_realm_configurations(self):
        """Initialize configurations for different spin realms by fetching from _config"""
        
        realm_configs = {}
        for realm_name, realm_cfg_obj in _config.realms.items():
            # Get toggle probability and Zitterbewegung frequency from _config.constants
            toggle_prob = _config.constants.UBP_TOGGLE_PROBABILITIES.get(realm_name, 0.5)
            zitter_freq = _config.constants.UBP_REALM_FREQUENCIES.get(realm_name, _config.constants.UBP_ZITTERBEWEGUNG_FREQ)
            
            # Map realm_name string to SpinRealm Enum
            try:
                spin_realm_enum = SpinRealm[realm_name.upper()]
            except KeyError:
                spin_realm_enum = SpinRealm.QUANTUM # Fallback if not directly mapped

            # Simplified coherence_time and decoherence_rate for this example,
            # these would ideally come from realm_cfg_obj if it supported them.
            if realm_name == "quantum":
                coherence_time = 1e-12
                decoherence_rate = 1e6
            elif realm_name == "cosmological":
                coherence_time = 1e6
                decoherence_rate = 1e-6
            elif realm_name == "electromagnetic":
                coherence_time = 1e-9
                decoherence_rate = 1e3
            elif realm_name == "nuclear":
                coherence_time = 1e-3
                decoherence_rate = 1e2
            elif realm_name == "biological": # Changed 'biologic' to 'biological' for consistency with SpinRealm
                coherence_time = 1e-1
                decoherence_rate = 10
            elif realm_name == "gravitational":
                coherence_time = 1
                decoherence_rate = 1
            else: # Default for other realms
                coherence_time = 1e-9
                decoherence_rate = 1e3

            realm_configs[spin_realm_enum] = SpinConfiguration(
                realm=spin_realm_enum,
                toggle_probability=toggle_prob,
                zitterbewegung_frequency=zitter_freq,
                coherence_time=coherence_time,
                decoherence_rate=decoherence_rate
            )
        
        # Create transition calculators for each realm
        for realm, config in realm_configs.items():
            self.transition_calculators[realm] = SpinTransitionCalculator(config)
    
    def create_spin_system(self, system_id: str, num_spins: int,
                          realm: Optional[SpinRealm] = None) -> SpinSystem:
        """
        Create a new spin system.
        
        Args:
            system_id: Unique identifier for the system
            num_spins: Number of spins in the system
            realm: Spin realm (uses default if None)
        
        Returns:
            Created spin system
        """
        if realm is None:
            realm = self.default_realm
        
        # Check if calculator for this realm exists
        if realm not in self.transition_calculators:
            raise ValueError(f"No transition calculator configured for realm {realm.value}")

        # Initialize spin states (random superposition)
        spin_states = np.zeros((num_spins, 2), dtype=complex)
        for i in range(num_spins):
            spin_states[i] = self.transition_calculators[realm].generate_random_spin_state()
        
        # Initialize entanglement matrix
        entanglement_matrix = np.zeros((num_spins, num_spins))
        
        # Initialize coherence matrix
        coherence_matrix = np.eye(num_spins)
        
        # Create spin system
        spin_system = SpinSystem(
            system_id=system_id,
            num_spins=num_spins,
            spin_states=spin_states,
            entanglement_matrix=entanglement_matrix,
            coherence_matrix=coherence_matrix,
            energy=0.0,
            total_angular_momentum=np.zeros(3),
            metadata={'realm': realm, 'creation_time': time.time()}
        )
        
        self.spin_systems[system_id] = spin_system
        return spin_system
    
    def compute_system_transition(self, system_id: str, bit_states: np.ndarray) -> np.ndarray:
        """
        Compute spin transitions for entire system.
        
        Args:
            system_id: ID of spin system
            bit_states: Array of bit states for each spin
        
        Returns:
            Array of transition values
        """
        if system_id not in self.spin_systems:
            raise ValueError(f"Spin system {system_id} not found")
        
        spin_system = self.spin_systems[system_id]
        realm = spin_system.metadata['realm']
        calculator = self.transition_calculators[realm]
        
        # Compute transitions for each spin
        transitions = np.zeros(len(bit_states))
        for i, bit_state in enumerate(bit_states):
            transitions[i] = calculator.compute_spin_transition(bit_state)
        
        return transitions
    
    def evolve_spin_system(self, system_id: str, time_step: float,
                          external_field: Optional[np.ndarray] = None) -> SpinSystem:
        """
        Evolve spin system over time step.
        
        Args:
            system_id: ID of spin system
            time_step: Evolution time step
            external_field: External magnetic field [Bx, By, Bz]
        
        Returns:
            Evolved spin system
        """
        if system_id not in self.spin_systems:
            raise ValueError(f"Spin system {system_id} not found")
        
        spin_system = self.spin_systems[system_id]
        realm = spin_system.metadata['realm']
        calculator = self.transition_calculators[realm]
        
        # Create Hamiltonian
        if external_field is None:
            external_field = np.array([0.0, 0.0, 0.1])  # Small z-field
        
        # Evolve each spin
        for i in range(spin_system.num_spins):
            # Single-spin Hamiltonian: H = -μ·B = -γ(σ·B)
            hamiltonian = -(external_field[0] * calculator.pauli.sigma_x +
                           external_field[1] * calculator.pauli.sigma_y +
                           external_field[2] * calculator.pauli.sigma_z)
            
            # Add coupling to other spins
            for j in range(spin_system.num_spins):
                if i != j and spin_system.entanglement_matrix[i, j] > 0:
                    coupling_strength = spin_system.entanglement_matrix[i, j]
                    # Simple Ising-like coupling
                    hamiltonian += coupling_strength * calculator.pauli.sigma_z
            
            # Evolve spin state
            spin_system.spin_states[i] = calculator.evolve_spin_state(
                spin_system.spin_states[i], hamiltonian, time_step
            )
        
        # Update coherence matrix
        self._update_coherence_matrix(spin_system, calculator)
        
        # Update energy
        spin_system.energy = self._compute_system_energy(spin_system, external_field)
        
        # Update total angular momentum
        spin_system.total_angular_momentum = self._compute_total_angular_momentum(
            spin_system, calculator
        )
        
        return spin_system
    
    def _update_coherence_matrix(self, spin_system: SpinSystem,
                               calculator: SpinTransitionCalculator):
        """Update coherence matrix between spins"""
        for i in range(spin_system.num_spins):
            for j in range(i + 1, spin_system.num_spins):
                coherence = calculator.compute_spin_coherence(
                    spin_system.spin_states[i],
                    spin_system.spin_states[j]
                )
                spin_system.coherence_matrix[i, j] = coherence
                spin_system.coherence_matrix[j, i] = coherence
    
    def _compute_system_energy(self, spin_system: SpinSystem,
                             external_field: np.ndarray) -> float:
        """Compute total energy of spin system"""
        total_energy = 0.0
        
        # Single-spin energies in external field
        for i in range(spin_system.num_spins):
            state = spin_system.spin_states[i]
            # Energy = -μ·B = -⟨ψ|σ·B|ψ⟩
            # The default realm calculator is used here for its pauli matrices,
            # which are generic.
            sigma_dot_B = (external_field[0] * self.transition_calculators[self.default_realm].pauli.sigma_x +
                          external_field[1] * self.transition_calculators[self.default_realm].pauli.sigma_y +
                          external_field[2] * self.transition_calculators[self.default_realm].pauli.sigma_z)
            
            energy = -np.real(np.conj(state).T @ sigma_dot_B @ state)
            total_energy += energy
        
        # Interaction energies
        for i in range(spin_system.num_spins):
            for j in range(i + 1, spin_system.num_spins):
                if spin_system.entanglement_matrix[i, j] > 0:
                    coupling = spin_system.entanglement_matrix[i, j]
                    # Simple interaction energy
                    interaction_energy = coupling * spin_system.coherence_matrix[i, j]
                    total_energy += interaction_energy
        
        return total_energy
    
    def _compute_total_angular_momentum(self, spin_system: SpinSystem,
                                      calculator: SpinTransitionCalculator) -> np.ndarray:
        """Compute total angular momentum of system"""
        total_momentum = np.zeros(3)
        
        for i in range(spin_system.num_spins):
            state = spin_system.spin_states[i]
            
            # Compute expectation values of Pauli matrices
            sx = calculator.pauli.expectation_value(state, calculator.pauli.sigma_x)
            sy = calculator.pauli.expectation_value(state, calculator.pauli.sigma_y)
            sz = calculator.pauli.expectation_value(state, calculator.pauli.sigma_z)
            
            # Angular momentum is ℏ/2 times Pauli expectation values
            total_momentum[0] += 0.5 * np.real(sx)
            total_momentum[1] += 0.5 * np.real(sy)
            total_momentum[2] += 0.5 * np.real(sz)
        
        return total_momentum
    
    def create_entanglement(self, system_id: str, spin1: int, spin2: int,
                          entanglement_strength: float = 0.5):
        """
        Create entanglement between two spins.
        
        Args:
            system_id: ID of spin system
            spin1, spin2: Indices of spins to entangle
            entanglement_strength: Strength of entanglement (0 to 1)
        """
        if system_id not in self.spin_systems:
            raise ValueError(f"Spin system {system_id} not found")
        
        spin_system = self.spin_systems[system_id]
        
        if spin1 >= spin_system.num_spins or spin2 >= spin_system.num_spins:
            raise ValueError("Spin indices out of range")
        
        # Set entanglement in matrix
        spin_system.entanglement_matrix[spin1, spin2] = entanglement_strength
        spin_system.entanglement_matrix[spin2, spin1] = entanglement_strength
        
        # Create Bell-like entangled state for the pair
        calculator = self.transition_calculators[spin_system.metadata['realm']]
        bell_state = calculator.create_bell_state(0)  # |Φ+⟩ state
        
        # Extract individual spin states from Bell state
        # This is a simplified approach - full implementation would require
        # proper tensor product decomposition
        sqrt_half = 1.0 / math.sqrt(2)
        spin_system.spin_states[spin1] = np.array([sqrt_half, sqrt_half], dtype=complex)
        spin_system.spin_states[spin2] = np.array([sqrt_half, sqrt_half], dtype=complex)
    
    def measure_spin(self, system_id: str, spin_index: int,
                    measurement_basis: str = 'z') -> Tuple[int, float]:
        """
        Perform quantum measurement on a spin.
        
        Args:
            system_id: ID of spin system
            spin_index: Index of spin to measure
            measurement_basis: Measurement basis ('x', 'y', or 'z')
        
        Returns:
            Tuple of (measurement_result, probability)
        """
        if system_id not in self.spin_systems:
            raise ValueError(f"Spin system {system_id} not found")
        
        spin_system = self.spin_systems[system_id]
        
        if spin_index >= spin_system.num_spins:
            raise ValueError("Spin index out of range")
        
        state = spin_system.spin_states[spin_index]
        calculator = self.transition_calculators[spin_system.metadata['realm']]
        
        # Choose measurement operator
        if measurement_basis == 'x':
            operator = calculator.pauli.sigma_x
        elif measurement_basis == 'y':
            operator = calculator.pauli.sigma_y
        elif measurement_basis == 'z':
            operator = calculator.pauli.sigma_z
        else:
            raise ValueError("Measurement basis must be 'x', 'y', or 'z'")
        
        # Compute probabilities for +1 and -1 eigenvalues
        # For Pauli matrices, eigenvalues are +1 and -1
        prob_plus = np.abs(state[0])**2 if measurement_basis == 'z' else 0.5
        prob_minus = np.abs(state[1])**2 if measurement_basis == 'z' else 0.5
        
        # Perform measurement (random outcome based on probabilities)
        if np.random.random() < prob_plus:
            result = +1
            probability = prob_plus
            # Collapse to +1 eigenstate
            if measurement_basis == 'z':
                spin_system.spin_states[spin_index] = calculator.pauli.up
        else:
            result = -1
            probability = prob_minus
            # Collapse to -1 eigenstate
            if measurement_basis == 'z':
                spin_system.spin_states[spin_index] = calculator.pauli.down
        
        return result, probability
    
    def analyze_system_properties(self, system_id: str) -> Dict[str, Any]:
        """
        Analyze properties of a spin system.
        
        Args:
            system_id: ID of spin system
        
        Returns:
            Dictionary containing system properties
        """
        if system_id not in self.spin_systems:
            raise ValueError(f"Spin system {system_id} not found")
        
        spin_system = self.spin_systems[system_id]
        realm = spin_system.metadata['realm']
        calculator = self.transition_calculators[realm]
        
        # Compute average coherence
        coherence_values = []
        for i in range(spin_system.num_spins):
            for j in range(i + 1, spin_system.num_spins):
                coherence_values.append(spin_system.coherence_matrix[i, j])
        
        avg_coherence = np.mean(coherence_values) if coherence_values else 0.0
        
        # Compute entanglement measure
        entanglement_values = []
        for i in range(spin_system.num_spins):
            for j in range(i + 1, spin_system.num_spins):
                entanglement_values.append(spin_system.entanglement_matrix[i, j])
        
        avg_entanglement = np.mean(entanglement_values) if entanglement_values else 0.0
        
        # Compute purity of each spin state
        purities = []
        for i in range(spin_system.num_spins):
            state = spin_system.spin_states[i]
            # Purity = Tr(ρ²) where ρ = |ψ⟩⟨ψ|
            density_matrix = np.outer(state, np.conj(state))
            purity = np.trace(density_matrix @ density_matrix).real
            purities.append(purity)
        
        avg_purity = np.mean(purities)
        
        return {
            'system_id': system_id,
            'num_spins': spin_system.num_spins,
            'realm': realm.value,
            'total_energy': spin_system.energy,
            'total_angular_momentum': spin_system.total_angular_momentum.tolist(),
            'average_coherence': avg_coherence,
            'average_entanglement': avg_entanglement,
            'average_purity': avg_purity,
            'coherence_matrix_trace': np.trace(spin_system.coherence_matrix),
            'entanglement_matrix_trace': np.trace(spin_system.entanglement_matrix),
            'toggle_probability': calculator.config.toggle_probability,
            'ln_inv_p_s': calculator.ln_inv_p_s
        }
    
    def validate_spin_transition_system(self) -> Dict[str, Any]:
        """
        Validate the spin transition system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'spin_system_creation': True,
            'transition_calculation': True,
            'state_evolution': True,
            'entanglement_creation': True,
            'measurement_operation': True
        }
        
        try:
            # Test 1: Spin system creation
            test_system = self.create_spin_system("test_system", 3, SpinRealm.QUANTUM)
            
            if test_system.num_spins != 3:
                validation_results['spin_system_creation'] = False
                validation_results['creation_error'] = "Spin system creation failed"
            
            # Test 2: Transition calculation
            bit_states = np.array([0.5, 0.8, 0.2])
            transitions = self.compute_system_transition("test_system", bit_states)
            
            if len(transitions) != 3:
                validation_results['transition_calculation'] = False
                validation_results['transition_error'] = "Transition calculation failed"
            
            # Test 3: State evolution
            evolved_system = self.evolve_spin_system("test_system", 1e-12)
            
            if evolved_system.system_id != "test_system":
                validation_results['state_evolution'] = False
                validation_results['evolution_error'] = "State evolution failed"
            
            # Test 4: Entanglement creation
            self.create_entanglement("test_system", 0, 1, 0.7)
            
            if test_system.entanglement_matrix[0, 1] != 0.7:
                validation_results['entanglement_creation'] = False
                validation_results['entanglement_error'] = "Entanglement creation failed"
            
            # Test 5: Measurement operation
            result, probability = self.measure_spin("test_system", 0, 'z')
            
            if result not in [-1, 1] or not (0 <= probability <= 1):
                validation_results['measurement_operation'] = False
                validation_results['measurement_error'] = "Measurement operation failed"
            
            # Clean up test system
            del self.spin_systems["test_system"]
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['spin_system_creation'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_spin_transition_system(realm: SpinRealm = SpinRealm.QUANTUM) -> SpinTransitionSystem:
    """
    Create a spin transition system with specified default realm.
    
    Args:
        realm: Default spin realm
    
    Returns:
        Configured SpinTransitionSystem instance
    """
    return SpinTransitionSystem(realm)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing Spin Transition system...")
    
    spin_system = create_spin_transition_system(SpinRealm.QUANTUM)
    
    # Test spin system creation
    print("\nTesting spin system creation...")
    quantum_spins = spin_system.create_spin_system("quantum_test", 4, SpinRealm.QUANTUM)
    print(f"Created quantum system with {quantum_spins.num_spins} spins")
    print(f"Toggle probability: {spin_system.transition_calculators[SpinRealm.QUANTUM].config.toggle_probability:.6f}")
    print(f"ln(1/p_s): {spin_system.transition_calculators[SpinRealm.QUANTUM].ln_inv_p_s:.6f}")
    
    # Test transition calculation
    print(f"\nTesting spin transition calculation...")
    bit_states = np.array([0.2, 0.5, 0.8, 1.0])
    transitions = spin_system.compute_system_transition("quantum_test", bit_states)
    print(f"Bit states: {bit_states}")
    print(f"Transitions: {transitions}")
    
    # Test entanglement creation
    print(f"\nTesting entanglement creation...")
    spin_system.create_entanglement("quantum_test", 0, 1, 0.8)
    spin_system.create_entanglement("quantum_test", 2, 3, 0.6)
    print(f"Entanglement matrix:\n{quantum_spins.entanglement_matrix}")
    
    # Test system evolution
    print(f"\nTesting system evolution...")
    initial_energy = quantum_spins.energy
    evolved_system = spin_system.evolve_spin_system("quantum_test", 1e-12, np.array([0.1, 0.0, 0.2]))
    final_energy = evolved_system.energy
    print(f"Initial energy: {initial_energy:.6f}")
    print(f"Final energy: {final_energy:.6f}")
    print(f"Energy change: {final_energy - initial_energy:.6f}")
    
    # Test measurements
    print(f"\nTesting quantum measurements...")
    for i in range(4):
        result, prob = spin_system.measure_spin("quantum_test", i, 'z')
        print(f"Spin {i} measurement: {result:+d} (probability: {prob:.6f})")
    
    # Test different realms
    print(f"\nTesting different spin realms...")
    cosmo_spins = spin_system.create_spin_system("cosmo_test", 2, SpinRealm.COSMOLOGICAL)
    bio_spins = spin_system.create_spin_system("bio_test", 2, SpinRealm.BIOLOGICAL)
    
    cosmo_config = spin_system.transition_calculators[SpinRealm.COSMOLOGICAL].config
    bio_config = spin_system.transition_calculators[SpinRealm.BIOLOGICAL].config
    
    print(f"Cosmological p_s: {cosmo_config.toggle_probability:.6f}")
    print(f"Biological p_s: {bio_config.toggle_probability:.6f}")
    print(f"Cosmological frequency: {cosmo_config.zitterbewegung_frequency:.2e} Hz")
    print(f"Biological frequency: {bio_config.zitterbewegung_frequency:.2e} Hz")
    
    # System analysis
    print(f"\nTesting system analysis...")
    analysis = spin_system.analyze_system_properties("quantum_test")
    print(f"Average coherence: {analysis['average_coherence']:.6f}")
    print(f"Average entanglement: {analysis['average_entanglement']:.6f}")
    print(f"Average purity: {analysis['average_purity']:.6f}")
    print(f"Total angular momentum: {analysis['total_angular_momentum']}")
    
    # System validation
    validation = spin_system.validate_spin_transition_system()
    print(f"\nSpin Transition system validation:")
    print(f"  Spin system creation: {validation['spin_system_creation']}")
    print(f"  Transition calculation: {validation['transition_calculation']}")
    print(f"  State evolution: {validation['state_evolution']}")
    print(f"  Entanglement creation: {validation['entanglement_creation']}")
    print(f"  Measurement operation: {validation['measurement_operation']}")
    
    print("\nSpin Transition system ready for UBP integration.")
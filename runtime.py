"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Virtual Machine Runtime
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

The runtime orchestrates UBP semantic functions and manages system state.
It provides a high-level interface for executing UBP operations and simulations.
"""

import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core UBP semantics functions (assuming they are already refactored
# to use a central config or will be soon).
# Temporarily re-importing from specific modules where needed to avoid circular dependencies
# if ubp_semantics.__init__.py is currently problematic.
# The eventual goal is to import from ubp_semantics directly.

from state import OffBit, MutableBitfield, UBPState
from energy import energy, resonance_strength, structural_optimality, observer_effect_factor, cosmic_constant, spin_information_factor, quantum_spin_entropy, cosmological_spin_entropy, weighted_toggle_matrix_sum, calculate_energy_for_realm
from metrics import nrci, coherence_pressure_spatial, fractal_dimension, calculate_system_coherence_score
from toggle_ops import toggle_and, toggle_xor, toggle_or, resonance_toggle, entanglement_toggle, superposition_toggle, hybrid_xor_resonance, spin_transition, apply_tgic_constraint

# Import the centralized UBPConfig
from ubp_config import get_config, UBPConfig, RealmConfig
from global_coherence import GlobalCoherenceIndex
from hex_dictionary import HexDictionary # Import HexDictionary for persistent storage


# Initialize global config and other systems
_config: UBPConfig = get_config()
_global_coherence_system = GlobalCoherenceIndex()


@dataclass
class SimulationState:
    """Represents the current state of a UBP simulation."""
    time_step: int = 0
    global_time: float = 0.0
    active_realm: str = "quantum"
    energy_value: float = 0.0
    nrci_value: float = 0.0
    coherence_pressure: float = 0.0
    total_toggles: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class SimulationResult:
    """Results from a UBP simulation run."""
    initial_state: SimulationState
    final_state: SimulationState
    metrics: Dict[str, float]
    timeline: List[SimulationState]
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'initial_state': self.initial_state.to_dict(),
            'final_state': self.final_state.to_dict(),
            'metrics': self.metrics,
            'timeline': [state.to_dict() for state in self.timeline],
            'execution_time': self.execution_time
        }


class Runtime:
    """
    UBP Virtual Machine Runtime
    
    Manages the execution environment for UBP simulations and operations.
    """
    
    def __init__(self, hardware_profile: str = "desktop_8gb"):
        """
        Initialize the UBP runtime.
        
        Args:
            hardware_profile: Hardware configuration to use
        """
        self.hardware_profile = hardware_profile
        # No longer load_constants(), use _config directly or its sub-components
        self.config = _config # Reference the global config instance
        
        # Initialize Bitfield
        # Bitfield dimensions now come from config.BITFIELD_DIMENSIONS
        # The Bitfield class itself needs to be updated to accept dimensions not just a single size
        # Assuming Bitfield constructor can take dimensions as a tuple, defaulting to a flat array.
        # This will need to be addressed more robustly if Bitfield expects a flat size.
        bitfield_dimensions = self.config.BITFIELD_DIMENSIONS
        total_size = 1
        for dim in bitfield_dimensions:
            total_size *= dim
        
        self.bitfield = MutableBitfield(size=total_size) # Defaulting to total_size
        
        # Runtime state
        self.state = SimulationState()
        self.timeline: List[SimulationState] = []
        
        # Performance tracking
        self.operation_count = 0
        self.start_time = 0.0
        
        # Realm configurations are now directly accessed from self.config.realms
        self._load_realm_configs() # This will populate self.config.realms, not self.realm_configs directly

        # Initialize HexDictionary for persistent knowledge base
        self.hex_dict = HexDictionary()
        # Load any existing persistent UBP knowledge - REMOVED DIRECT LOADING OF PERIODIC TABLE RESULTS
        # AS THAT IS HANDLED BY THE PERIODIC TABLE TEST ITSELF.
        # self._load_persistent_knowledge()

    def _load_realm_configs(self):
        """No longer needed as realms are loaded at UBPConfig initialization."""
        # This method is effectively deprecated/empty as realms are managed by UBPConfig itself.
        pass
    
    # Commented out the _load_persistent_knowledge method to prevent duplicate storage
    # of the periodic table results. The Periodic Table Test script now handles its own
    # storage of individual elements and overall results.
    # def _load_persistent_knowledge(self):
    #     """
    #     Loads and integrates existing UBP knowledge from persistent_state
    #     into the HexDictionary.
    #     """
    #     persistent_file_path = '/persistent_state/ubp_complete_periodic_table_results_20250822_093936.json'
        
    #     if os.path.exists(persistent_file_path):
    #         print(f"Runtime: Found persistent knowledge file: {persistent_file_path}")
    #         try:
    #             with open(persistent_file_path, 'r', encoding='utf-8') as f:
    #                 knowledge_data = json.load(f)
                
    #             # Create a hash of the file content to check if already stored
    #             file_hash = HexDictionary()._serialize_data(knowledge_data, 'json') # Use HexDictionary's internal serialize for consistent hashing
    #             content_hash = HexDictionary().store(knowledge_data, 'json', metadata={'data_type': 'persistent_ubp_periodic_table_results', 'source_file': persistent_file_path})

    #             # Check if already stored based on content hash
    #             # if self.hex_dict.retrieve(content_hash): # This check is redundant, HexDictionary.store handles it internally
    #             #     print(f"Runtime: Persistent knowledge already loaded and stored with hash: {content_hash[:8]}...")
    #             #     return

    #             # If the data is a list of individual results, iterate and store them
    #             if isinstance(knowledge_data, list):
    #                 for i, item in enumerate(knowledge_data):
    #                     item_hash = self.hex_dict.store(item, 'json', metadata={
    #                         'data_type': 'ubp_periodic_table_entry',
    #                         'source_file': persistent_file_path,
    #                         'entry_index': i
    #                     })
    #                     print(f"Runtime: Stored periodic table entry {i} with hash: {item_hash[:8]}...")
    #             else:
    #                 # If it's a single object, store it directly
    #                 # content_hash = self.hex_dict.store(knowledge_data, 'json', metadata={
    #                 #     'data_type': 'ubp_periodic_table_results',
    #                 #     'source_file': persistent_file_path
    #                 # }) # This part now handled above
    #                 print(f"Runtime: Stored persistent UBP periodic table results with hash: {content_hash[:8]}...")
                
    #         except json.JSONDecodeError as e:
    #             print(f"Error loading persistent knowledge: Invalid JSON in {persistent_file_path}: {e}")
    #         except Exception as e:
    #             print(f"Unexpected error while loading persistent knowledge: {e}")
    #     else:
    #         print(f"Runtime: No persistent knowledge file found at {persistent_file_path}")
    
    def set_realm(self, realm_name: str):
        """
        Set the active realm for operations.
        
        Args:
            realm_name: Name of the realm to activate
        """
        if realm_name.lower() not in self.config.realms:
            available = list(self.config.realms.keys())
            raise ValueError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        self.state.active_realm = realm_name.lower()
    
    def get_realm_config(self, realm_name: str = None) -> Optional[RealmConfig]:
        """
        Get configuration for a realm.
        
        Args:
            realm_name: Realm name (uses active realm if None)
            
        Returns:
            Realm configuration dictionary
        """
        if realm_name is None:
            realm_name = self.state.active_realm
        
        return self.config.get_realm_config(realm_name) # Delegate to UBPConfig
    
    def initialize_bitfield(self, pattern: str = "sparse_random", 
                           density: float = 0.01, seed: int = None):
        """
        Initialize the Bitfield with a specific pattern.
        
        Args:
            pattern: Initialization pattern ("sparse_random", "quantum_bias", etc.)
            density: Density of active OffBits
            seed: Random seed for reproducibility
        """
        import random
        if seed is not None:
            random.seed(seed)
        
        self.bitfield.clear()
        
        # Determine total_cells based on new BITFIELD_DIMENSIONS from config
        total_cells = 1
        for dim_size in self.config.BITFIELD_DIMENSIONS:
            total_cells *= dim_size
        
        if pattern == "sparse_random":
            self._init_sparse_random(density, total_cells)
        elif pattern == "quantum_bias":
            self._init_quantum_bias(density, total_cells)
        elif pattern == "realm_specific":
            self._init_realm_specific(density, total_cells)
        else:
            raise ValueError(f"Unknown initialization pattern: {pattern}")
        
        # Reset bitfield statistics, assuming it has such a method
        # If MutableBitfield doesn't have reset_statistics, this call will fail.
        # For now, let's just update modified time.
        self.bitfield.last_modified = time.time()
    
    def _init_sparse_random(self, density: float, total_cells: int):
        """Initialize with sparse random pattern."""
        import random
        
        target_count = int(total_cells * density)
        target_count = min(target_count, self.bitfield.size) # Ensure not to exceed allocated size
        
        # Randomly choose indices to activate
        indices_to_activate = random.sample(range(self.bitfield.size), target_count)
        
        for index in indices_to_activate:
            value = random.randint(1, 0xFFFFFF)  # Non-zero 24-bit value
            offbit = OffBit(value)
            self.bitfield.set_offbit(index, offbit)
    
    def _init_quantum_bias(self, density: float, total_cells: int):
        """Initialize with quantum realm bias."""
        import random
        
        target_count = int(total_cells * density)
        target_count = min(target_count, self.bitfield.size)
        
        quantum_bias = self.config.constants.UBP_TOGGLE_PROBABILITIES.get('quantum', self.config.constants.E / 12)
        
        indices_to_activate = random.sample(range(self.bitfield.size), target_count)

        for index in indices_to_activate:
            # Bias toward quantum-like values
            if random.random() < quantum_bias:
                value = random.randint(0x100000, 0xFFFFFF)  # Higher values
            else:
                value = random.randint(1, 0x0FFFFF)  # Lower values
            
            offbit = OffBit(value)
            self.bitfield.set_offbit(index, offbit)
    
    def _init_realm_specific(self, density: float, total_cells: int):
        """Initialize with active realm-specific pattern."""
        realm_cfg = self.get_realm_config()
        
        if realm_cfg and realm_cfg.name.lower() in self.config.constants.UBP_TOGGLE_PROBABILITIES:
            bias = self.config.constants.UBP_TOGGLE_PROBABILITIES[realm_cfg.name.lower()]
            self._init_with_bias(density, total_cells, bias)
        else:
            self._init_sparse_random(density, total_cells)
    
    def _init_with_bias(self, density: float, total_cells: int, bias: float):
        """Initialize with specific toggle bias."""
        import random
        
        target_count = int(total_cells * density)
        target_count = min(target_count, self.bitfield.size)
        
        indices_to_activate = random.sample(range(self.bitfield.size), target_count)

        for index in indices_to_activate:
            # Apply bias to value generation
            if random.random() < bias:
                value = random.randint(0x800000, 0xFFFFFF)  # Upper half
            else:
                value = random.randint(1, 0x7FFFFF)  # Lower half
            
            offbit = OffBit(value)
            self.bitfield.set_offbit(index, offbit)
    
    def execute_toggle_operation(self, operation: str, coord1_idx: int, 
                                coord2_idx: Optional[int] = None, **kwargs) -> OffBit:
        """
        Execute a toggle operation between OffBits.
        
        Args:
            operation: Operation name ("and", "xor", "or", "resonance", etc.)
            coord1_idx: Index of the first OffBit in the flattened bitfield.
            coord2_idx: Index of the second OffBit in the flattened bitfield (if needed).
            **kwargs: Additional operation parameters
            
        Returns:
            Result OffBit
        """
        offbit1 = self.bitfield.get_offbit(coord1_idx)
        
        if coord2_idx is not None:
            offbit2 = self.bitfield.get_offbit(coord2_idx)
        else:
            offbit2 = OffBit(0) # Default for single-operand operations or if second coord is not relevant
        
        # Execute operation based on type
        if operation == "and":
            result = toggle_and(offbit1, offbit2)
        elif operation == "xor":
            result = toggle_xor(offbit1, offbit2)
        elif operation == "or":
            result = toggle_or(offbit1, offbit2)
        elif operation == "resonance":
            frequency = kwargs.get('frequency', 1.0)
            time_param = kwargs.get('time', self.state.global_time)
            result = resonance_toggle(offbit1, frequency, time_param)
        elif operation == "entanglement":
            coherence_val = kwargs.get('coherence', self.config.performance.COHERENCE_THRESHOLD)
            result = entanglement_toggle(offbit1, offbit2, coherence_val)
        elif operation == "superposition":
            weights = kwargs.get('weights', [0.5, 0.5])
            result = superposition_toggle([offbit1, offbit2], weights)
        elif operation == "hybrid_xor_resonance":
            distance = kwargs.get('distance', 1.0)
            result = hybrid_xor_resonance(offbit1, offbit2, distance)
        elif operation == "spin_transition":
            # Use realm-specific toggle probability from config for p_s
            p_s = self.config.constants.UBP_TOGGLE_PROBABILITIES.get(self.state.active_realm, self.config.constants.E / 12)
            result = spin_transition(offbit1, p_s)
        elif operation == "tgic":
            x_state = kwargs.get('x_state', True)
            y_state = kwargs.get('y_state', True)
            z_state = kwargs.get('z_state', False)
            result = apply_tgic_constraint(x_state, y_state, z_state, 
                                         offbit1, offbit2, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Update statistics
        self.operation_count += 1
        self.state.total_toggles += 1
        
        return result
    
    def run_simulation(self, steps: int, operations_per_step: int = 10,
                      target_indices: Optional[List[int]] = None,
                      record_timeline: bool = True) -> SimulationResult:
        """
        Run a UBP simulation for specified steps.
        
        Args:
            steps: Number of simulation steps
            operations_per_step: Toggle operations per step
            target_indices: Specific indices in the flattened bitfield to operate on (random if None)
            record_timeline: Whether to record state timeline
            
        Returns:
            SimulationResult with metrics and timeline
        """
        start_time = time.time()
        self.start_time = start_time
        
        # Record initial state
        initial_state = SimulationState(
            time_step=0,
            global_time=0.0,
            active_realm=self.state.active_realm,
            energy_value=self._calculate_current_energy(),
            nrci_value=0.0,  # Will be calculated during simulation
            coherence_pressure=0.0,
            total_toggles=0
        )
        
        timeline = [initial_state] if record_timeline else []
        
        # Run simulation steps
        for step in range(steps):
            self._execute_simulation_step(operations_per_step, target_indices)
            
            # Update state
            self.state.time_step = step + 1
            self.state.global_time = (step + 1) * self.config.temporal.BITTIME_UNIT_DURATION
            self.state.energy_value = self._calculate_current_energy()
            
            # Record timeline if requested
            if record_timeline:
                current_state = SimulationState(
                    time_step=self.state.time_step,
                    global_time=self.state.global_time,
                    active_realm=self.state.active_realm,
                    energy_value=self.state.energy_value,
                    nrci_value=self.state.nrci_value,
                    coherence_pressure=self.state.coherence_pressure,
                    total_toggles=self.state.total_toggles
                )
                timeline.append(current_state)
        
        # Calculate final metrics
        final_metrics = self._calculate_final_metrics()
        execution_time = time.time() - start_time
        
        # Create result
        result = SimulationResult(
            initial_state=initial_state,
            final_state=self.state,
            metrics=final_metrics,
            timeline=timeline,
            execution_time=execution_time
        )
        
        # Store SimulationResult in HexDictionary
        self._store_simulation_result(result)
        
        return result
    
    def _store_simulation_result(self, result: SimulationResult):
        """
        Stores a SimulationResult object in the HexDictionary.
        """
        result_dict = result.to_dict()
        
        # Define structured metadata for easy retrieval
        metadata = {
            'data_type': 'ubp_simulation_result',
            'simulation_id': f"sim_{int(time.time())}",
            'final_nrci': result.final_state.nrci_value,
            'total_toggles': result.final_state.total_toggles,
            'active_realm': result.final_state.active_realm,
            'execution_time': result.execution_time,
            'timestamp': time.time()
        }
        
        content_hash = self.hex_dict.store(result_dict, 'json', metadata=metadata)
        print(f"Runtime: Stored simulation result with hash: {content_hash[:8]}... (NRCI: {result.final_state.nrci_value:.4f})")

    def _execute_simulation_step(self, operations_per_step: int, 
                                target_indices: Optional[List[int]] = None):
        """Execute a single simulation step."""
        import random
        
        active_offbits_indexed = self.bitfield.get_active_offbits()
        if not active_offbits_indexed:
            return  # No active OffBits to operate on
        
        indices_list = [idx for idx, _ in active_offbits_indexed]
        
        for _ in range(operations_per_step):
            if target_indices:
                idx1 = random.choice(target_indices)
                idx2 = random.choice(target_indices) if len(target_indices) > 1 else None
            else:
                idx1 = random.choice(indices_list)
                idx2 = random.choice(indices_list) if len(indices_list) > 1 else None
            
            # Choose operation based on realm
            operation = self._choose_realm_operation()
            
            try:
                result = self.execute_toggle_operation(operation, idx1, idx2)
                # Store result back to first coordinate
                self.bitfield.set_offbit(idx1, result)
            except Exception as e:
                # Skip failed operations, log for debugging if necessary
                # print(f"Warning: Toggle operation '{operation}' failed at step {self.state.time_step}: {e}")
                continue
    
    def _choose_realm_operation(self) -> str:
        """Choose an operation based on the active realm."""
        import random
        
        realm_operations = {
            "quantum": ["resonance", "spin_transition", "tgic", "entanglement"],
            "electromagnetic": ["and", "or", "resonance", "hybrid_xor_resonance"],
            "gravitational": ["entanglement", "superposition", "resonance"],
            "biological": ["hybrid_xor_resonance", "superposition", "spin_transition"],
            "cosmological": ["spin_transition", "entanglement", "superposition"],
            "nuclear": ["resonance", "tgic", "hybrid_xor_resonance"],
            "optical": ["and", "xor", "resonance"]
        }
        
        operations = realm_operations.get(self.state.active_realm, ["xor", "and", "or"])
        return random.choice(operations)
    
    def _calculate_current_energy(self) -> float:
        """Calculate current system energy."""
        active_count = self.bitfield.active_count
        if active_count == 0:
            return 0.0
        
        # All required constants for 'energy' function come from _config globally or are defaults.
        # R_0 = _config.constants.UBP_ENERGY_PARAMS['R0'] # UBP_ENERGY_PARAMS is gone.
        # H_t = _config.constants.UBP_ENERGY_PARAMS['Ht']
        # The energy function directly calculates resonance_strength, etc., so we just pass M.
        return energy(M=active_count)
    
    def _calculate_final_metrics(self) -> Dict[str, float]:
        """Calculate final simulation metrics."""
        active_offbits_indexed = self.bitfield.get_active_offbits()
        if not active_offbits_indexed:
            return {"nrci": 0.0, "coherence_pressure": 0.0, "fractal_dimension": 0.0, "coherence_score": 0.0, "active_offbits": 0, "total_offbits": self.bitfield.size, "sparsity": 0.0, "energy": self.state.energy_value}
        
        # Extract values for NRCI calculation
        simulated_values = [float(offbit.value) for _, offbit in active_offbits_indexed]
        
        # Create synthetic target (for demonstration) - should be a more robust target in reality
        import random
        if not simulated_values:
            target_values = [0.0] * self.bitfield.size
        else:
            target_values = [val + random.gauss(0, val * 0.01) if val != 0 else random.gauss(0, 0.01) for val in simulated_values]
            
        # If simulated_values is shorter than bitfield size, pad target_values
        if len(simulated_values) < self.bitfield.size:
             # For NRCI, we compare based on available active elements
            pass # NRCI takes lists, not bitfields directly.

        # Calculate NRCI
        # NRCI takes List[float], so ensure both are lists of floats
        nrci_value = nrci(simulated_values, target_values)
        self.state.nrci_value = nrci_value
        
        # Calculate coherence pressure
        # Simplified parameters for coherence_pressure_spatial
        distances = [1.0] * len(active_offbits_indexed)  # Placeholder
        # The number 12 here comes from the expected number of layers/dimensions in the UBP model
        # which can be mapped to Max_Bitfield_Dimensions from system_constants.
        max_distances = [10.0] * len(active_offbits_indexed)  # Placeholder, should be derived from bitfield geometry
        active_bits = [offbit.active_bits for _, offbit in active_offbits_indexed]
        
        # The `active_bits` sum `Σb_j` in the `coherence_pressure_spatial` formula,
        # is normally divided by 12 (as per its `metrics.py` implementation), referring
        # to the 12 bits in the Reality/Information layer. This value needs to be contextualized.
        coherence_pressure = coherence_pressure_spatial(distances, max_distances, active_bits)
        self.state.coherence_pressure = coherence_pressure
        
        # Calculate fractal dimension (using active offbits count for a simplified fractal_dimension calculation)
        fractal_dim = fractal_dimension(len(active_offbits_indexed)) # Assumes number of clusters
        
        # Calculate overall coherence score
        coherence_score = calculate_system_coherence_score(
            nrci=nrci_value,
            coherence_pressure=coherence_pressure,
            fractal_dim=fractal_dim,
            sri=0.8,  # Simplified Spatial Resonance Index
            cri=0.9   # Simplified Coherence Resonance Index
        )
        
        return {
            "nrci": nrci_value,
            "coherence_pressure": coherence_pressure,
            "fractal_dimension": fractal_dim,
            "coherence_score": coherence_score,
            "active_offbits": len(active_offbits_indexed),
            "total_offbits": self.bitfield.size,
            "sparsity": self.bitfield.current_sparsity, # Assuming this is available
            "energy": self.state.energy_value
        }
    
    def export_state(self, filepath: str, format: str = "json"):
        """
        Export current runtime state to file.
        
        Args:
            filepath: Output file path
            format: Export format ("json", "yaml")
        """
        state_data = {
            "runtime_state": self.state.to_dict(),
            "bitfield_stats": {
                "dimensions": self.config.BITFIELD_DIMENSIONS,
                "active_count": self.bitfield.active_count,
                "total_offbits": self.bitfield.size,
                "sparsity": self.bitfield.current_sparsity,
                "toggle_count": self.operation_count # Total operations acts as toggle count here
            },
            "realm_configs": {name: realm.to_dict() for name, realm in self.config.realms.items()},
            "operation_count": self.operation_count,
            "hex_dictionary_stats": self.hex_dict.get_metadata_stats() # Include HexDictionary stats
        }
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
        elif format == "yaml":
            import yaml
            with open(filepath, 'w') as f:
                yaml.dump(state_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset(self):
        """Reset the runtime to initial state."""
        self.bitfield.clear()
        self.state = SimulationState()
        self.timeline.clear()
        self.operation_count = 0
        self.start_time = 0.0
        # HexDictionary should not be cleared on runtime reset, as it's persistent
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get runtime performance statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time > 0 else 0.0
        
        return {
            "elapsed_time": elapsed_time,
            "operations_per_second": self.operation_count / elapsed_time if elapsed_time > 0 else 0.0,
            "total_operations": self.operation_count,
            "memory_efficiency": self.bitfield.current_sparsity, # Assuming this property exists
            "active_offbits": self.bitfield.active_count
        }
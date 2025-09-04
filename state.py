"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP State Management Module
Author: Euan Craig, New Zealand
Date: 03 September 2025
======================================

Defines core state classes for the Universal Binary Principle system,
including OffBit, MutableBitfield, and UBPState.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import time
import math

# Import UBPConfig and get_config for constant loading
from ubp_config import get_config, UBPConfig


_config: UBPConfig = get_config() # Initialize configuration


@dataclass(frozen=True)
class OffBit:
    """
    Immutable 24-bit UBP OffBit with layer properties.
    
    Represents a fundamental unit of UBP computation with 24-bit data
    and layer-based access patterns.
    """
    value: int
    
    def __post_init__(self):
        # Ensure value is within 24-bit range
        if not (0 <= self.value <= 0xFFFFFF):
            object.__setattr__(self, 'value', self.value & 0xFFFFFF)
    
    @property
    def layer(self) -> int:
        """Get the 24-bit layer value."""
        return self.value & 0xFFFFFF
    
    @property
    def bits(self) -> List[int]:
        """Get individual bits as a list."""
        return [(self.value >> i) & 1 for i in range(24)]
    
    @property
    def active_bits(self) -> int:
        """Count of active (1) bits."""
        return bin(self.value).count('1')
    
    @property
    def is_active(self) -> bool:
        """Check if OffBit has any active bits."""
        return self.value > 0
    
    def toggle(self) -> 'OffBit':
        """
        Create a new OffBit with toggled state.
        
        Returns:
            New OffBit with inverted bits
        """
        return OffBit(self.value ^ 0xFFFFFF)
    
    def toggle_bit(self, position: int) -> 'OffBit':
        """
        Create a new OffBit with a specific bit toggled.
        
        Args:
            position: Bit position to toggle (0-23)
        
        Returns:
            New OffBit with specified bit toggled
        """
        if not (0 <= position < 24):
            raise ValueError(f"Bit position {position} out of range [0, 23]")
        
        return OffBit(self.value ^ (1 << position))
    
    def get_bit(self, position: int) -> int:
        """
        Get the value of a specific bit.
        
        Args:
            position: Bit position (0-23)
        
        Returns:
            Bit value (0 or 1)
        """
        if not (0 <= position < 24):
            raise ValueError(f"Bit position {position} out of range [0, 23]")
        
        return (self.value >> position) & 1
    
    def set_bit(self, position: int, value: int) -> 'OffBit':
        """
        Create a new OffBit with a specific bit set.
        
        Args:
            position: Bit position (0-23)
            value: Bit value (0 or 1)
        
        Returns:
            New OffBit with specified bit set
        """
        if not (0 <= position < 24):
            raise ValueError(f"Bit position {position} out of range [0, 23]")
        if value not in (0, 1):
            raise ValueError(f"Bit value must be 0 or 1, got {value}")
        
        if value == 1:
            return OffBit(self.value | (1 << position))
        else:
            return OffBit(self.value & ~(1 << position))
    
    def extract_data(self) -> int:
        """
        Extract 24-bit data for Golay correction.
        
        Returns:
            24-bit data value
        """
        return self.layer
    
    def __str__(self) -> str:
        return f"OffBit(0x{self.value:06X})"
    
    def __repr__(self) -> str:
        return f"OffBit(value={self.value}, layer=0x{self.layer:06X}, active_bits={self.active_bits})"


class MutableBitfield:
    """
    Mutable bitfield for UBP operations.
    
    Provides efficient storage and manipulation of large collections of OffBits.
    """
    
    def __init__(self, size: int = 1000):
        """
        Initialize mutable bitfield.
        
        Args:
            size: Number of OffBits to store
        """
        self.size = size
        self.data = np.zeros(size, dtype=np.uint32)
        self.active_count = 0
        self.last_modified = time.time()
    
    @property
    def current_sparsity(self) -> float:
        """Calculate the current sparsity of the bitfield."""
        if self.size == 0:
            return 1.0 # Fully sparse if no capacity
        return (self.size - self.active_count) / self.size

    def get_offbit(self, index: int) -> OffBit:
        """
        Get OffBit at specified index.
        
        Args:
            index: Index in the bitfield
        
        Returns:
            OffBit at the specified index
        """
        if not (0 <= index < self.size):
            raise IndexError(f"Index {index} out of range [0, {self.size})")
        
        return OffBit(int(self.data[index]) & 0xFFFFFF)
    
    def set_offbit(self, index: int, offbit: OffBit) -> None:
        """
        Set OffBit at specified index.
        
        Args:
            index: Index in the bitfield
            offbit: OffBit to set
        """
        if not (0 <= index < self.size):
            raise IndexError(f"Index {index} out of range [0, {self.size})")
        
        old_value = self.data[index]
        new_value = offbit.value & 0xFFFFFF
        
        self.data[index] = new_value
        
        # Update active count
        if old_value == 0 and new_value != 0:
            self.active_count += 1
        elif old_value != 0 and new_value == 0:
            self.active_count -= 1
        
        self.last_modified = time.time()
    
    def toggle_offbit(self, index: int) -> None:
        """
        Toggle OffBit at specified index.
        
        Args:
            index: Index in the bitfield
        """
        current_offbit = self.get_offbit(index)
        toggled_offbit = current_offbit.toggle()
        self.set_offbit(index, toggled_offbit)
    
    def get_active_offbits(self) -> List[Tuple[int, OffBit]]:
        """
        Get all active OffBits.
        
        Returns:
            List of (index, OffBit) tuples for active OffBits
        """
        active_offbits = []
        for i in range(self.size):
            if self.data[i] != 0:
                active_offbits.append((i, self.get_offbit(i)))
        return active_offbits
    
    def get_coherence(self) -> float:
        """
        Compute bitfield coherence.
        
        Returns:
            Coherence value (0 to 1)
        """
        if self.size == 0:
            return 1.0
        
        # Compute statistical coherence
        active_ratio = self.active_count / self.size
        
        # Compute spatial coherence (clustering)
        if self.active_count > 1:
            active_indices = np.where(self.data != 0)[0]
            if len(active_indices) > 1:
                distances = np.diff(active_indices)
                mean_distance = np.mean(distances)
                std_distance = np.std(distances)
                
                # Lower standard deviation = higher coherence
                spatial_coherence = 1.0 / (1.0 + std_distance / (mean_distance + 1e-10))
            else:
                spatial_coherence = 1.0
        else:
            spatial_coherence = 1.0
        
        # Combine coherence measures
        total_coherence = 0.5 * active_ratio + 0.5 * spatial_coherence
        
        return min(1.0, total_coherence)
    
    def compute_nrci(self, target_bitfield: 'MutableBitfield') -> float:
        """
        Compute Non-Random Coherence Index with target bitfield.
        
        Args:
            target_bitfield: Target bitfield for comparison
        
        Returns:
            NRCI value (0 to 1)
        """
        if self.size != target_bitfield.size:
            raise ValueError("Bitfields must have the same size for NRCI calculation")
        
        # Convert to float arrays for better precision
        data1 = self.data.astype(np.float64)
        data2 = target_bitfield.data.astype(np.float64)
        
        # Compute correlation coefficient
        if np.std(data1) == 0 or np.std(data2) == 0:
            # If either dataset has no variation, use exact match
            exact_matches = np.sum(data1 == data2)
            return exact_matches / self.size
        
        # Compute Pearson correlation coefficient
        correlation = np.corrcoef(data1, data2)[0, 1]
        
        # Handle NaN correlation (when one or both arrays are constant)
        if np.isnan(correlation):
            exact_matches = np.sum(data1 == data2)
            return exact_matches / self.size
        
        # Convert correlation to NRCI (0 to 1 scale)
        # Perfect correlation (1.0) = NRCI 1.0
        # No correlation (0.0) = NRCI 0.5
        # Perfect anti-correlation (-1.0) = NRCI 0.0
        nrci = (correlation + 1.0) / 2.0
        
        return max(0.0, min(1.0, nrci))
    
    def resize(self, new_size: int) -> None:
        """
        Resize the bitfield.
        
        Args:
            new_size: New size for the bitfield
        """
        if new_size <= 0:
            raise ValueError("New size must be positive")
        
        old_data = self.data
        self.data = np.zeros(new_size, dtype=np.uint32)
        
        # Copy existing data
        copy_size = min(self.size, new_size)
        self.data[:copy_size] = old_data[:copy_size]
        
        # Update size and active count
        self.size = new_size
        self.active_count = np.count_nonzero(self.data)
        self.last_modified = time.time()
    
    def clear(self) -> None:
        """Clear all OffBits in the bitfield."""
        self.data.fill(0)
        self.active_count = 0
        self.last_modified = time.time()
    
    def copy(self) -> 'MutableBitfield':
        """
        Create a copy of the bitfield.
        
        Returns:
            Copy of the bitfield
        """
        new_bitfield = MutableBitfield(self.size)
        new_bitfield.data = self.data.copy()
        new_bitfield.active_count = self.active_count
        new_bitfield.last_modified = self.last_modified
        return new_bitfield
    
    def __len__(self) -> int:
        return self.size
    
    def __str__(self) -> str:
        return f"MutableBitfield(size={self.size}, active={self.active_count}, coherence={self.get_coherence():.4f})"
    
    def __repr__(self) -> str:
        return f"MutableBitfield(size={self.size}, active_count={self.active_count}, last_modified={self.last_modified})"


@dataclass
class UBPState:
    """
    Complete UBP system state.
    
    Represents the full state of a UBP system including bitfields,
    coherence metrics, and temporal information.
    """
    bitfield: MutableBitfield
    timestamp: float = field(default_factory=time.time)
    realm: str = "quantum"
    coherence: float = 0.0
    nrci: float = 0.0
    energy: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Update coherence after initialization."""
        self.update_coherence()
    
    def update_coherence(self) -> None:
        """Update coherence metrics."""
        self.coherence = self.bitfield.get_coherence()
        self.timestamp = time.time()
    
    def compute_energy(self) -> float:
        """
        Compute UBP energy for the current state.
        
        Returns:
            UBP energy value
        """
        # Get energy parameters directly from config
        M = self.bitfield.active_count
        C = _config.constants.SPEED_OF_LIGHT
        
        # These constants are no longer in UBP_ENERGY_PARAMS, use direct config lookup
        R_0 = 0.95 # Default from resonance_strength
        H_t = 0.05 # Default from resonance_strength
        R = R_0 * (1 - H_t / math.log(4)) # resonance_strength calculation
        
        S_opt_default = 0.98 # Default from structural_optimality
        S_opt = S_opt_default
        
        # Simplified energy calculation (matching the current energy function's structure for basic use)
        # Note: A full energy calculation would involve P_GCI, O_observer, c_infinity etc.
        # This is a simplified proxy for `UBPState` to track its own energy.
        self.energy = M * C * R * S_opt * self.coherence
        
        return self.energy
    
    def evolve(self, delta_t: float = 0.001) -> None:
        """
        Evolve the UBP state over time.
        
        Args:
            delta_t: Time step for evolution
        """
        # Get toggle probability for the current realm from config
        toggle_prob = _config.constants.UBP_TOGGLE_PROBABILITIES.get(self.realm, 0.5)
        
        # Determine how many OffBits to toggle
        num_toggles = int(self.bitfield.size * toggle_prob * delta_t)
        
        # Randomly select OffBits to toggle
        if num_toggles > 0:
            indices = np.random.choice(self.bitfield.size, size=min(num_toggles, self.bitfield.size), replace=False)
            
            for index in indices:
                self.bitfield.toggle_offbit(index)
        
        # Update state
        self.update_coherence()
        self.compute_energy()
        self.timestamp = time.time()
    
    def copy(self) -> 'UBPState':
        """
        Create a copy of the UBP state.
        
        Returns:
            Copy of the UBP state
        """
        return UBPState(
            bitfield=self.bitfield.copy(),
            timestamp=self.timestamp,
            realm=self.realm,
            coherence=self.coherence,
            nrci=self.nrci,
            energy=self.energy,
            metadata=self.metadata.copy()
        )
    
    def __str__(self) -> str:
        return f"UBPState(realm={self.realm}, coherence={self.coherence:.4f}, nrci={self.nrci:.6f}, energy={self.energy:.2e})"


def create_test_bitfield(size: int = 1000, active_ratio: float = 0.1) -> MutableBitfield:
    """
    Create a test bitfield with specified parameters.
    
    Args:
        size: Size of the bitfield
        active_ratio: Ratio of active OffBits
    
    Returns:
        Test bitfield
    """
    bitfield = MutableBitfield(size)
    
    # Randomly activate OffBits
    num_active = int(size * active_ratio)
    active_indices = np.random.choice(size, size=num_active, replace=False)
    
    for index in active_indices:
        # Create random OffBit value
        value = np.random.randint(1, 0xFFFFFF)
        offbit = OffBit(value)
        bitfield.set_offbit(index, offbit)
    
    return bitfield


def create_test_state(size: int = 1000, realm: str = "quantum") -> UBPState:
    """
    Create a test UBP state.
    
    Args:
        size: Size of the bitfield
        realm: UBP realm
    
    Returns:
        Test UBP state
    """
    bitfield = create_test_bitfield(size)
    state = UBPState(bitfield=bitfield, realm=realm)
    state.compute_energy()
    return state


if __name__ == "__main__":
    # Test OffBit functionality
    print("Testing OffBit...")
    
    offbit = OffBit(0xABCDEF)
    print(f"OffBit: {offbit}")
    print(f"Layer: 0x{offbit.layer:06X}")
    print(f"Active bits: {offbit.active_bits}")
    print(f"Bit 0: {offbit.get_bit(0)}")
    print(f"Bit 23: {offbit.get_bit(23)}")
    
    toggled = offbit.toggle()
    print(f"Toggled: {toggled}")
    
    # Test MutableBitfield
    print(f"\nTesting MutableBitfield...")
    
    bitfield = create_test_bitfield(100, 0.2)
    print(f"Bitfield: {bitfield}")
    print(f"Active OffBits: {len(bitfield.get_active_offbits())}")
    print(f"Coherence: {bitfield.get_coherence():.4f}")
    
    # Test UBPState
    print(f"\nTesting UBPState...")
    
    state = create_test_state(100, "quantum")
    print(f"State: {state}")
    
    # Evolve state
    print(f"\nEvolving state...")
    for i in range(5):
        state.evolve(0.01)
        print(f"Step {i+1}: coherence={state.coherence:.4f}, energy={state.energy:.2e}")
    
    print(f"\nUBP state management tests completed.")
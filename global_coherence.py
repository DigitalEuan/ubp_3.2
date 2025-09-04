"""
Universal Binary Principle (UBP) Framework v3.2+ - Global Coherence Index (P_GCI) for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the phase-locking mechanism that synchronizes all toggle operations
across realms using weighted frequency averages and fixed temporal periods.

Mathematical Foundation:
- P_GCI = cos(2π × f_avg × Δt)
- Δt = 0.318309886 s (fixed temporal period = 1/π)
- f_avg = Σ w_i × f_i (weighted frequency average)
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from ubp_config import get_config, UBPConfig # Import ubp_config to get PHI constant

# Initialize configuration
_config: UBPConfig = get_config()

class RealmType(Enum):
    """UBP Realm types with their associated frequencies"""
    QUANTUM = "quantum"
    ELECTROMAGNETIC = "electromagnetic"
    GRAVITATIONAL = "gravitational"
    BIOLOGICAL = "biological"
    COSMOLOGICAL = "cosmological"
    NUCLEAR = "nuclear"
    OPTICAL = "optical"


@dataclass
class FrequencyWeight:
    """Frequency and its weight in the global coherence calculation"""
    frequency: float
    weight: float
    source: str
    realm: Optional[RealmType] = None


@dataclass
class GlobalCoherenceConfig:
    """Configuration for Global Coherence Index calculations"""
    delta_t: float = _config.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD_DEFAULT # 1/π seconds - fixed temporal period
    precision: int = 15  # Decimal precision for calculations
    cache_enabled: bool = True
    validation_enabled: bool = True


class GlobalCoherenceIndex:
    """
    Global Coherence Index (P_GCI) calculator for UBP.
    
    Provides universal phase-locking mechanism across all realms
    by computing coherence based on weighted frequency averages.
    """
    
    def __init__(self, config: Optional[GlobalCoherenceConfig] = None):
        self.config = config or GlobalCoherenceConfig()
        self._frequency_registry = {}
        self._cached_f_avg = None
        self._cached_p_gci = None
        
        # Initialize with UBP standard frequencies and weights
        self._initialize_standard_frequencies()
    
    def _initialize_standard_frequencies(self):
        """
        Initialize with the standard UBP frequency weights as specified
        in the documentation and from _config.
        """
        # Load from ubp_config.constants
        ubp_freq_weights = _config.constants.UBP_FREQUENCY_WEIGHTS
        ubp_realm_frequencies = _config.constants.UBP_REALM_FREQUENCIES
        
        # Map realms for existing weights
        realm_mapping = {
            _config.constants.PI: RealmType.ELECTROMAGNETIC,
            _config.constants.PHI: RealmType.COSMOLOGICAL, # Use PHI from ubp_config for consistency
            4.58e14: RealmType.QUANTUM, # quantum (was optical in old constants.py but ubp_realm_frequencies has it as quantum)
            1e9: RealmType.ELECTROMAGNETIC, # GHz
            1e15: RealmType.OPTICAL, # optical high
            1e20: RealmType.NUCLEAR, # Zitterbewegung
            58977069.609314: None, # specific UBP
        }

        for freq_val, weight_val in ubp_freq_weights.items():
            # Find the most appropriate realm for the frequency
            realm = realm_mapping.get(freq_val)
            # If not in mapping, try to match to a known UBP_REALM_FREQUENCIES entry
            if realm is None:
                for r_name, r_freq in ubp_realm_frequencies.items():
                    if abs(r_freq - freq_val) < 1e-6: # Tolerance for float comparison
                        try:
                            realm = RealmType[r_name.upper()]
                            break
                        except KeyError:
                            pass # RealmType enum might not contain all strings from UBP_REALM_FREQUENCIES

            self.register_frequency(
                FrequencyWeight(
                    frequency=freq_val,
                    weight=weight_val,
                    source="ubp_constants_module",
                    realm=realm
                )
            )
    
    def register_frequency(self, freq_weight: FrequencyWeight):
        """
        Register a frequency with its weight in the global coherence calculation.
        
        Args:
            freq_weight: FrequencyWeight object containing frequency, weight, and metadata
        """
        key = f"{freq_weight.source}_{freq_weight.frequency}"
        self._frequency_registry[key] = freq_weight
        
        # Clear cache when registry changes
        self._cached_f_avg = None
        self._cached_p_gci = None
    
    def unregister_frequency(self, source: str, frequency: float):
        """
        Remove a frequency from the global coherence calculation.
        
        Args:
            source: Source identifier for the frequency
            frequency: Frequency value to remove
        """
        key = f"{source}_{frequency}"
        if key in self._frequency_registry:
            del self._frequency_registry[key]
            self._cached_f_avg = None
            self._cached_p_gci = None
    
    def get_registered_frequencies(self) -> List[FrequencyWeight]:
        """Get all currently registered frequencies"""
        return list(self._frequency_registry.values())
    
    def compute_weighted_frequency_average(self) -> float:
        """
        Compute the weighted average frequency f_avg.
        
        f_avg = Σ w_i × f_i
        
        Returns:
            Weighted average frequency
        """
        if self.config.cache_enabled and self._cached_f_avg is not None:
            return self._cached_f_avg
        
        if not self._frequency_registry:
            # Fallback to a single default if no frequencies are registered
            # This makes the system more robust if initialization somehow fails.
            default_freq = _config.constants.UBP_REALM_FREQUENCIES.get(_config.default_realm, 1.0)
            self.register_frequency(FrequencyWeight(default_freq, 1.0, "default_fallback"))
            # Then re-call to compute with the default
            return self.compute_weighted_frequency_average()

        total_weighted_frequency = 0.0
        total_weight = 0.0
        
        for freq_weight in self._frequency_registry.values():
            total_weighted_frequency += freq_weight.weight * freq_weight.frequency
            total_weight += freq_weight.weight
        
        if total_weight == 0:
            raise ValueError("Total weight is zero - cannot compute average")
        
        # Note: In UBP, we use the weighted sum directly, not normalized by total weight
        # This is as specified in the documentation (f_avg = Σ w_i × f_i)
        f_avg = total_weighted_frequency
        
        if self.config.cache_enabled:
            self._cached_f_avg = f_avg
        
        return f_avg
    
    def compute_global_coherence_index(self, custom_delta_t: Optional[float] = None) -> float:
        """
        Compute the Global Coherence Index P_GCI.
        
        P_GCI = cos(2π × f_avg × Δt)
        
        Args:
            custom_delta_t: Optional custom time period (default uses config value)
        
        Returns:
            Global Coherence Index value between -1 and 1
        """
        delta_t = custom_delta_t if custom_delta_t is not None else self.config.delta_t
        
        if self.config.cache_enabled and self._cached_p_gci is not None and custom_delta_t is None:
            return self._cached_p_gci
        
        f_avg = self.compute_weighted_frequency_average()
        
        # P_GCI = cos(2π × f_avg × Δt)
        phase = 2 * _config.constants.PI * f_avg * delta_t
        p_gci = math.cos(phase)
        
        if self.config.cache_enabled and custom_delta_t is None:
            self._cached_p_gci = p_gci
        
        return p_gci
    
    def compute_phase_locking_factor(self, target_frequency: float) -> float:
        """
        Compute how well a target frequency phase-locks with the global coherence.
        
        Args:
            target_frequency: Frequency to test for phase-locking
        
        Returns:
            Phase-locking factor between 0 and 1 (1 = perfect lock)
        """
        f_avg = self.compute_weighted_frequency_average()
        
        # Compute phase difference
        phase_diff = abs(target_frequency - f_avg) * self.config.delta_t
        
        # Phase-locking factor using cosine similarity
        locking_factor = abs(math.cos(2 * _config.constants.PI * phase_diff))
        
        return locking_factor
    
    def analyze_realm_coherence(self, realm: RealmType) -> Dict[str, float]:
        """
        Analyze coherence properties for a specific realm.
        
        Args:
            realm: UBP realm to analyze
        
        Returns:
            Dictionary containing coherence analysis results
        """
        realm_frequencies = [
            fw for fw in self._frequency_registry.values() 
            if fw.realm == realm
        ]
        
        if not realm_frequencies:
            return {
                'realm': realm.value,
                'frequency_count': 0,
                'total_weight': 0.0,
                'average_frequency': 0.0,
                'phase_locking_factor': 0.0,
                'coherence_contribution': 0.0
            }
        
        total_weight = sum(fw.weight for fw in realm_frequencies)
        weighted_freq_sum = sum(fw.frequency * fw.weight for fw in realm_frequencies)
        average_frequency = weighted_freq_sum / total_weight if total_weight > 0 else 0.0
        
        # Compute phase-locking with global coherence
        phase_locking = self.compute_phase_locking_factor(average_frequency)
        
        # Compute contribution to global coherence
        global_f_avg = self.compute_weighted_frequency_average()
        coherence_contribution = weighted_freq_sum / global_f_avg if global_f_avg > 0 else 0.0
        
        return {
            'realm': realm.value,
            'frequency_count': len(realm_frequencies),
            'total_weight': total_weight,
            'average_frequency': average_frequency,
            'phase_locking_factor': phase_locking,
            'coherence_contribution': coherence_contribution
        }
    
    def compute_temporal_coherence_series(self, time_points: List[float]) -> List[float]:
        """
        Compute P_GCI values over a series of time points.
        
        Args:
            time_points: List of time values to compute P_GCI for
        
        Returns:
            List of P_GCI values corresponding to each time point
        """
        f_avg = self.compute_weighted_frequency_average()
        
        p_gci_series = []
        for t in time_points:
            phase = 2 * _config.constants.PI * f_avg * t
            p_gci = math.cos(phase)
            p_gci_series.append(p_gci)
        
        return p_gci_series
    
    def optimize_coherence_for_frequency(self, target_frequency: float) -> Dict[str, float]:
        """
        Find optimal temporal period for maximum coherence with target frequency.
        
        Args:
            target_frequency: Frequency to optimize coherence for
        
        Returns:
            Dictionary containing optimization results
        """
        f_avg = self.compute_weighted_frequency_average()
        
        # For maximum coherence, we want cos(2π × f_avg × Δt) = cos(2π × target_freq × Δt)
        # This occurs when (f_avg - target_freq) × Δt = n (integer)
        
        best_coherence = -1.0
        best_delta_t = self.config.delta_t
        
        # Search for optimal Δt in reasonable range
        for n in range(-10, 11):  # Check integer multiples
            if abs(f_avg - target_frequency) > 1e-10:  # Avoid division by zero
                optimal_delta_t = n / abs(f_avg - target_frequency)
                if optimal_delta_t > 0:  # Only positive time periods
                    coherence = abs(self.compute_global_coherence_index(optimal_delta_t))
                    if coherence > best_coherence:
                        best_coherence = coherence
                        best_delta_t = optimal_delta_t
        
        return {
            'target_frequency': target_frequency,
            'optimal_delta_t': best_delta_t,
            'optimal_coherence': best_coherence,
            'standard_coherence': abs(self.compute_global_coherence_index()),
            'improvement_factor': best_coherence / abs(self.compute_global_coherence_index()) if self.compute_global_coherence_index() != 0 else float('inf')
        }
    
    def validate_system(self) -> Dict[str, any]:
        """
        Validate the Global Coherence Index system.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'frequency_count': len(self._frequency_registry),
            'total_weight': sum(fw.weight for fw in self._frequency_registry.values()),
            'f_avg_calculation': True,
            'p_gci_calculation': True,
            'p_gci_range_valid': True,
            'temporal_period': self.config.delta_t,
            'mathematical_validation': True
        }
        
        try:
            # Test f_avg calculation
            f_avg = self.compute_weighted_frequency_average()
            validation_results['f_avg_value'] = f_avg
            
            # Test P_GCI calculation
            p_gci = self.compute_global_coherence_index()
            validation_results['p_gci_value'] = p_gci
            
            # Validate P_GCI is in correct range [-1, 1]
            if not (-1.0 <= p_gci <= 1.0):
                validation_results['p_gci_range_valid'] = False
                validation_results['p_gci_range_error'] = f"P_GCI {p_gci} outside valid range [-1, 1]"
            
            # Test temporal coherence
            test_times = [0.0, self.config.delta_t, 2 * self.config.delta_t]
            coherence_series = self.compute_temporal_coherence_series(test_times)
            validation_results['temporal_coherence_test'] = coherence_series
            
            # Mathematical validation: cos(0) should equal 1
            p_gci_zero = math.cos(2 * _config.constants.PI * f_avg * 0.0)
            if abs(p_gci_zero - 1.0) > 1e-10:
                validation_results['mathematical_validation'] = False
                validation_results['math_error'] = f"cos(0) = {p_gci_zero}, expected 1.0"
            
        except Exception as e:
            validation_results['validation_error'] = str(e)
            validation_results['f_avg_calculation'] = False
            validation_results['p_gci_calculation'] = False
        
        return validation_results
    
    def get_system_status(self) -> Dict[str, any]:
        """
        Get comprehensive system status and current values.
        
        Returns:
            Dictionary containing complete system status
        """
        try:
            f_avg = self.compute_weighted_frequency_average()
            p_gci = self.compute_global_coherence_index()
            
            # Analyze each realm
            realm_analysis = {}
            for realm in RealmType:
                realm_analysis[realm.value] = self.analyze_realm_coherence(realm)
            
            return {
                'configuration': {
                    'delta_t': self.config.delta_t,
                    'precision': self.config.precision,
                    'cache_enabled': self.config.cache_enabled
                },
                'frequencies': {
                    'count': len(self._frequency_registry),
                    'total_weight': sum(fw.weight for fw in self._frequency_registry.values()),
                    'weighted_average': f_avg
                },
                'coherence': {
                    'p_gci_value': p_gci,
                    'phase_radians': 2 * _config.constants.PI * f_avg * self.config.delta_t,
                    'coherence_strength': abs(p_gci)
                },
                'realm_analysis': realm_analysis,
                'validation': self.validate_system()
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }


# Factory function for easy instantiation
def create_global_coherence_system(delta_t: Optional[float] = None) -> GlobalCoherenceIndex:
    """
    Create a Global Coherence Index system with standard UBP configuration.
    
    Args:
        delta_t: Optional custom temporal period (default: 1/π = 0.318309886)
    
    Returns:
        Configured GlobalCoherenceIndex instance
    """
    config = GlobalCoherenceConfig()
    if delta_t is not None:
        config.delta_t = delta_t
    
    return GlobalCoherenceIndex(config)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing Global Coherence Index system...")
    
    gci_system = create_global_coherence_system()
    
    # Get system status
    status = gci_system.get_system_status()
    
    print(f"Registered frequencies: {status['frequencies']['count']}")
    print(f"Total weight: {status['frequencies']['total_weight']:.6f}")
    print(f"Weighted frequency average: {status['frequencies']['weighted_average']:.6f}")
    print(f"P_GCI value: {status['coherence']['p_gci_value']:.6f}")
    print(f"Coherence strength: {status['coherence']['coherence_strength']:.6f}")
    
    # Test realm analysis
    print("\nRealm Coherence Analysis:")
    for realm_name, analysis in status['realm_analysis'].items():
        if analysis['frequency_count'] > 0:
            print(f"  {realm_name}: {analysis['frequency_count']} frequencies, "
                  f"weight={analysis['total_weight']:.3f}, "
                  f"phase_lock={analysis['phase_locking_factor']:.6f}")
    
    # Validation
    validation = status['validation']
    print(f"\nValidation results:")
    print(f"  F_avg calculation: {validation['f_avg_calculation']}")
    print(f"  P_GCI calculation: {validation['p_gci_calculation']}")
    print(f"  P_GCI range valid: {validation['p_gci_range_valid']}")
    print(f"  Mathematical validation: {validation['mathematical_validation']}")
    
    print("\nGlobal Coherence Index system ready for UBP integration.")
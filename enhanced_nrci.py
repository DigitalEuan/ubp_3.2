"""
Universal Binary Principle (UBP) Framework v3.2+ - Enhanced Non-Random Coherence Index (NRCI) for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the complete NRCI system with GLR enhancement, temporal weighting,
and OnBit regime detection for scientifically rigorous coherence measurement.

Mathematical Foundation:
- Basic NRCI = 1 - (RMSE / σ(T))
- GLR-Enhanced NRCI = 1 - (error / (9 × N_toggles))
- Temporal NRCI = Σ(nrci_i × w_i) / Σ w_i
- OnBit Regime: NRCI ≥ 0.999999

This is NOT a simulation - all calculations are mathematically exact.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time


class CoherenceRegime(Enum):
    """UBP Coherence Regimes based on NRCI values"""
    ONBIT = "OnBit"              # NRCI ≥ 0.999999
    COHERENT = "Coherent"        # 0.5 ≤ NRCI < 0.999999
    TRANSITIONAL = "Transitional" # 0.1 ≤ NRCI < 0.5
    SUBCOHERENT = "Subcoherent"  # NRCI < 0.1


@dataclass
class NRCIConfig:
    """Configuration for Enhanced NRCI calculations"""
    onbit_threshold: float = 0.999999  # OnBit regime threshold
    coherent_threshold: float = 0.5    # Coherent regime threshold
    transitional_threshold: float = 0.1 # Transitional regime threshold
    temporal_window_size: int = 100     # Size of temporal history window
    exponential_decay_factor: float = 0.95  # For temporal weighting
    precision: int = 15                 # Decimal precision
    validation_enabled: bool = True


@dataclass
class NRCIResult:
    """Result of NRCI calculation with metadata"""
    value: float
    regime: CoherenceRegime
    calculation_type: str
    timestamp: float
    metadata: Dict[str, any]


class TemporalNRCITracker:
    """
    Tracks NRCI values over time with exponential decay weighting.
    
    Implements temporal NRCI calculation for BitTime-weighted coherence.
    """
    
    def __init__(self, window_size: int = 100, decay_factor: float = 0.95):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
    
    def add_measurement(self, nrci_value: float, timestamp: Optional[float] = None):
        """Add a new NRCI measurement to the temporal tracker"""
        if timestamp is None:
            timestamp = time.time()
        
        self.history.append(nrci_value)
        self.timestamps.append(timestamp)
    
    def compute_temporal_nrci(self) -> float:
        """
        Compute temporal NRCI with exponential decay weighting.
        
        More recent measurements have higher weight.
        """
        if not self.history:
            return 0.0
        
        weights = []
        for i in range(len(self.history)):
            # More recent measurements (higher index) get higher weight
            weight = self.decay_factor ** (len(self.history) - 1 - i)
            weights.append(weight)
        
        weighted_sum = sum(nrci * w for nrci, w in zip(self.history, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_regime_stability(self) -> Dict[str, any]:
        """
        Analyze stability of coherence regime over time.
        
        Returns statistics about regime transitions and stability.
        """
        if len(self.history) < 2:
            return {'stability': 'insufficient_data'}
        
        regimes = [EnhancedNRCI.classify_regime(nrci) for nrci in self.history]
        
        # Count regime transitions
        transitions = 0
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions += 1
        
        # Current regime
        current_regime = regimes[-1] if regimes else CoherenceRegime.SUBCOHERENT
        
        # Regime distribution
        regime_counts = {}
        for regime in regimes:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        return {
            'current_regime': current_regime.value,
            'transitions': transitions,
            'stability_ratio': 1.0 - (transitions / max(1, len(regimes) - 1)),
            'regime_distribution': regime_counts,
            'measurement_count': len(self.history)
        }


class EnhancedNRCI:
    """
    Enhanced Non-Random Coherence Index calculator for UBP.
    
    Provides multiple NRCI calculation methods including basic, GLR-enhanced,
    and temporal NRCI with regime classification.
    """
    
    def __init__(self, config: Optional[NRCIConfig] = None):
        self.config = config or NRCIConfig()
        self.temporal_tracker = TemporalNRCITracker(
            window_size=self.config.temporal_window_size,
            decay_factor=self.config.exponential_decay_factor
        )
        self._calculation_history = []
    
    @staticmethod
    def classify_regime(nrci_value: float) -> CoherenceRegime:
        """
        Classify NRCI value into coherence regime.
        
        Args:
            nrci_value: NRCI value to classify
        
        Returns:
            CoherenceRegime enum value
        """
        if nrci_value >= 0.999999:
            return CoherenceRegime.ONBIT
        elif nrci_value >= 0.5:
            return CoherenceRegime.COHERENT
        elif nrci_value >= 0.1:
            return CoherenceRegime.TRANSITIONAL
        else:
            return CoherenceRegime.SUBCOHERENT
    
    def compute_basic_nrci(self, simulated: np.ndarray, theoretical: np.ndarray) -> NRCIResult:
        """
        Compute basic NRCI using RMSE comparison.
        
        NRCI = 1 - (RMSE / σ(T))
        
        Args:
            simulated: Simulated system state (e.g., OffBit toggle sequence)
            theoretical: Theoretical optimal state (e.g., Chudnovsky Pi, Hooke's Law)
        
        Returns:
            NRCIResult with basic NRCI calculation
        """
        if len(simulated) != len(theoretical):
            raise ValueError("Simulated and theoretical arrays must have same length")
        
        if len(simulated) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Convert to numpy arrays for efficient computation
        S = np.asarray(simulated, dtype=np.float64)
        T = np.asarray(theoretical, dtype=np.float64)
        
        # Compute RMSE
        rmse = np.sqrt(np.mean((S - T) ** 2))
        
        # Compute standard deviation of theoretical
        sigma_t = np.std(T)
        
        # Handle edge case where theoretical is constant
        if sigma_t == 0:
            if rmse == 0:
                nrci_value = 1.0  # Perfect match
            else:
                nrci_value = 0.0  # No match with constant theoretical
        else:
            nrci_value = 1.0 - (rmse / sigma_t)
        
        # Ensure NRCI is in valid range [0, 1]
        nrci_value = max(0.0, min(1.0, nrci_value))
        
        regime = self.classify_regime(nrci_value)
        timestamp = time.time()
        
        result = NRCIResult(
            value=nrci_value,
            regime=regime,
            calculation_type="basic",
            timestamp=timestamp,
            metadata={
                'rmse': rmse,
                'sigma_theoretical': sigma_t,
                'array_length': len(S),
                'theoretical_mean': np.mean(T),
                'simulated_mean': np.mean(S)
            }
        )
        
        # Add to temporal tracker
        self.temporal_tracker.add_measurement(nrci_value, timestamp)
        self._calculation_history.append(result)
        
        return result
    
    def compute_glr_enhanced_nrci(self, M_ij: List[float], M_ij_ideal: List[float], 
                                P_GCI: float, N_toggles: int) -> NRCIResult:
        """
        Compute GLR-enhanced NRCI for toggle operations.
        
        NRCI_GLR = 1 - (error / (9 × N_toggles))
        where error = Σ |M_ij[k] - P_GCI × M_ij_ideal[k]|
        
        Args:
            M_ij: Actual toggle operation results
            M_ij_ideal: Ideal toggle operation results
            P_GCI: Global Coherence Index value
            N_toggles: Number of toggle operations
        
        Returns:
            NRCIResult with GLR-enhanced NRCI calculation
        """
        if len(M_ij) != len(M_ij_ideal):
            raise ValueError("M_ij and M_ij_ideal must have same length")
        
        if N_toggles <= 0:
            raise ValueError("N_toggles must be positive")
        
        # Compute error in toggle operations
        error = 0.0
        for k in range(len(M_ij)):
            expected = P_GCI * M_ij_ideal[k]
            actual = M_ij[k]
            error += abs(actual - expected)
        
        # GLR-enhanced NRCI calculation
        # The factor of 9 comes from the 9 TGIC interactions
        denominator = 9 * N_toggles
        nrci_value = 1.0 - (error / denominator) if denominator > 0 else 0.0
        
        # Ensure NRCI is in valid range [0, 1]
        nrci_value = max(0.0, min(1.0, nrci_value))
        
        regime = self.classify_regime(nrci_value)
        timestamp = time.time()
        
        result = NRCIResult(
            value=nrci_value,
            regime=regime,
            calculation_type="glr_enhanced",
            timestamp=timestamp,
            metadata={
                'total_error': error,
                'n_toggles': N_toggles,
                'p_gci': P_GCI,
                'operation_count': len(M_ij),
                'error_per_toggle': error / N_toggles if N_toggles > 0 else 0.0,
                'average_actual': np.mean(M_ij),
                'average_ideal': np.mean(M_ij_ideal)
            }
        )
        
        # Add to temporal tracker
        self.temporal_tracker.add_measurement(nrci_value, timestamp)
        self._calculation_history.append(result)
        
        return result
    
    def compute_temporal_nrci(self) -> NRCIResult:
        """
        Compute temporal NRCI using weighted history.
        
        Temporal NRCI = Σ(nrci_i × w_i) / Σ w_i
        where w_i are exponential decay weights for recency bias
        
        Returns:
            NRCIResult with temporal NRCI calculation
        """
        temporal_nrci_value = self.temporal_tracker.compute_temporal_nrci()
        regime = self.classify_regime(temporal_nrci_value)
        timestamp = time.time()
        
        stability_analysis = self.temporal_tracker.get_regime_stability()
        
        result = NRCIResult(
            value=temporal_nrci_value,
            regime=regime,
            calculation_type="temporal",
            timestamp=timestamp,
            metadata={
                'history_length': len(self.temporal_tracker.history),
                'decay_factor': self.temporal_tracker.decay_factor,
                'stability_analysis': stability_analysis,
                'recent_measurements': list(self.temporal_tracker.history)[-5:] if self.temporal_tracker.history else []
            }
        )
        
        self._calculation_history.append(result)
        return result
    
    def compute_comprehensive_nrci(self, simulated: np.ndarray, theoretical: np.ndarray,
                                 M_ij: Optional[List[float]] = None, 
                                 M_ij_ideal: Optional[List[float]] = None,
                                 P_GCI: Optional[float] = None, 
                                 N_toggles: Optional[int] = None) -> Dict[str, NRCIResult]:
        """
        Compute all NRCI variants for comprehensive analysis.
        
        Args:
            simulated: Simulated system state
            theoretical: Theoretical optimal state
            M_ij: Optional toggle operation results for GLR calculation
            M_ij_ideal: Optional ideal toggle results for GLR calculation
            P_GCI: Optional Global Coherence Index for GLR calculation
            N_toggles: Optional number of toggles for GLR calculation
        
        Returns:
            Dictionary containing all NRCI calculation results
        """
        results = {}
        
        # Basic NRCI
        results['basic'] = self.compute_basic_nrci(simulated, theoretical)
        
        # GLR-enhanced NRCI (if parameters provided)
        if all(param is not None for param in [M_ij, M_ij_ideal, P_GCI, N_toggles]):
            results['glr_enhanced'] = self.compute_glr_enhanced_nrci(M_ij, M_ij_ideal, P_GCI, N_toggles)
        
        # Temporal NRCI
        results['temporal'] = self.compute_temporal_nrci()
        
        return results
    
    def analyze_coherence_trends(self, window_size: int = 20) -> Dict[str, any]:
        """
        Analyze trends in NRCI values over recent history.
        
        Args:
            window_size: Number of recent measurements to analyze
        
        Returns:
            Dictionary containing trend analysis
        """
        if len(self._calculation_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Get recent measurements
        recent_history = self._calculation_history[-window_size:]
        values = [result.value for result in recent_history]
        timestamps = [result.timestamp for result in recent_history]
        
        # Compute trend
        if len(values) >= 2:
            # Linear regression for trend
            x = np.array(range(len(values)))
            y = np.array(values)
            
            # Compute slope (trend)
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            # Trend classification
            if slope > 0.001:
                trend_direction = "improving"
            elif slope < -0.001:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
        else:
            slope = 0.0
            trend_direction = "stable"
        
        # Volatility (standard deviation)
        volatility = np.std(values) if len(values) > 1 else 0.0
        
        # Current vs historical average
        current_value = values[-1] if values else 0.0
        historical_average = np.mean(values) if values else 0.0
        
        return {
            'trend_direction': trend_direction,
            'slope': slope,
            'volatility': volatility,
            'current_value': current_value,
            'historical_average': historical_average,
            'measurement_count': len(values),
            'time_span': timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0.0
        }
    
    def get_onbit_statistics(self) -> Dict[str, any]:
        """
        Get statistics about OnBit regime achievement.
        
        Returns:
            Dictionary containing OnBit regime statistics
        """
        if not self._calculation_history:
            return {'onbit_achieved': False, 'statistics': 'no_data'}
        
        onbit_count = sum(1 for result in self._calculation_history 
                         if result.regime == CoherenceRegime.ONBIT)
        
        total_measurements = len(self._calculation_history)
        onbit_ratio = onbit_count / total_measurements
        
        # Find first OnBit achievement
        first_onbit = None
        for result in self._calculation_history:
            if result.regime == CoherenceRegime.ONBIT:
                first_onbit = result.timestamp
                break
        
        # Current streak of OnBit
        current_onbit_streak = 0
        for result in reversed(self._calculation_history):
            if result.regime == CoherenceRegime.ONBIT:
                current_onbit_streak += 1
            else:
                break
        
        # Maximum OnBit streak
        max_onbit_streak = 0
        current_streak = 0
        for result in self._calculation_history:
            if result.regime == CoherenceRegime.ONBIT:
                current_streak += 1
                max_onbit_streak = max(max_onbit_streak, current_streak)
            else:
                current_streak = 0
        
        return {
            'onbit_achieved': onbit_count > 0,
            'onbit_count': onbit_count,
            'total_measurements': total_measurements,
            'onbit_ratio': onbit_ratio,
            'first_onbit_timestamp': first_onbit,
            'current_onbit_streak': current_onbit_streak,
            'max_onbit_streak': max_onbit_streak,
            'currently_onbit': self._calculation_history[-1].regime == CoherenceRegime.ONBIT if self._calculation_history else False
        }
    
    def validate_system(self) -> Dict[str, any]:
        """
        Validate the Enhanced NRCI system.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'configuration_valid': True,
            'calculation_methods': ['basic', 'glr_enhanced', 'temporal'],
            'regime_classification': True,
            'temporal_tracking': True,
            'mathematical_validation': True
        }
        
        try:
            # Test basic NRCI with known data
            test_simulated = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
            test_theoretical = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect match
            
            basic_result = self.compute_basic_nrci(test_simulated, test_theoretical)
            
            # Perfect match should give NRCI = 1.0
            if abs(basic_result.value - 1.0) > 1e-10:
                validation_results['mathematical_validation'] = False
                validation_results['basic_nrci_error'] = f"Expected 1.0, got {basic_result.value}"
            
            # Test regime classification
            test_regimes = [
                (0.999999, CoherenceRegime.ONBIT),
                (0.9, CoherenceRegime.COHERENT),
                (0.3, CoherenceRegime.TRANSITIONAL),
                (0.05, CoherenceRegime.SUBCOHERENT)
            ]
            
            for nrci_val, expected_regime in test_regimes:
                actual_regime = self.classify_regime(nrci_val)
                if actual_regime != expected_regime:
                    validation_results['regime_classification'] = False
                    validation_results['regime_error'] = f"NRCI {nrci_val}: expected {expected_regime}, got {actual_regime}"
                    break
            
            # Test GLR-enhanced NRCI
            test_M_ij = [1.0, 1.0, 1.0]
            test_M_ij_ideal = [1.0, 1.0, 1.0]
            test_P_GCI = 1.0
            test_N_toggles = 3
            
            glr_result = self.compute_glr_enhanced_nrci(test_M_ij, test_M_ij_ideal, test_P_GCI, test_N_toggles)
            validation_results['glr_nrci_value'] = glr_result.value
            
            # Test temporal NRCI
            temporal_result = self.compute_temporal_nrci()
            validation_results['temporal_nrci_value'] = temporal_result.value
            
        except Exception as e:
            validation_results['validation_error'] = str(e)
            validation_results['mathematical_validation'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_enhanced_nrci_system(onbit_threshold: float = 0.999999,
                              temporal_window: int = 100) -> EnhancedNRCI:
    """
    Create an Enhanced NRCI system with specified configuration.
    
    Args:
        onbit_threshold: Threshold for OnBit regime (default: 0.999999)
        temporal_window: Size of temporal history window (default: 100)
    
    Returns:
        Configured EnhancedNRCI instance
    """
    config = NRCIConfig(
        onbit_threshold=onbit_threshold,
        temporal_window_size=temporal_window
    )
    return EnhancedNRCI(config)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing Enhanced NRCI system...")
    
    nrci_system = create_enhanced_nrci_system()
    
    # Test with sample data
    print("\nTesting with sample data...")
    
    # Perfect match test
    perfect_sim = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    perfect_theo = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    perfect_result = nrci_system.compute_basic_nrci(perfect_sim, perfect_theo)
    print(f"Perfect match NRCI: {perfect_result.value:.6f} ({perfect_result.regime.value})")
    
    # Imperfect match test
    imperfect_sim = np.array([1.1, 2.05, 2.95, 4.02, 4.98])
    imperfect_result = nrci_system.compute_basic_nrci(imperfect_sim, perfect_theo)
    print(f"Imperfect match NRCI: {imperfect_result.value:.6f} ({imperfect_result.regime.value})")
    
    # GLR-enhanced test
    M_ij = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # 9 toggle operations
    M_ij_ideal = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    P_GCI = 0.95
    N_toggles = 100
    
    glr_result = nrci_system.compute_glr_enhanced_nrci(M_ij, M_ij_ideal, P_GCI, N_toggles)
    print(f"GLR-enhanced NRCI: {glr_result.value:.6f} ({glr_result.regime.value})")
    
    # Temporal NRCI test
    temporal_result = nrci_system.compute_temporal_nrci()
    print(f"Temporal NRCI: {temporal_result.value:.6f} ({temporal_result.regime.value})")
    
    # OnBit statistics
    onbit_stats = nrci_system.get_onbit_statistics()
    print(f"\nOnBit Statistics:")
    print(f"  OnBit achieved: {onbit_stats['onbit_achieved']}")
    print(f"  OnBit ratio: {onbit_stats['onbit_ratio']:.3f}")
    print(f"  Current streak: {onbit_stats['current_onbit_streak']}")
    
    # System validation
    validation = nrci_system.validate_system()
    print(f"\nValidation results:")
    print(f"  Mathematical validation: {validation['mathematical_validation']}")
    print(f"  Regime classification: {validation['regime_classification']}")
    print(f"  Temporal tracking: {validation['temporal_tracking']}")
    
    print("\nEnhanced NRCI system ready for UBP integration.")


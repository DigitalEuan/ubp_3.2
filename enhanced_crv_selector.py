"""
Universal Binary Principle (UBP) Framework v3.2+ - Enhanced CRV Database with Sub-CRVs
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

# Import the centralized UBPConfig and related data structures
from ubp_config import get_config, UBPConfig, RealmConfig
# Import CRVProfile and SubCRV from crv_database, as they are used by EnhancedCRVDatabase
from crv_database import EnhancedCRVDatabase, CRVProfile, SubCRV

@dataclass
class CRVSelectionResult:
    """Result of an optimal CRV selection."""
    selected_crv: float
    reason: str
    nrci_predicted: float
    compute_time_predicted: float
    confidence_score: float
    fallback_crvs_attempted: int = 0
    optimization_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            'selected_crv': self.selected_crv,
            'reason': self.reason,
            'nrci_predicted': self.nrci_predicted,
            'compute_time_predicted': self.compute_time_predicted,
            'confidence_score': self.confidence_score,
            'fallback_crvs_attempted': self.fallback_crvs_attempted,
            'optimization_notes': self.optimization_notes
        }

class AdaptiveCRVSelector:
    """
    Selects the optimal CRV based on dynamic data characteristics and system performance.
    Utilizes heuristics and historical data for adaptive optimization.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: UBPConfig = get_config()
        self.crv_database = EnhancedCRVDatabase() # This instance will now load profiles from UBPConfig
        self.performance_monitor = CRVPerformanceMonitor()
        self.harmonic_analyzer = HarmonicPatternAnalyzer()

    def select_optimal_crv(self, realm_name: str, data_characteristics: Dict[str, Any],
                           performance_history: Optional[Dict[str, Any]] = None) -> CRVSelectionResult:
        """
        Selects the optimal CRV for a given realm and data.

        Args:
            realm_name: The name of the realm (e.g., 'electromagnetic').
            data_characteristics: Dictionary of data properties (e.g., 'frequency', 'complexity', 'noise_level').
            performance_history: Optional historical performance data for fine-tuning.

        Returns:
            A CRVSelectionResult object.
        """
        self.logger.info(f"Initiating optimal CRV selection for realm: {realm_name}")
        
        realm_config = self.config.get_realm_config(realm_name)
        if not realm_config:
            self.logger.error(f"Realm '{realm_name}' not found in UBPConfig.")
            # Fallback to default if realm not configured
            default_realm_config = self.config.get_realm_config(self.config.default_realm)
            if default_realm_config:
                realm_name = self.config.default_realm
                realm_config = default_realm_config
                self.logger.info(f"Falling back to default realm: {realm_name}")
            else:
                raise ValueError(f"Realm '{realm_name}' not found and no default realm configured.")

        # Use EnhancedCRVDatabase's selection logic
        crv_selection_tuple = self.crv_database.get_optimal_crv(realm_name, data_characteristics)
        self.logger.debug(f"DEBUG(CRV_Selector): get_optimal_crv returned: {crv_selection_tuple}")
        if crv_selection_tuple is None:
            self.logger.error(f"Failed to get optimal CRV for {realm_name}.")
            raise ValueError(f"Could not select optimal CRV for realm {realm_name}.")
        
        selected_crv, reason = crv_selection_tuple

        # Predict Non-Random Coherence Index (NRCI) and compute time (simplified for example)
        predicted_nrci = self.performance_monitor.predict_nrci(realm_name, data_characteristics, selected_crv)
        predicted_compute_time = self.performance_monitor.predict_compute_time(realm_name, data_characteristics, selected_crv)
        
        # Confidence score based on various factors
        confidence = self._calculate_confidence(realm_name, selected_crv, data_characteristics, predicted_nrci, performance_history)

        self.logger.info(f"Optimal CRV selected: {selected_crv:.6e} for {realm_name} (Reason: {reason})")

        return CRVSelectionResult(
            selected_crv=selected_crv,
            reason=reason,
            nrci_predicted=predicted_nrci,
            compute_time_predicted=predicted_compute_time,
            confidence_score=confidence,
            fallback_crvs_attempted=0, # This would be populated by a more complex fallback system
            optimization_notes="Dynamic selection based on data characteristics."
        )

    def _calculate_confidence(self, realm_name: str, crv: float, data_characteristics: Dict,
                              predicted_nrci: float, performance_history: Optional[Dict]) -> float:
        """Calculates a confidence score for the selected CRV."""
        confidence = 0.5 # Base confidence
        
        config_crv_config = self.config.crv # Get the CRV-specific configuration values

        # Frequency match boost
        data_freq = data_characteristics.get('frequency', 0)
        if data_freq > 0:
            if abs(crv - data_freq) / max(crv, data_freq) < config_crv_config.crv_match_tolerance:
                confidence += config_crv_config.confidence_freq_boost
        
        # Noise level penalty/boost
        noise_level = data_characteristics.get('noise_level', 0.1)
        confidence += (1.0 - noise_level) * config_crv_config.confidence_noise_boost
        
        # Historical performance boost
        if performance_history and realm_name in performance_history:
            avg_nrci = performance_history[realm_name].get('avg_nrci', 0)
            if predicted_nrci >= avg_nrci: # If predicted NRCI is better or equal to historical average
                confidence += config_crv_config.confidence_historical_perf_boost
        
        # Clamp confidence between 0 and 1
        return max(0.0, min(1.0, confidence))

class CRVPerformanceMonitor:
    """Monitors CRV performance and predicts metrics like NRCI and compute time."""
    def __init__(self):
        self.config: UBPConfig = get_config()
        # In a real system, this would load/store historical data
        self.historical_data: Dict[str, Dict] = {}

    def record_performance(self, realm_name: str, crv_used: float, nrci_actual: float,
                           compute_time_actual: float, toggle_count: int):
        """Records actual performance metrics."""
        if realm_name not in self.historical_data:
            self.historical_data[realm_name] = {'nrci_history': [], 'compute_time_history': [], 'toggle_count_history': []}
        
        self.historical_data[realm_name]['nrci_history'].append(nrci_actual)
        self.historical_data[realm_name]['compute_time_history'].append(compute_time_actual)
        self.historical_data[realm_name]['toggle_count_history'].append(toggle_count)
        # Keep history manageable, e.g., last 100 entries
        self.historical_data[realm_name]['nrci_history'] = self.historical_data[realm_name]['nrci_history'][-100:]
        self.historical_data[realm_name]['compute_time_history'] = self.historical_data[realm_name]['compute_time_history'][-100:]
        self.historical_data[realm_name]['toggle_count_history'] = self.historical_data[realm_name]['toggle_count_history'][-100:]

    def predict_nrci(self, realm_name: str, data_characteristics: Dict, crv: float) -> float:
        """Predicts the NRCI score for a given CRV and data."""
        realm_cfg = self.config.get_realm_config(realm_name)
        base_nrci = realm_cfg.nrci_baseline if realm_cfg else 0.9 # Default if realm_cfg not found
        
        # Simple prediction: adjust based on complexity and noise
        complexity_factor = data_characteristics.get('complexity', 0.5) * self.config.crv.prediction_complexity_factor
        noise_factor = data_characteristics.get('noise_level', 0.1) * self.config.crv.prediction_noise_factor
        
        predicted = base_nrci - complexity_factor - noise_factor
        return max(0.0, min(1.0, predicted)) # Clamp between 0 and 1

    def predict_compute_time(self, realm_name: str, data_characteristics: Dict, crv: float) -> float:
        """Predicts the computation time for a given CRV and data."""
        base_compute_time = self.config.crv.prediction_base_computation_time
        
        # Simple prediction: adjust based on complexity
        complexity_adjustment = data_characteristics.get('complexity', 0.5) * 0.00001
        
        predicted = base_compute_time + complexity_adjustment
        return max(0.0, predicted)

class HarmonicPatternAnalyzer:
    """Analyzes data for harmonic patterns to inform CRV selection."""
    def __init__(self):
        self.config: UBPConfig = get_config()

    def analyze_harmonics(self, data_series: List[float]) -> Dict[str, Any]:
        """
        Performs a simplified harmonic analysis on a data series.
        In a real scenario, this would involve FFT, wavelet analysis etc.
        """
        if not data_series:
            return {"primary_frequency": 0, "harmonic_peaks": [], "noise_signature": 0}
        
        # Simple peak detection (placeholder)
        primary_freq = max(data_series) # Simplistic: highest value indicates primary freq
        
        # Identify "harmonic" peaks (e.g., multiples or fractions)
        harmonic_peaks = []
        for freq in data_series:
            # Check for simple 2x, 3x, 0.5x, 0.33x harmonics
            if primary_freq > 0:
                ratio = freq / primary_freq
                if abs(ratio - round(ratio)) < self.config.crv.harmonic_ratio_tolerance: # Close to an integer multiple
                    harmonic_peaks.append({"frequency": freq, "ratio_to_primary": round(ratio), "type": "harmonic"})
                elif abs(ratio - 1/round(1/ratio)) < self.config.crv.harmonic_ratio_tolerance and round(1/ratio) <= self.config.crv.harmonic_fraction_denominator_limit: # Close to a simple fraction
                    harmonic_peaks.append({"frequency": freq, "ratio_to_primary": f"1/{round(1/ratio)}", "type": "subharmonic"})
        
        # Simple noise estimation
        noise_signature = sum(abs(x - primary_freq) for x in data_series) / len(data_series)
        
        return {
            "primary_frequency": primary_freq,
            "harmonic_peaks": harmonic_peaks,
            "noise_signature": noise_signature
        }

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Test AdaptiveCRVSelector
    selector = AdaptiveCRVSelector()
    
    data_chars = {
        'frequency': 3.14,
        'complexity': 0.7,
        'noise_level': 0.05,
        'target_nrci': 0.99,
        'speed_priority': True
    }
    
    # Test a known realm
    result = selector.select_optimal_crv('electromagnetic', data_chars)
    print(f"\nSelected CRV: {result.selected_crv:.6e}")
    print(f"Reason: {result.reason}")
    print(f"Predicted NRCI: {result.nrci_predicted:.4f}")
    print(f"Predicted Compute Time: {result.compute_time_predicted:.6f}s")
    print(f"Confidence: {result.confidence_score:.2f}")

    # Test an unknown realm (should fall back to default)
    print("\nAttempting selection for 'unknown_realm'...")
    try:
        result_unknown = selector.select_optimal_crv('unknown_realm', data_chars)
        print(f"Selected CRV for unknown: {result_unknown.selected_crv:.6e}")
    except ValueError as e:
        print(f"Error selecting CRV for unknown realm: {e}")
        
    # Test HarmonicPatternAnalyzer
    analyzer = HarmonicPatternAnalyzer()
    sample_data = [100.0, 200.0, 50.0, 150.0, 300.0, 10.0, 101.5]
    harmonic_analysis = analyzer.analyze_harmonics(sample_data)
    print(f"\nHarmonic Analysis of Sample Data: {harmonic_analysis}")
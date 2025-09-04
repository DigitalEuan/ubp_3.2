"""
Universal Binary Principle (UBP) Framework v3.2+ - CRV Database with Sub-CRVs
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

This module contains the refined Core Resonance Values (CRVs) with Sub-CRV fallback systems
based on frequency scanning research and harmonic pattern analysis.

Updated to pull CRV definitions dynamically from ubp_config.py.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import logging

# Import the centralized UBPConfig
from ubp_config import get_config, UBPConfig, RealmConfig

@dataclass
class SubCRV:
    """Sub-CRV with performance metrics and harmonic relationship."""
    frequency: float
    nrci_score: float
    compute_time: float
    toggle_count: int
    harmonic_type: str  # e.g., "2x_harmonic", "0.5x_subharmonic", "fundamental"
    confidence: float
    
@dataclass
class CRVProfile:
    """Complete CRV profile with main CRV and Sub-CRV fallbacks."""
    realm: str
    main_crv: float
    wavelength: float  # nm
    platonic_solid: str # Changed from 'geometry' to 'platonic_solid'
    coordination_number: int
    sub_crvs: List[SubCRV]
    nrci_baseline: float
    optimization_notes: str

class EnhancedCRVDatabase:
    """
    Enhanced CRV Database with Sub-CRV fallback system and adaptive selection.
    
    Based on frequency scanning research showing harmonic patterns in each realm
    with specific Sub-CRVs that provide optimization pathways for different
    data characteristics and computational requirements.
    """
    
    # Constant for scaling compute time in fitness evaluation
    COMPUTE_TIME_SCALING_FACTOR = 50000
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: UBPConfig = get_config() # Get the global UBPConfig instance
        self.crv_profiles = self._initialize_crv_profiles()
        self.performance_history = {} # Placeholder for actual history management
        
    def _initialize_crv_profiles(self) -> Dict[str, CRVProfile]:
        """
        Initialize CRV profiles by pulling data from UBPConfig's realm definitions.
        """
        profiles = {}
        for realm_name, realm_cfg in self.config.realms.items():
            self.logger.debug(f"DEBUG(CRV_DB): Initializing profile for realm '{realm_name}' with RealmConfig: {realm_cfg}")
            self.logger.debug(f"DEBUG(CRV_DB): RealmConfig '{realm_name}' has platonic_solid attribute: {hasattr(realm_cfg, 'platonic_solid')}")
            # Convert the list of sub_crvs (floats) from UBPConfig to SubCRV objects
            # This is a simplification; a full SubCRV definition from research might be more complex
            # For now, create placeholder SubCRV objects based on frequencies from ubp_config
            sub_crv_objects = []
            if realm_cfg.sub_crvs:
                for i, freq in enumerate(realm_cfg.sub_crvs):
                    # These values are placeholders and should ideally come from research data
                    # or config for each specific sub-CRV.
                    # For now, derive simple harmonic_type based on relation to main_crv
                    harmonic_type = "sub_crv_dynamic"
                    if realm_cfg.main_crv > 0:
                        ratio = freq / realm_cfg.main_crv
                        if abs(ratio - 0.5) < 0.01: harmonic_type = "0.5x_subharmonic"
                        elif abs(ratio - 2.0) < 0.01: harmonic_type = "2x_harmonic"
                        elif abs(ratio - 1.0) < 0.01: harmonic_type = "fundamental"
                        elif ratio < 1.0: harmonic_type = f"{ratio:.2f}x_subharmonic"
                        elif ratio > 1.0: harmonic_type = f"{ratio:.2f}x_harmonic"

                    sub_crv_objects.append(SubCRV(
                        frequency=freq,
                        nrci_score=0.99 - (i * 0.01), # Placeholder
                        compute_time=0.000015 + (i * 0.000001), # Placeholder
                        toggle_count=1180 - (i * 5), # Placeholder
                        harmonic_type=harmonic_type,
                        confidence=0.95 - (i * 0.01) # Placeholder
                    ))

            profiles[realm_name] = CRVProfile(
                realm=realm_cfg.name,
                main_crv=realm_cfg.main_crv,
                wavelength=realm_cfg.wavelength,
                platonic_solid=realm_cfg.platonic_solid, # Changed to platonic_solid
                coordination_number=realm_cfg.coordination_number,
                sub_crvs=sub_crv_objects,
                nrci_baseline=realm_cfg.nrci_baseline,
                optimization_notes=f"Loaded from UBPConfig for {realm_cfg.name} realm"
            )
        self.logger.info(f"Initialized {len(profiles)} CRV profiles from UBPConfig.")
        return profiles
    
    def get_crv_profile(self, realm: str) -> Optional[CRVProfile]:
        """Get complete CRV profile for a realm."""
        return self.crv_profiles.get(realm.lower())
    
    def get_optimal_crv(self, realm: str, data_characteristics: Dict) -> Optional[Tuple[float, str]]:
        """
        Select optimal CRV based on data characteristics.
        
        Args:
            realm: Target realm name
            data_characteristics: Dict with keys like 'frequency', 'complexity', 'noise_level'
            
        Returns:
            Tuple of (optimal_crv_frequency, selection_reason) or None if realm unknown.
        """
        profile = self.get_crv_profile(realm)
        if not profile:
            self.logger.warning(f"Unknown realm: {realm}")
            return None # Changed return to None to match type hint and indicate failure
        
        # Extract data characteristics
        data_freq = data_characteristics.get('frequency', 0)
        complexity = data_characteristics.get('complexity', 0.5)
        noise_level = data_characteristics.get('noise_level', 0.1)
        target_nrci = data_characteristics.get('target_nrci', self.config.performance.TARGET_NRCI) # Use config's target NRCI
        
        # Start with main CRV
        best_crv = profile.main_crv
        best_score = 0.0
        best_reason = "main_crv_default"
        
        # Evaluate main CRV
        main_score = self._evaluate_crv_fitness(profile.main_crv, data_characteristics, profile)
        if main_score > best_score:
            best_crv = profile.main_crv
            best_score = main_score
            best_reason = "main_crv_optimal"
        
        # Evaluate Sub-CRVs
        for sub_crv in profile.sub_crvs:
            score = self._evaluate_crv_fitness(sub_crv.frequency, data_characteristics, profile, sub_crv)
            
            # Bonus for high NRCI Sub-CRVs
            if sub_crv.nrci_score >= target_nrci:
                score += 0.1
            
            # Bonus for low compute time if speed is priority
            if data_characteristics.get('speed_priority', False) and sub_crv.compute_time < self.config.crv.prediction_base_computation_time: # Use config's prediction base compute time
                score += 0.05
            
            if score > best_score:
                best_crv = sub_crv.frequency
                best_score = score
                best_reason = f"sub_crv_{sub_crv.harmonic_type}"
        
        # Log selection
        self.logger.info(f"Selected CRV {best_crv:.6e} for {realm} (reason: {best_reason}, score: {best_score:.3f})")
        
        return best_crv, best_reason
    
    def _evaluate_crv_fitness(self, crv_freq: float, data_chars: Dict, profile: CRVProfile, sub_crv: Optional[SubCRV] = None) -> float:
        """Evaluate how well a CRV matches the data characteristics."""
        score = 0.0
        
        config_crv = self.config.crv # Get CRV specific config parameters for weights
        
        # Frequency matching (weighted from config)
        data_freq = data_chars.get('frequency', 0)
        if data_freq > 0:
            freq_ratio = min(crv_freq, data_freq) / max(crv_freq, data_freq)
            score += config_crv.score_weights_frequency * freq_ratio
        else:
            score += config_crv.score_weights_frequency * 0.5 # Neutral score if no frequency info
        
        # Complexity matching (weighted from config)
        complexity = data_chars.get('complexity', 0.5)
        if sub_crv:
            complexity_match = min(1.0, sub_crv.nrci_score + complexity * 0.1)
            score += config_crv.score_weights_complexity * complexity_match
        else:
            score += config_crv.score_weights_complexity * profile.nrci_baseline
        
        # Noise tolerance (weighted from config)
        noise_level = data_chars.get('noise_level', 0.1)
        if sub_crv:
            noise_tolerance = sub_crv.confidence * (1.0 - noise_level)
            score += config_crv.score_weights_noise * noise_tolerance
        else:
            score += config_crv.score_weights_noise * (1.0 - noise_level)
        
        # Performance considerations (weighted from config)
        if sub_crv:
            perf_score = (sub_crv.nrci_score * 0.7) + ((1.0 - min(1.0, sub_crv.compute_time * self.COMPUTE_TIME_SCALING_FACTOR)) * 0.3)
            score += config_crv.score_weights_performance * perf_score
        else:
            score += config_crv.score_weights_performance * profile.nrci_baseline
        
        return score
    
    def get_harmonic_crvs(self, realm: str, base_frequency: float, max_harmonics: int = 5) -> List[float]:
        """Generate harmonic CRVs based on a base frequency."""
        harmonics = []
        
        # Fundamental and subharmonics
        for i in range(1, max_harmonics + 1):
            harmonics.append(base_frequency / i)  # Subharmonics
            if i > 1:
                harmonics.append(base_frequency * i)  # Harmonics
        
        return sorted(harmonics)
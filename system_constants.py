"""
Universal Binary Principle (UBP) Framework v3.2+ - System Constants
Author: Euan Craig, New Zealand
Date: 03 September 2025
======================================

This module defines all fundamental constants used across the UBP Framework.
This ensures a single, consistent source of truth for physical, mathematical,
and UBP-specific parameters.
"""

import numpy as np
import math # Import math for PI, E in case np is not used directly
from typing import Tuple, Dict, List # Add Dict, List for frequency weights


class UBPConstants:
    """
    Collection of universal, mathematical, and UBP-specific constants.
    All values are defined here for consistency across the framework.
    """

    # --- Universal Physical Constants ---
    # These constants are derived from fundamental physics and are used across all realms.
    SPEED_OF_LIGHT: float = 299792458  # meters per second (m/s)
    PLANCK_CONSTANT: float = 6.62607015e-34  # Joule-seconds (J⋅s)
    PLANCK_REDUCED: float = 1.054571817e-34 # J⋅s (hbar)
    BOLTZMANN_CONSTANT: float = 1.380649e-23  # Joules per Kelvin (J/K)
    FINE_STRUCTURE_CONSTANT: float = 0.0072973525693  # Dimensionless
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
    AVOGADRO_NUMBER: float = 6.02214076e23  # mol⁻¹
    ELEMENTARY_CHARGE: float = 1.602176634e-19  # Coulombs (C)
    VACUUM_PERMITTIVITY: float = 8.8541878128e-12 # Farads per meter (F/m)
    VACUUM_PERMEABILITY: float = 1.25663706212e-6 # Henries per meter (N/A²)

    ELECTRON_MASS: float = 9.1093837015e-31 # kg
    PROTON_MASS: float = 1.67262192369e-27 # kg
    NEUTRON_MASS: float = 1.67492749804e-27 # kg

    NUCLEAR_MAGNETTON: float = 5.0507837461e-27 # J/T
    PROTON_GYROMAGNETIC: float = 2.6752218744e8 # rad/(s*T)
    NEUTRON_GYROMAGNETIC: float = -1.8324717e8 # rad/(s*T)
    DEUTERON_BINDING_ENERGY: float = 2.224573e6 # eV

    RYDBERG_CONSTANT: float = 1.097373156853967e7 # m⁻¹

    # --- Mathematical Constants ---
    # Fundamental mathematical constants used for various calculations within the framework.
    PI: float = math.pi  # π (Pi)
    E: float = math.e  # e (Euler's number)
    PHI: float = (1 + math.sqrt(5)) / 2  # φ (Golden Ratio)
    EULER_MASCHERONI: float = 0.5772156649  # γ (Euler-Mascheroni constant)

    # --- UBP-Specific Core Values ---
    # These constants define core conceptual and operational parameters unique to the UBP.
    # Core Resonance Values (CRVs) - Reference only; actual values might be dynamically loaded
    # from ubp_config.py or crv_database.py for dynamic management.
    CRV_ELECTROMAGNETIC_BASE: float = PI  # Base for EM realm
    CRV_QUANTUM_BASE: float = E / 12  # Base for Quantum realm
    CRV_GRAVITATIONAL_BASE: float = 160.19  # Empirical, derived from gravitational wave research
    CRV_BIOLOGICAL_BASE: float = 10.0  # Empirical, related to neural frequencies
    CRV_COSMOLOGICAL_BASE: float = PI ** PHI # Empirical, π^φ
    CRV_NUCLEAR_BASE: float = 1.2356e20 # Zitterbewegung frequency
    CRV_OPTICAL_BASE: float = 5.0e14 # 600 nm light frequency

    # Toggle Algebra & Bitfield Parameters
    OFFBIT_DEFAULT_SIZE_BYTES: int = 4  # Each OffBit is typically 32 bits
    BITFIELD_DEFAULT_SPARSITY: float = 0.01
    MAX_BITFIELD_DIMENSIONS: int = 6 # 6D operational space

    # UBP-specific constants for system operation
    C_INFINITY: float = 1.0e+308 # Conceptual maximum speed/information propagation rate
    OFFBIT_ENERGY_UNIT: float = 1.0e-30 # Base energy unit for a single OffBit operation/state
    EPSILON_UBP: float = 1e-18 # Smallest significant UBP value, prevents division by zero in log/etc.
    UBP_ZITTERBEWEGUNG_FREQ: float = 1.2356e20  # Hz, explicitly defined here as it was in constants.py
    MAX_PRIME_DEFAULT: int = 282281 # Prime cutoff for PrimeResonanceCoordinateSystem

    # OffBit counts for different hardware profiles (used by hardware_profiles.py)
    # These values are aligned with memory limitations and performance expectations.
    OFFBITS_4GB_MOBILE: int = 10000       # Memory optimized for mobile
    OFFBITS_RASPBERRY_PI5: int = 100000   # Balanced for RPi5
    OFFBITS_8GB_IMAC: int = 1000000       # High performance desktop
    OFFBITS_GOOGLE_COLAB: int = 2500000   # Optimized for Colab's typical resources
    OFFBITS_KAGGLE: int = 2000000         # Optimized for Kaggle's typical resources
    OFFBITS_HPC: int = 10000000           # High-Performance Computing
    OFFBITS_DEVELOPMENT: int = 10000      # Small for fast testing

    # Bitfield dimension configurations (used by hardware_profiles.py)
    # Dimensions are (X, Y, Z, A, B, C) where X,Y,Z are spatial/primary, A,B,C are conceptual/secondary.
    BITFIELD_6D_FULL: Tuple[int, ...] = (150, 150, 150, 5, 2, 2)    # Large configuration for high-end systems
    BITFIELD_6D_MEDIUM: Tuple[int, ...] = (80, 80, 80, 5, 2, 2)     # Medium configuration for balanced systems
    BITFIELD_6D_SMALL: Tuple[int, ...] = (30, 30, 30, 5, 2, 2)      # Small configuration for memory-constrained systems

    # Harmonic Toggle Resonance (HTR) Parameters
    HTR_DEFAULT_THRESHOLD: float = 0.05  # Threshold for harmonic resonance detection
    HTR_MAX_ITERATIONS: int = 1000
    HTR_GENETIC_POPULATION_SIZE: int = 50
    HTR_GENETIC_GENERATIONS: int = 100

    # Error Correction Parameters
    NRCI_TARGET_HIGH_COHERENCE: float = 0.999999  # Target NRCI for optimal coherence
    NRCI_TARGET_STANDARD: float = 0.9999  # Standard NRCI target
    COHERENCE_THRESHOLD: float = 0.95  # Minimum coherence for stable operations
    GOLAY_CODE_PARAMS: Tuple[int, int] = (23, 12)  # (n, k) for Golay[23,12]
    HAMMING_CODE_PARAMS: Tuple[int, int] = (7, 4)  # (n, k) for Hamming[7,4]
    BCH_CODE_PARAMS: Tuple[int, int] = (31, 21)  # (n, k) for BCH[31,21]
    REED_SOLOMON_DEFAULT_COMPRESSION_RATIO: float = 0.30

    # Temporal Mechanics (BitTime)
    BIT_TIME_UNIT_SECONDS: float = 1e-12  # Base unit of BitTime (picoseconds)
    PLANCK_TIME_SECONDS: float = 5.391247e-44  # Smallest unit of time
    COHERENT_SYNCHRONIZATION_CYCLE_SECONDS: float = 1 / PI  # CSC period
    TAUTFLUENCE_TIME_SECONDS: float = 2.117e-15 # Tautfluence period (empirical)

    # Realm Specific Frequencies / Baselines (Consolidated into a dictionary)
    UBP_REALM_FREQUENCIES: Dict[str, float] = {
        'nuclear': 1.2356e20,
        'optical': 5.0e14,
        'quantum': 4.58e14,
        'electromagnetic': PI, # Matches PI
        'gravitational': 100.0,
        'biological': 10.0,
        'cosmological': 1e-11,
    }

    # Default performance targets
    DEFAULT_TARGET_OPS_PER_SECOND: int = 5000
    DEFAULT_MAX_OPERATION_TIME_SECONDS: float = 1.0
    DEFAULT_VALIDATION_ITERATIONS: int = 1000

    # Directory Naming
    DATA_DIR_NAME: str = "data"
    OUTPUT_DIR_NAME: str = "output"
    TEMP_DIR_NAME: str = "temp"
    CACHE_DIR_NAME: str = "cache"
    LOGS_DIR_NAME: str = "logs"

    # Configuration Defaults for UBPConfig
    UBP_CONFIG_DEFAULT_MEMORY_LIMIT_MB: int = 1000
    UBP_CONFIG_DEFAULT_PARALLEL_PROCESSING: bool = True
    UBP_CONFIG_DEFAULT_GPU_ACCELERATION: bool = False
    UBP_CONFIG_DEFAULT_CACHE_ENABLED: bool = True

    # UBP frequency weights for global coherence (Moved from constants.py)
    UBP_FREQUENCY_WEIGHTS: Dict[float, float] = {
        PI: 0.2,      # π (electromagnetic)
        PHI: 0.2,      # φ (golden ratio)
        4.58e14: 0.35, # Quantum entanglement frequency (e.g., specific molecular transition)
        1e9: 0.1,     # GHz range (e.g., microwave interactions)
        1e15: 0.1,    # Optical range (e.g., visible light interactions)
        1e20: 0.05,   # Zitterbewegung / nuclear frequencies
        58977069.609314: 0.05,  # A specific composite resonance frequency (C / (PI * PHI))
    }

    # UBP toggle probabilities by realm (Moved from constants.py)
    UBP_TOGGLE_PROBABILITIES: Dict[str, float] = {
        'quantum': E / 12,          # UBP_TOGGLE_QUANTUM
        'cosmological': PI ** PHI,  # UBP_TOGGLE_COSMOLOGICAL
        'electromagnetic': PI / 4,
        'gravitational': 1.0 / PI,
        'biological': 1.0 / E,
        'nuclear': 1.0 / PHI,
        'optical': 1.0 / math.sqrt(2)
    }


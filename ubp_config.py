"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Config
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

"""
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import json

# Import system constants to populate ConstantConfig
from system_constants import UBPConstants as RawUBPConstants
# Import HardwareProfileManager for auto-detection
from hardware_profiles import HardwareProfileManager, HardwareProfile # Import HardwareProfile for type hinting

# --- Dataclasses for Configuration Structure ---

@dataclass
class ConstantConfig:
    """Stores fundamental UBP and physical constants."""
    # Populated from system_constants.UBPConstants
    PI: float = RawUBPConstants.PI
    E: float = RawUBPConstants.E
    PHI: float = RawUBPConstants.PHI
    EULER_MASCHERONI: float = RawUBPConstants.EULER_MASCHERONI
    SPEED_OF_LIGHT: float = RawUBPConstants.SPEED_OF_LIGHT
    PLANCK_CONSTANT: float = RawUBPConstants.PLANCK_CONSTANT
    PLANCK_REDUCED: float = RawUBPConstants.PLANCK_REDUCED
    BOLTZMANN_CONSTANT: float = RawUBPConstants.BOLTZMANN_CONSTANT
    FINE_STRUCTURE_CONSTANT: float = RawUBPConstants.FINE_STRUCTURE_CONSTANT
    GRAVITATIONAL_CONSTANT: float = RawUBPConstants.GRAVITATIONAL_CONSTANT
    AVOGADRO_NUMBER: float = RawUBPConstants.AVOGADRO_NUMBER
    ELEMENTARY_CHARGE: float = RawUBPConstants.ELEMENTARY_CHARGE
    VACUUM_PERMITTIVITY: float = RawUBPConstants.VACUUM_PERMITTIVITY
    VACUUM_PERMEABILITY: float = RawUBPConstants.VACUUM_PERMEABILITY
    ELECTRON_MASS: float = RawUBPConstants.ELECTRON_MASS
    PROTON_MASS: float = RawUBPConstants.PROTON_MASS
    NEUTRON_MASS: float = RawUBPConstants.NEUTRON_MASS
    NUCLEAR_MAGNETTON: float = RawUBPConstants.NUCLEAR_MAGNETTON
    PROTON_GYROMAGNETIC: float = RawUBPConstants.PROTON_GYROMAGNETIC
    NEUTRON_GYROMAGNETIC: float = RawUBPConstants.NEUTRON_GYROMAGNETIC
    DEUTERON_BINDING_ENERGY: float = RawUBPConstants.DEUTERON_BINDING_ENERGY
    RYDBERG_CONSTANT: float = RawUBPConstants.RYDBERG_CONSTANT

    # UBP-specific constants
    C_INFINITY: float = RawUBPConstants.C_INFINITY # Conceptual maximum speed/information propagation rate
    OFFBIT_ENERGY_UNIT: float = RawUBPConstants.OFFBIT_ENERGY_UNIT # Base energy unit for a single OffBit operation/state
    UBP_QUANTUM_COHERENCE_UNIT: float = 1.0e-15 # Baseline for quantum coherence
    EPSILON_UBP: float = RawUBPConstants.EPSILON_UBP # Smallest significant UBP value, prevents division by zero in log/etc.
    UBP_ZITTERBEWEGUNG_FREQ: float = RawUBPConstants.UBP_ZITTERBEWEGUNG_FREQ
    PLANCK_TIME_SECONDS: float = RawUBPConstants.PLANCK_TIME_SECONDS # For use in kernels.py etc.
    MAX_PRIME_DEFAULT: int = RawUBPConstants.MAX_PRIME_DEFAULT # Max prime for PrimeResonanceCoordinateSystem

    # Dictionaries moved from RawUBPConstants for direct access (as they were formerly in constants.py)
    UBP_FREQUENCY_WEIGHTS: Dict[float, float] = field(default_factory=lambda: RawUBPConstants.UBP_FREQUENCY_WEIGHTS.copy())
    UBP_TOGGLE_PROBABILITIES: Dict[str, float] = field(default_factory=lambda: RawUBPConstants.UBP_TOGGLE_PROBABILITIES.copy())
    UBP_REALM_FREQUENCIES: Dict[str, float] = field(default_factory=lambda: RawUBPConstants.UBP_REALM_FREQUENCIES.copy())


@dataclass
class PerformanceConfig:
    """Configures performance-related thresholds and targets."""
    TARGET_NRCI: float = 0.999999 # Target Normalized Resonance Coherence Index (0-1)
    COHERENCE_THRESHOLD: float = 0.95 # Minimum coherence for stable operations
    MIN_STABILITY: float = 0.85 # Minimum stability for system integrity
    MAX_ERROR_TOLERANCE: float = 0.001 # Maximum allowable error rate

@dataclass
class TemporalConfig:
    """Configures time-related parameters."""
    COHERENT_SYNCHRONIZATION_CYCLE_PERIOD: float = RawUBPConstants.COHERENT_SYNCHRONIZATION_CYCLE_SECONDS # Corrected to load from RawUBPConstants
    BITTIME_UNIT_DURATION: float = 1.0e-12 # Seconds (picoseconds)
    PLANCK_TIME_SECONDS: float = RawUBPConstants.PLANCK_TIME_SECONDS
    COHERENT_SYNCHRONIZATION_CYCLE_PERIOD_DEFAULT: float = RawUBPConstants.COHERENT_SYNCHRONIZATION_CYCLE_SECONDS

@dataclass
class ObserverConfig:
    """Configures parameters related to the observer/consciousness model."""
    DEFAULT_INTENT_LEVEL: float = 1.0 # Neutral intent
    MIN_INTENT_LEVEL: float = 0.0 # Unfocused
    MAX_INTENT_LEVEL: float = 2.0 # Highly intentional
    OBSERVER_INFLUENCE_FACTOR: float = 0.1 # Multiplier for observer impact

@dataclass
class RealmConfig:
    """
    Configuration for a specific computational realm in the UBP framework.
    Includes fundamental parameters for various "platonic solids" of reality.
    """
    name: str
    platonic_solid: str
    main_crv: float  # Central Resonance Value (Hz or arbitrary unit)
    wavelength: float  # Associated wavelength (e.g., nm for EM for Grav)
    coordination_number: int = 12 # Default, e.g., for FCC lattice
    spatial_coherence: float = 0.99  # Baseline spatial coherence for the realm
    temporal_coherence: float = 0.99  # Baseline temporal coherence for the realm
    nrci_baseline: float = 0.8  # Default NRCI baseline for this realm
    lattice_type: str = "Resonant manifold" # Generic description
    optimization_factor: float = 1.0 # Multiplier for certain optimizations
    sub_crvs: List[float] = field(default_factory=list)
    frequency_range: Tuple[float, float] = (0.0, 0.0)
    geometry: str = field(default="dodecahedron", init=False, repr=False) # Dummy for backward compatibility

@dataclass
class MoleculeConfig:
    """Configuration for molecular simulation in HTR."""
    name: str
    nodes: int
    bond_length: float  # L_0 in meters
    bond_energy: float  # eV
    geometry_type: str
    smiles: Optional[str] = None

# Default factory for molecules to prevent mutable default argument issues
def molecules_default_factory():
    return {
        'propane': MoleculeConfig('propane', 10, 0.154e-9, 4.8, 'alkane', 'CCC'),
        'benzene': MoleculeConfig('benzene', 6, 0.14e-9, 5.0, 'aromatic', 'c1ccccc1'),
        'methane': MoleculeConfig('methane', 5, 0.109e-9, 4.5, 'tetrahedral', 'C'),
        'butane': MoleculeConfig('butane', 13, 0.154e-9, 4.8, 'alkane', 'CCCC')
    }

@dataclass
class EnergyConfig:
    """Configuration for the UBP Energy Equation parameters."""
    R_0_DEFAULT: float = 0.95 # Base resonance strength
    H_T_DEFAULT: float = 0.05 # Tonal entropy
    S_OPT_DEFAULT: float = 0.98 # Structural optimality factor

# Simplified config for CRV, Error Correction, and Bitfield sizing within UBPConfig
@dataclass
class CRVConfig:
    prediction_base_computation_time: float = 0.00001 # Base time in seconds
    prediction_complexity_factor: float = 0.1 # Factor for complexity adjustment
    prediction_noise_factor: float = 0.05 # Factor for noise adjustment
    score_weights_frequency: float = 0.4 # Weight for frequency matching in CRV selection
    score_weights_complexity: float = 0.3 # Weight for complexity matching
    score_weights_noise: float = 0.2 # Weight for noise tolerance
    score_weights_performance: float = 0.1 # Weight for performance
    crv_match_tolerance: float = 0.05 # Tolerance for CRV frequency matching
    confidence_freq_boost: float = 0.2 # Confidence boost for frequency match
    confidence_noise_boost: float = 0.1 # Confidence boost for low noise
    confidence_historical_perf_boost: float = 0.1 # Confidence boost for good historical perf
    harmonic_ratio_tolerance: float = 0.02 # Tolerance for detecting harmonic ratios
    harmonic_fraction_denominator_limit: int = 4 # Max denominator for simple fractional harmonics
    resonance_threshold_default: float = 0.01 # For PrimeResonanceCoordinateSystem

@dataclass
class ErrorCorrectionConfig:
    error_threshold: float = 0.05 # General error threshold for correction
    golay_code: str = "23,12" # (n,k) for Golay code
    bch_code: str = "31,21" # (n,k) for BCH code
    hamming_code: str = "7,4" # (n,k) for Hamming code
    padic_prime: int = 7 # P-adic prime for certain error models
    fibonacci_depth: int = 50 # Depth for Fibonacci encoding/decoding (sufficient for large numbers)
    nrci_base_score: float = 0.9 # Base NRCI for error correction

@dataclass
class BitfieldConfig:
    size_mobile: Tuple[int, int, int, int, int, int] = (10, 10, 10, 1, 1, 1)
    size_raspberry_pi: Tuple[int, int, int, int, int, int] = (20, 20, 20, 2, 2, 1)
    size_local: Tuple[int, int, int, int, int, int] = (50, 50, 50, 5, 2, 2)
    size_colab: Tuple[int, int, int, int, int, int] = (70, 70, 70, 5, 3, 2)
    size_kaggle: Tuple[int, int, int, int, int, int] = (60, 60, 60, 5, 3, 2)
    size_production: Tuple[int, int, int, int, int, int] = (100, 100, 100, 10, 5, 5)

@dataclass
class UBPConfig:
    """
    The main UBP Framework Configuration container.
    Initializes all sub-configurations and realm definitions.
    """
    environment: str = "development" # "development", "production", "testing", "auto"
    
    # Global constants
    constants: ConstantConfig = field(default_factory=ConstantConfig)
    
    # Performance parameters
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Temporal parameters
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    # Observer parameters
    observer: ObserverConfig = field(default_factory=ObserverConfig)

    # Energy parameters
    energy: EnergyConfig = field(default_factory=EnergyConfig) # Added EnergyConfig
    
    # Bitfield Dimensions (6D tuple as specified by UBP design)
    BITFIELD_DIMENSIONS: Tuple[int, int, int, int, int, int] = (10, 10, 10, 10, 10, 10)
    
    # Realm configurations - a dictionary for easy access
    realms: Dict[str, RealmConfig] = field(default_factory=dict)

    # HTR Molecule configurations
    molecules: Dict[str, MoleculeConfig] = field(default_factory=molecules_default_factory)

    # Simplified config for CRV, Error Correction, and Bitfield sizing within UBPConfig
    crv: CRVConfig = field(default_factory=CRVConfig)
    error_correction: ErrorCorrectionConfig = field(default_factory=ErrorCorrectionConfig)
    bitfield: BitfieldConfig = field(default_factory=BitfieldConfig)
    
    default_realm: str = "electromagnetic"


    def __post_init__(self):
        # Define default realms with their specific CRVs and properties.
        self._initialize_default_realms()
        self.apply_environment_settings()

    def _initialize_default_realms(self):
        """Initializes the predefined computational realms."""
        
        # Helper for wavelength calculation, using SPEED_OF_LIGHT from constants
        def calculate_wavelength(freq_hz):
            if freq_hz > 0:
                return RawUBPConstants.SPEED_OF_LIGHT / freq_hz
            return 0.0 # Return 0 for invalid frequencies

        self.realms = {
            "quantum": RealmConfig(
                name="quantum",
                platonic_solid="icosahedron",
                main_crv=4.4439e+13, # UPDATED from 4.2e12 (Highest NRCI peak: 4.4439e+13 Hz)
                wavelength=calculate_wavelength(4.4439e+13), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.9,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # New sub_crvs: 0.25x, 0.5x, 1x, 2x, 4x of new main_crv
                sub_crvs=[1.1110e+13, 2.2219e+13, 4.4439e+13, 8.8878e+13, 1.7776e+14],
                frequency_range=(1e12, 1e15)
            ),
            "electromagnetic": RealmConfig(
                name="electromagnetic",
                platonic_solid="octahedron",
                main_crv=1.4042e+09, # UPDATED from 2.45e9 (Highest NRCI peak: 1.4042e+09 Hz)
                wavelength=calculate_wavelength(1.4042e+09), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.85,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # New sub_crvs: 0.25x, 0.5x, 1x, 2x, 4x of new main_crv
                sub_crvs=[3.5105e+08, 7.0210e+08, 1.4042e+09, 2.8084e+09, 5.6168e+09],
                frequency_range=(1e9, 1e11)
            ),
            "gravitational": RealmConfig(
                name="gravitational",
                platonic_solid="dodecahedron",
                main_crv=1.6019e+02, # UPDATED based on materials_research and default range.
                wavelength=calculate_wavelength(1.6019e+02), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.7,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # UPDATED sub_crvs for consistency
                sub_crvs=[4.0048e+01, 8.0095e+01, 1.6019e+02, 3.2038e+02, 6.4076e+02], # Derived from new main_crv
                frequency_range=(1e-2, 1e3) # UPDATED from 1e-18, 1e-15 (Based on user's prior research)
            ),
            "plasma": RealmConfig(
                name="plasma",
                platonic_solid="tetrahedron",
                main_crv=1.7560e+06, # UPDATED from 1.0e6 (Highest NRCI peak: 1.7560e+06 Hz)
                wavelength=calculate_wavelength(1.7560e+06), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.75,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # New sub_crvs: 0.25x, 0.5x, 1x, 2x, 4x of new main_crv
                sub_crvs=[4.3900e+05, 8.7800e+05, 1.7560e+06, 3.5120e+06, 7.0240e+06],
                frequency_range=(1e5, 1e8)
            ),
            "nuclear": RealmConfig(
                name="nuclear",
                platonic_solid="star_tetrahedron",
                main_crv=5.6569e+20, # UPDATED from 1.0e20 (Highest NRCI peak: 5.6569e+20 Hz)
                wavelength=calculate_wavelength(5.6569e+20), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.95,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # New sub_crvs: 0.25x, 0.5x, 1x, 2x, 4x of new main_crv
                sub_crvs=[1.4142e+20, 2.8284e+20, 5.6569e+20, 1.1314e+21, 2.2628e+21],
                frequency_range=(1e19, 1e21)
            ),
            "optical": RealmConfig(
                name="optical",
                platonic_solid="cuboctahedron",
                main_crv=6.7056e+14, # UPDATED from 5.0e14 (Highest NRCI peak: 6.7056e+14 Hz)
                wavelength=calculate_wavelength(6.7056e+14), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.9,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # UPDATED sub_crvs for consistency
                sub_crvs=[1.6764e+14, 3.3528e+14, 6.7056e+14, 1.3411e+15, 2.6822e+15],
                frequency_range=(4e14, 8e14)
            ),
            "biologic": RealmConfig(
                name="biologic",
                platonic_solid="rhombic_dodecahedron",
                main_crv=1.0000e+02, # UPDATED from 7.8300e+00 (Highest NRCI peak: 1.0000e+02 Hz)
                wavelength=calculate_wavelength(1.0000e+02), # Derived from new main_crv
                coordination_number=12,
                spatial_coherence=0.99,
                temporal_coherence=0.99,
                nrci_baseline=0.65,
                lattice_type="Resonant manifold",
                optimization_factor=1.0,
                # New sub_crvs: 0.25x, 0.5x, 1x, 2x, 4x of new main_crv
                sub_crvs=[2.5000e+01, 5.0000e+01, 1.0000e+02, 2.0000e+02, 4.0000e+02],
                frequency_range=(1e0, 1e3)
            )
        }

    def _initialize_default_molecules(self):
        """Initializes predefined molecular configurations for HTR."""
        self.molecules = {
            'propane': MoleculeConfig('propane', 10, 0.154e-9, 4.8, 'alkane', 'CCC'),
            'benzene': MoleculeConfig('benzene', 6, 0.14e-9, 5.0, 'aromatic', 'c1ccccc1'),
            'methane': MoleculeConfig('methane', 5, 0.109e-9, 4.5, 'tetrahedral', 'C'),
            'butane': MoleculeConfig('butane', 13, 0.154e-9, 4.8, 'alkane', 'CCCC')
        }


    def apply_environment_settings(self):
        """
        Applies environment-specific adjustments to configuration.
        If environment is "auto", it delegates to HardwareProfileManager.
        """
        if self.environment == "auto":
            print("UBPConfig: Auto-detecting hardware profile...")
            hw_manager = HardwareProfileManager()
            detected_profile_name = hw_manager.auto_detect_profile()
            detected_profile = hw_manager.get_profile(detected_profile_name)
            self._apply_hardware_profile_settings(detected_profile)
            print(f"UBPConfig: Applied AUTO-DETECTED profile '{detected_profile_name}' settings.")
        elif self.environment == "development":
            self.performance.TARGET_NRCI = 0.99
            self.performance.COHERENCE_THRESHOLD = 0.90
            self.BITFIELD_DIMENSIONS = self.bitfield.size_local
            print(f"UBPConfig: Applied DEVELOPMENT environment settings.")
        elif self.environment == "production":
            self.performance.TARGET_NRCI = 0.999999
            self.performance.COHERENCE_THRESHOLD = 0.95
            self.BITFIELD_DIMENSIONS = self.bitfield.size_production
            print(f"UBPConfig: Applied PRODUCTION environment settings.")
        elif self.environment == "testing":
            self.performance.TARGET_NRCI = 0.9
            self.performance.COHERENCE_THRESHOLD = 0.8
            self.BITFIELD_DIMENSIONS = (1, 1, 1, 1, 1, 1)
            print(f"UBPConfig: Applied TESTING environment settings.")
        else:
            print(f"UBPConfig: Unknown environment '{self.environment}'. Using default settings.")

    def _apply_hardware_profile_settings(self, profile: HardwareProfile):
        """Applies settings from a detected HardwareProfile to the UBPConfig."""
        self.performance.TARGET_NRCI = RawUBPConstants.NRCI_TARGET_STANDARD if profile.error_correction_level == "basic" else RawUBPConstants.NRCI_TARGET_HIGH_COHERENCE
        
        # Coherence threshold from profile, default to a sensible value if not directly available
        self.performance.COHERENCE_THRESHOLD = getattr(profile, 'coherence_threshold', RawUBPConstants.COHERENCE_THRESHOLD)
        
        self.BITFIELD_DIMENSIONS = profile.bitfield_dimensions
        self.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD = RawUBPConstants.COHERENCE_SYNCHRONIZATION_CYCLE_SECONDS
        self.observer.DEFAULT_INTENT_LEVEL = 1.0
        
        # Ensure 'enable_error_correction' is consistently handled as a boolean
        if isinstance(profile.enable_error_correction, bool):
            self.error_correction.nrci_base_score = 0.9 if profile.enable_error_correction else 0.7
        else:
            # Fallback if the attribute is not a boolean (e.g., might be a string)
            self.error_correction.nrci_base_score = 0.8 # Neutral value if type is unexpected

    def get_bitfield_dimensions(self) -> Tuple[int, ...]:
        """Returns the configured Bitfield dimensions."""
        return self.BITFIELD_DIMENSIONS

    def get_realm_config(self, realm_name: str) -> Optional[RealmConfig]:
        """Returns the configuration for a specific realm."""
        return self.realms.get(realm_name.lower())
    
    def get_molecule_config(self, molecule_name: str) -> Optional[MoleculeConfig]:
        """Returns the configuration for a specific molecule."""
        return self.molecules.get(molecule_name.lower())
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current configuration."""
        return {
            "environment": self.environment,
            "bitfield_dimensions": self.BITFIELD_DIMENSIONS,
            "target_nrci": self.performance.TARGET_NRCI,
            "num_realms_configured": len(self.realms),
            "num_molecules_configured": len(self.molecules),
            "example_quantum_crv": self.realms.get("quantum").main_crv if "quantum" in self.realms else None,
            "epsilon_ubp": self.constants.EPSILON_UBP,
        }

# --- Singleton Instance Management ---
_ubp_config_instance: Optional[UBPConfig] = None

def get_config(environment: Optional[str] = None) -> UBPConfig:
    """
    Returns the singleton UBPConfig instance.
    Initializes it if it doesn't exist. Can set environment on first call.
    """
    global _ubp_config_instance
    if _ubp_config_instance is None:
        if environment:
            _ubp_config_instance = UBPConfig(environment=environment)
        else:
            _ubp_config_instance = UBPConfig()
    elif environment and _ubp_config_instance.environment != environment:
        print(f"⚠️ Warning: UBPConfig already initialized with environment '{_ubp_config_instance.environment}'. "
              f"Ignoring request to set environment to '{environment}'.")
    return _ubp_config_instance

def reset_config():
    """
    Resets the singleton UBPConfig instance, allowing for re-initialization
    with different parameters or environments. Useful for testing.
    """
    global _ubp_config_instance
    _ubp_config_instance = None
    print("UBPConfig: Global configuration instance reset.")

# --- Example Usage (for testing/demonstration) ---
if __name__ == "__main__":
    print("--- Testing ubp_config.py ---")

    # Test default initialization
    config_default = get_config()
    print(f"\nDefault Config Environment: {config_default.environment}")
    print(f"Bitfield Dimensions: {config_default.get_bitfield_dimensions()}")
    print(f"Pi: {config_default.constants.PI}")
    print(f"E: {config_default.constants.E}")
    print(f"Golden Ratio: {config_default.constants.PHI}")
    print(f"Euler-Mascheroni: {config_default.constants.EULER_MASCHERONI}")
    print(f"Target NRCI: {config_default.performance.TARGET_NRCI}")
    print(f"CSC Period: {config_default.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD} seconds")
    print(f"Default Observer Intent: {config_default.observer.DEFAULT_INTENT_LEVEL}")
    print(f"UBP Zitterbewegung Freq: {config_default.constants.UBP_ZITTERBEWEGUNG_FREQ} Hz")
    print(f"Max Prime Default: {config_default.constants.MAX_PRIME_DEFAULT}")
    print(f"CRV Resonance Threshold Default: {config_default.crv.resonance_threshold_default}")
    print(f"UBP Frequency Weights (sample): {list(config_default.constants.UBP_FREQUENCY_WEIGHTS.items())[0]}")
    print(f"UBP Toggle Probabilities (quantum): {config_default.constants.UBP_TOGGLE_PROBABILITIES['quantum']}")
    print(f"UBP Realm Frequencies (electromagnetic): {config_default.constants.UBP_REALM_FREQUENCIES['electromagnetic']}")

    # Test new energy constants
    print(f"Default R_0 for Energy: {config_default.energy.R_0_DEFAULT}")
    print(f"Default H_T for Energy: {config_default.energy.H_T_DEFAULT}")
    print(f"Default S_OPT for Energy: {config_default.energy.S_OPT_DEFAULT}")


    # Test getting a specific realm
    em_realm = config_default.get_realm_config("electromagnetic")
    if em_realm:
        print(f"\nElectromagnetic Realm:")
        print(f"  Platonic Solid: {em_realm.platonic_solid}")
        print(f"  Wavelength: {em_realm.wavelength} m")
        print(f"  NRCI Baseline: {em_realm.nrci_baseline}")
        print(f"  Sub CRVs: {em_realm.sub_crvs}")
        print(f"  Frequency Range: {em_realm.frequency_range}")
        print(f"  Dummy Geometry (for retro-compatibility): {em_realm.geometry}") # Test dummy attribute
    else:
        print("Electromagnetic realm not found.")

    # Test getting a specific molecule
    propane_mol = config_default.get_molecule_config("propane")
    if propane_mol:
        print(f"\nPropane Molecule:")
        print(f"  Nodes: {propane_mol.nodes}")
        print(f"  Bond Length: {propane_mol.bond_length} m")
    else:
        print("Propane molecule not found.")

    # Test setting a different environment (should work only on first call for global instance)
    print("\nAttempting to re-initialize with 'testing' environment...")
    config_test = get_config(environment="testing") # Should print a warning
    print(f"Config after setting to 'testing': {config_test.environment}")
    print(f"Bitfield Dimensions in 'testing': {config_test.get_bitfield_dimensions()}")

    # To truly test different environments, you'd need to reset the global _ubp_config_instance
    # For demonstration, let's simulate by manually setting it to None and re-initializing
    print("\n--- Simulating fresh start for 'production' environment using reset_config() ---")
    reset_config() # Use the new function
    config_prod = get_config(environment="production")
    print(f"Config Environment: {config_prod.environment}")
    print(f"Bitfield Dimensions: {config_prod.get_bitfield_dimensions()}")
    print(f"Target NRCI: {config_prod.performance.TARGET_NRCI}")

    print("\n--- Simulating fresh start for 'auto' environment using reset_config() ---")
    reset_config() # Use the new function
    config_auto = get_config(environment="auto")
    print(f"Config Environment: {config_auto.environment}")
    print(f"Bitfield Dimensions: {config_auto.get_bitfield_dimensions()}")
    print(f"Target NRCI: {config_auto.performance.TARGET_NRCI}")
    
    print("\n✅ ubp_config.py test completed successfully!")
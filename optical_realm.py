"""
Universal Binary Principle (UBP) Framework v3.2+ - Enhanced Optical Realm Module
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

This module implements the complete Enhanced Optical Realm with photonic lattice
structures, WGE charge quantization, advanced photonics calculations, and 
comprehensive optical validation capabilities.

The Optical realm operates at 5×10^14 Hz (600 nm), targeting NRCI > 0.999999
through precise photonic modeling and Weyl Geometric Electromagnetism integration.

"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
from scipy.special import factorial, spherical_jn, spherical_yn
from scipy.optimize import minimize_scalar
from scipy.constants import c, h, hbar, e, epsilon_0, mu_0
import json

# Corrected imports:
from system_constants import UBPConstants # UBPConstants is in system_constants.py
from state import MutableBitfield, OffBit # Bitfield is now MutableBitfield and is in state.py


@dataclass
class OpticalRealmMetrics:
    """Comprehensive metrics for Optical Realm operations."""
    photonic_frequency: float
    wavelength: float
    refractive_index: float
    group_velocity: float
    phase_velocity: float
    dispersion_coefficient: float
    nonlinear_coefficient: float
    photonic_bandgap: float
    mode_confinement: float
    coupling_efficiency: float
    transmission_loss: float
    wge_charge_quantization: float


@dataclass
class PhotonicLatticeStructure:
    """Photonic lattice structure for optical realm operations."""
    lattice_type: str  # 'hexagonal', 'square', 'triangular', 'photonic_crystal'
    lattice_constant: float  # meters
    refractive_index_core: float
    refractive_index_cladding: float
    fill_factor: float
    bandgap_center: float  # Hz
    bandgap_width: float   # Hz
    mode_structure: np.ndarray
    dispersion_relation: np.ndarray
    field_distribution: np.ndarray


@dataclass
class WGEParameters:
    """Weyl Geometric Electromagnetism parameters for optical realm."""
    weyl_gauge_field: np.ndarray
    metric_tensor: np.ndarray
    charge_quantization_factor: float = 0.0072973525893  # Fine structure constant
    electromagnetic_coupling: float = 1.0
    geometric_phase: float = 0.0
    berry_curvature: np.ndarray = None
    topological_charge: int = 0


@dataclass
class PhotonicModeProfile:
    """Profile of a photonic mode in the optical realm."""
    mode_index: int
    effective_index: float
    group_index: float
    mode_area: float  # m²
    confinement_factor: float
    propagation_constant: complex
    field_profile: np.ndarray
    power_fraction: float


class OpticalRealm:
    """
    Enhanced Optical Realm implementation for the UBP Framework.
    
    This class provides comprehensive photonics modeling with photonic lattices,
    WGE charge quantization, advanced optical calculations, and validation.
    """
    
    def __init__(self, bitfield: Optional[MutableBitfield] = None): # Corrected Bitfield to MutableBitfield
        """
        Initialize the Enhanced Optical Realm.
        
        Args:
            bitfield: Optional MutableBitfield instance for optical operations
        """
        self.bitfield = bitfield
        
        # Optical realm parameters
        self.frequency = 5e14  # Hz (600 nm)
        self.wavelength = c / self.frequency  # meters
        self.angular_frequency = 2 * np.pi * self.frequency
        
        # Photonic constants
        self.photonic_constants = {
            'speed_of_light': c,
            'planck_constant': h,
            'reduced_planck': hbar,
            'elementary_charge': e,
            'vacuum_permittivity': epsilon_0,
            'vacuum_permeability': mu_0,
            'fine_structure': 0.0072973525893,
            'impedance_free_space': np.sqrt(mu_0 / epsilon_0)
        }
        
        # Initialize photonic lattice structure
        self.photonic_lattice = self._initialize_photonic_lattice()
        
        # Initialize WGE parameters
        self.wge_params = self._initialize_wge_parameters()
        
        # Initialize photonic modes
        self.photonic_modes = self._initialize_photonic_modes()
        
        # Performance metrics
        self.metrics = OpticalRealmMetrics(
            photonic_frequency=self.frequency,
            wavelength=self.wavelength,
            refractive_index=1.0,
            group_velocity=c,
            phase_velocity=c,
            dispersion_coefficient=0.0,
            nonlinear_coefficient=0.0,
            photonic_bandgap=0.0,
            mode_confinement=0.0,
            coupling_efficiency=0.0,
            transmission_loss=0.0,
            wge_charge_quantization=self.photonic_constants['fine_structure']
        )
        
        print(f"🔆 Enhanced Optical Realm Initialized")
        print(f"   Frequency: {self.frequency:.2e} Hz")
        print(f"   Wavelength: {self.wavelength*1e9:.1f} nm")
        print(f"   Lattice Type: {self.photonic_lattice.lattice_type}")
        print(f"   WGE Charge Quantization: {self.wge_params.charge_quantization_factor:.10f}")
    
    def _initialize_photonic_lattice(self) -> PhotonicLatticeStructure:
        """Initialize the photonic lattice structure."""
        
        # Hexagonal photonic crystal lattice (common for high-performance devices)
        lattice_constant = self.wavelength / 2  # Half-wavelength spacing
        
        # Refractive indices (typical for silicon photonics)
        n_core = 3.48    # Silicon
        n_cladding = 1.44  # Silicon dioxide
        
        # Calculate photonic bandgap
        fill_factor = 0.3  # 30% fill factor
        contrast = (n_core**2 - n_cladding**2) / (n_core**2 + n_cladding**2)
        
        # Bandgap center frequency (approximate)
        bandgap_center = c / (2 * lattice_constant * np.sqrt((n_core**2 + n_cladding**2) / 2))
        bandgap_width = bandgap_center * contrast * fill_factor
        
        # Mode structure (simplified - fundamental TE and TM modes)
        mode_structure = np.array([
            [1, 0, 0],  # TE₀₁ mode
            [0, 1, 0],  # TM₀₁ mode
            [1, 1, 0],  # TE₁₁ mode
            [0, 0, 1]   # TM₁₁ mode
        ])
        
        # Dispersion relation (ω vs k)
        k_values = np.linspace(0, 2*np.pi/lattice_constant, 100)
        omega_values = c * k_values / np.sqrt(n_core**2 + n_cladding**2)
        dispersion_relation = np.column_stack([k_values, omega_values])
        
        # Field distribution (Gaussian approximation)
        x = np.linspace(-2*lattice_constant, 2*lattice_constant, 50)
        y = np.linspace(-2*lattice_constant, 2*lattice_constant, 50)
        X, Y = np.meshgrid(x, y)
        field_distribution = np.exp(-(X**2 + Y**2) / (lattice_constant/2)**2)
        
        return PhotonicLatticeStructure(
            lattice_type="hexagonal_photonic_crystal",
            lattice_constant=lattice_constant,
            refractive_index_core=n_core,
            refractive_index_cladding=n_cladding,
            fill_factor=fill_factor,
            bandgap_center=bandgap_center,
            bandgap_width=bandgap_width,
            mode_structure=mode_structure,
            dispersion_relation=dispersion_relation,
            field_distribution=field_distribution
        )
    
    def _initialize_wge_parameters(self) -> WGEParameters:
        """Initialize Weyl Geometric Electromagnetism parameters."""
        
        # Weyl gauge field (4-vector potential)
        A_weyl = np.array([0.0, 0.0, 0.0, 1.0])  # Temporal component dominant
        
        # Metric tensor (Minkowski + Weyl correction)
        eta = np.diag([-1, 1, 1, 1])  # Minkowski metric
        A_outer = np.outer(A_weyl, A_weyl)
        g_weyl = eta + self.photonic_constants['fine_structure'] * A_outer
        
        # Berry curvature for topological photonics
        berry_curvature = np.array([0.0, 0.0, self.photonic_constants['fine_structure']])
        
        return WGEParameters(
            weyl_gauge_field=A_weyl,
            metric_tensor=g_weyl,
            charge_quantization_factor=self.photonic_constants['fine_structure'],
            electromagnetic_coupling=1.0,
            geometric_phase=0.0,
            berry_curvature=berry_curvature,
            topological_charge=1
        )
    
    def _initialize_photonic_modes(self) -> List[PhotonicModeProfile]:
        """Initialize photonic mode profiles."""
        
        modes = []
        lattice = self.photonic_lattice
        
        # Fundamental TE mode
        te_mode = PhotonicModeProfile(
            mode_index=0,
            effective_index=2.4,  # Typical for silicon waveguide
            group_index=4.2,
            mode_area=0.25e-12,  # 0.25 μm²
            confinement_factor=0.8,
            propagation_constant=2*np.pi*2.4/self.wavelength + 0j,
            field_profile=lattice.field_distribution,
            power_fraction=0.85
        )
        modes.append(te_mode)
        
        # Fundamental TM mode
        tm_mode = PhotonicModeProfile(
            mode_index=1,
            effective_index=1.8,
            group_index=3.8,
            mode_area=0.35e-12,  # 0.35 μm²
            confinement_factor=0.7,
            propagation_constant=2*np.pi*1.8/self.wavelength + 0.01j,  # Small loss
            field_profile=lattice.field_distribution * 0.8,
            power_fraction=0.75
        )
        modes.append(tm_mode)
        
        return modes
    
    def calculate_photonic_bandgap(self, k_vector: np.ndarray) -> Dict[str, Any]:
        """
        Calculate photonic bandgap structure.
        
        Args:
            k_vector: Wave vector array
            
        Returns:
            Dictionary containing bandgap information
        """
        lattice = self.photonic_lattice
        
        # Plane wave expansion method (simplified)
        n_core = lattice.refractive_index_core
        n_clad = lattice.refractive_index_cladding
        a = lattice.lattice_constant
        
        # Calculate band structure
        bands = []
        for k in k_vector:
            # First band (fundamental)
            omega1 = c * k / n_clad
            
            # Second band (with bandgap)
            if k < np.pi / a:
                omega2 = c * np.sqrt(k**2 + (np.pi/a)**2) / np.sqrt(n_core**2 + n_clad**2)
            else:
                omega2 = c * k / n_core
            
            bands.append([omega1, omega2])
        
        bands = np.array(bands)
        
        # Find bandgap
        gap_start = np.max(bands[:, 0])
        gap_end = np.min(bands[:, 1])
        gap_width = gap_end - gap_start if gap_end > gap_start else 0
        
        return {
            'k_vector': k_vector,
            'band_structure': bands,
            'bandgap_start': gap_start,
            'bandgap_end': gap_end,
            'bandgap_width': gap_width,
            'bandgap_center': (gap_start + gap_end) / 2,
            'relative_gap': gap_width / ((gap_start + gap_end) / 2) if gap_width > 0 else 0
        }
    
    def calculate_wge_charge_quantization(self, field_strength: float) -> Dict[str, float]:
        """
        Calculate WGE charge quantization effects.
        
        Args:
            field_strength: Electromagnetic field strength
            
        Returns:
            Dictionary containing quantization results
        """
        wge = self.wge_params
        alpha = wge.charge_quantization_factor  # Fine structure constant
        
        # Orbital flux quantization: φ_orb = n * h / e
        flux_quantum = h / e  # Weber
        orbital_flux = field_strength * alpha
        quantization_number = orbital_flux / flux_quantum
        
        # Geometric phase calculation
        berry_phase = np.dot(wge.berry_curvature, [field_strength, 0, 0])
        geometric_phase = berry_phase * alpha
        
        # Topological charge contribution
        topological_contribution = wge.topological_charge * alpha * field_strength
        
        # Total quantized charge
        quantized_charge = e * (quantization_number + geometric_phase / (2*np.pi))
        
        return {
            'flux_quantum': flux_quantum,
            'orbital_flux': orbital_flux,
            'quantization_number': quantization_number,
            'geometric_phase': geometric_phase,
            'berry_phase': berry_phase,
            'topological_contribution': topological_contribution,
            'quantized_charge': quantized_charge,
            'fine_structure_constant': alpha
        }
    
    def calculate_nonlinear_optics(self, input_power: float, length: float) -> Dict[str, Any]:
        """
        Calculate nonlinear optical effects.
        
        Args:
            input_power: Input optical power (Watts)
            length: Propagation length (meters)
            
        Returns:
            Dictionary containing nonlinear optical results
        """
        # Nonlinear refractive index (typical for silicon)
        n2 = 4.5e-18  # m²/W
        
        # Effective mode area
        A_eff = self.photonic_modes[0].mode_area if self.photonic_modes else 1e-12
        
        # Nonlinear parameter
        gamma = 2 * np.pi * n2 / (self.wavelength * A_eff)
        
        # Nonlinear phase shift
        phi_nl = gamma * input_power * length
        
        # Self-phase modulation
        spm_phase = phi_nl
        
        # Kerr effect
        kerr_coefficient = n2 * input_power / A_eff
        
        # Four-wave mixing efficiency (simplified)
        fwm_efficiency = (gamma * input_power * length)**2 if phi_nl < np.pi else 0.1
        
        # Stimulated Brillouin scattering threshold
        sbs_threshold = 21 * A_eff / (gamma * length) if length > 0 else np.inf
        
        return {
            'nonlinear_parameter': gamma,
            'nonlinear_phase': phi_nl,
            'spm_phase': spm_phase,
            'kerr_coefficient': kerr_coefficient,
            'fwm_efficiency': fwm_efficiency,
            'sbs_threshold': sbs_threshold,
            'effective_area': A_eff,
            'propagation_length': length
        }
    
    def calculate_dispersion_effects(self, wavelength_range: np.ndarray) -> Dict[str, Any]:
        """
        Calculate chromatic dispersion effects.
        
        Args:
            wavelength_range: Array of wavelengths (meters)
            
        Returns:
            Dictionary containing dispersion results
        """
        # Material dispersion (Sellmeier equation for silicon)
        def sellmeier_silicon(lam):
            # Wavelength in micrometers
            lam_um = lam * 1e6
            n_sq = 1 + (10.6684293 * lam_um**2) / (lam_um**2 - 0.301516485**2) + \
                   (0.0030434748 * lam_um**2) / (lam_um**2 - 1.13475115**2) + \
                   (1.54133408 * lam_um**2) / (lam_um**2 - 1104**2)
            return np.sqrt(n_sq)
        
        # Calculate refractive index for wavelength range
        n_values = np.array([sellmeier_silicon(lam) for lam in wavelength_range])
        
        # Group velocity dispersion (GVD)
        c_light = self.photonic_constants['speed_of_light']
        
        # Numerical derivatives for dispersion calculation
        if len(wavelength_range) > 2:
            dn_dlam = np.gradient(n_values, wavelength_range)
            d2n_dlam2 = np.gradient(dn_dlam, wavelength_range)
            
            # Group velocity
            v_g = c_light / (n_values - wavelength_range * dn_dlam)
            
            # GVD parameter
            D = -(wavelength_range / c_light) * d2n_dlam2  # s/m²
            
            # Dispersion length
            pulse_width = 1e-12  # 1 ps pulse
            L_D = pulse_width**2 / np.abs(D)
        else:
            v_g = np.array([c_light / n_values[0]])
            D = np.array([0.0])
            L_D = np.array([np.inf])
        
        return {
            'wavelength_range': wavelength_range,
            'refractive_index': n_values,
            'group_velocity': v_g,
            'dispersion_parameter': D,
            'dispersion_length': L_D,
            'dn_dlambda': dn_dlam if len(wavelength_range) > 2 else np.array([0.0]),
            'd2n_dlambda2': d2n_dlam2 if len(wavelength_range) > 2 else np.array([0.0])
        }
    
    def calculate_coupling_efficiency(self, mode1: PhotonicModeProfile, 
                                    mode2: PhotonicModeProfile) -> float:
        """
        Calculate coupling efficiency between two photonic modes.
        
        Args:
            mode1: First photonic mode
            mode2: Second photonic mode
            
        Returns:
            Coupling efficiency (0 to 1)
        """
        # Overlap integral calculation
        field1 = mode1.field_profile
        field2 = mode2.field_profile
        
        # Ensure same dimensions
        if field1.shape != field2.shape:
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(field1.shape, field2.shape))
            field1 = field1[:min_shape[0], :min_shape[1]]
            field2 = field2[:min_shape[0], :min_shape[1]]
        
        # Normalize fields
        field1_norm = field1 / np.sqrt(np.sum(np.abs(field1)**2))
        field2_norm = field2 / np.sqrt(np.sum(np.abs(field2)**2))
        
        # Overlap integral
        overlap = np.abs(np.sum(field1_norm * np.conj(field2_norm)))**2
        
        # Mode mismatch factor
        area_ratio = mode1.mode_area / mode2.mode_area
        mismatch_factor = 4 * area_ratio / (1 + area_ratio)**2
        
        # Total coupling efficiency
        coupling_efficiency = overlap * mismatch_factor
        
        return min(coupling_efficiency, 1.0)
    
    def run_optical_computation(self, input_data: np.ndarray, 
                               computation_type: str = 'full') -> Dict[str, Any]:
        """
        Run comprehensive optical realm computation.
        
        Args:
            input_data: Input data for optical computation
            computation_type: Type of computation ('bandgap', 'wge', 'nonlinear', 'dispersion', 'full')
            
        Returns:
            Dictionary containing computation results
        """
        results = {
            'computation_type': computation_type,
            'input_size': len(input_data),
            'optical_frequency': self.frequency,
            'wavelength': self.wavelength
        }
        
        if computation_type in ['bandgap', 'full']:
            # Photonic bandgap calculation
            k_max = 2 * np.pi / self.photonic_lattice.lattice_constant
            k_vector = np.linspace(0, k_max, len(input_data))
            bandgap_results = self.calculate_photonic_bandgap(k_vector)
            results['bandgap'] = bandgap_results
            
            # Update metrics
            self.metrics.photonic_bandgap = bandgap_results['bandgap_width']
        
        if computation_type in ['wge', 'full']:
            # WGE charge quantization
            field_strength = np.mean(np.abs(input_data))
            wge_results = self.calculate_wge_charge_quantization(field_strength)
            results['wge'] = wge_results
            
            # Update metrics
            self.metrics.wge_charge_quantization = wge_results['fine_structure_constant']
        
        if computation_type in ['nonlinear', 'full']:
            # Nonlinear optics
            input_power = np.mean(input_data**2) * 1e-3  # Convert to Watts
            length = 1e-3  # 1 mm propagation length
            nonlinear_results = self.calculate_nonlinear_optics(input_power, length)
            results['nonlinear'] = nonlinear_results
            
            # Update metrics
            self.metrics.nonlinear_coefficient = nonlinear_results['nonlinear_parameter']
        
        if computation_type in ['dispersion', 'full']:
            # Dispersion effects
            wavelength_center = self.wavelength
            wavelength_range = np.linspace(wavelength_center * 0.95, 
                                         wavelength_center * 1.05, 
                                         min(len(input_data), 50))
            dispersion_results = self.calculate_dispersion_effects(wavelength_range)
            results['dispersion'] = dispersion_results
            
            # Update metrics
            if len(dispersion_results['group_velocity']) > 0:
                self.metrics.group_velocity = np.mean(dispersion_results['group_velocity'])
                self.metrics.dispersion_coefficient = np.mean(np.abs(dispersion_results['dispersion_parameter']))
        
        if computation_type in ['coupling', 'full'] and len(self.photonic_modes) >= 2:
            # Mode coupling
            coupling_eff = self.calculate_coupling_efficiency(
                self.photonic_modes[0], self.photonic_modes[1]
            )
            results['coupling_efficiency'] = coupling_eff
            
            # Update metrics
            self.metrics.coupling_efficiency = coupling_eff
        
        # Calculate overall optical realm NRCI
        nrci_components = []
        
        if 'bandgap' in results:
            # Higher bandgap width indicates better photonic control
            bandgap_nrci = min(results['bandgap']['relative_gap'] * 10, 1.0)
            nrci_components.append(bandgap_nrci)
        
        if 'wge' in results:
            # WGE quantization precision
            quantization_precision = 1.0 - abs(results['wge']['quantization_number'] % 1 - 0.5) * 2
            nrci_components.append(quantization_precision)
        
        if 'nonlinear' in results:
            # Nonlinear efficiency (moderate nonlinearity is optimal)
            nl_phase = results['nonlinear']['nonlinear_phase']
            nl_nrci = np.exp(-abs(nl_phase - np.pi/2)**2)  # Optimal at π/2
            nrci_components.append(nl_nrci)
        
        if 'dispersion' in results and len(dispersion_results['group_velocity']) > 0:
            # Dispersion control (low dispersion is better for most applications)
            disp_control = np.exp(-np.mean(np.abs(dispersion_results['dispersion_parameter'])) * 1e12)
            nrci_components.append(disp_control)
        
        if 'coupling_efficiency' in results:
            nrci_components.append(results['coupling_efficiency'])
        
        # Overall optical NRCI
        optical_nrci = np.mean(nrci_components) if nrci_components else 0.5
        results['optical_nrci'] = optical_nrci
        
        return results
    
    def get_optical_metrics(self) -> OpticalRealmMetrics:
        """Get current optical realm metrics."""
        return self.metrics
    
    def validate_optical_realm(self) -> Dict[str, Any]:
        """
        Comprehensive validation of optical realm implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'realm_name': 'Optical',
            'frequency': self.frequency,
            'wavelength': self.wavelength,
            'lattice_type': self.photonic_lattice.lattice_type,
            'wge_enabled': True
        }
        
        # Test with synthetic optical data
        test_data = np.random.normal(0, 1, 100) + 1j * np.random.normal(0, 1, 100)
        test_data_real = np.real(test_data)
        
        # Run comprehensive computation
        computation_results = self.run_optical_computation(test_data_real, 'full')
        validation_results.update(computation_results)
        
        # Validation criteria
        validation_criteria = {
            'frequency_valid': 4e14 < self.frequency < 8e14,  # Visible/near-IR range
            'wavelength_valid': 400e-9 < self.wavelength < 800e-9,  # nm range
            'bandgap_exists': computation_results.get('bandgap', {}).get('bandgap_width', 0) > 0,
            'wge_quantization_valid': 0 < computation_results.get('wge', {}).get('fine_structure_constant', 0) < 0.01,
            'nonlinear_realistic': 0 < computation_results.get('nonlinear', {}).get('nonlinear_parameter', 0) < 1e3,
            'dispersion_calculated': len(computation_results.get('dispersion', {}).get('group_velocity', [])) > 0,
            'optical_nrci_high': computation_results.get('optical_nrci', 0) > 0.7
        }
        
        validation_results['validation_criteria'] = validation_criteria
        validation_results['overall_valid'] = all(validation_criteria.values())
        
        return validation_results


# Alias for compatibility
OpticalRealmFramework = OpticalRealm
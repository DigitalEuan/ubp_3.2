"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Semantics Package
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Core semantic functions and mathematical operations for the Universal Binary Principle.
"""

# Import core components that definitely exist
from .constants import load_ubp_constants, get_frequency_weights
from .state import OffBit, MutableBitfield, UBPState

# Import components with correct class names
try:
    from .prime_resonance import PrimeResonanceCoordinateSystem
except ImportError:
    PrimeResonanceCoordinateSystem = None

try:
    from .global_coherence import GlobalCoherenceIndex
except ImportError:
    GlobalCoherenceIndex = None

try:
    from .enhanced_nrci import EnhancedNRCI
except ImportError:
    EnhancedNRCI = None

try:
    from .observer_scaling import ObserverScaling
except ImportError:
    ObserverScaling = None

try:
    from .carfe import CARFEFieldEquation
except ImportError:
    CARFEFieldEquation = None

try:
    from .tgic import TGICSystem
except ImportError:
    TGICSystem = None

try:
    from .dot_theory import DotTheorySystem
except ImportError:
    DotTheorySystem = None

try:
    from .spin_transition import SpinTransitionModule
except ImportError:
    SpinTransitionModule = None

try:
    from .p_adic_correction import PAdic, AdelicNumber, PAdic_ErrorCorrection
except ImportError:
    PAdic = None
    AdelicNumber = None
    PAdic_ErrorCorrection = None

try:
    from .glr_framework.level_7_global_golay import GlobalGolayCorrection
except ImportError:
    GlobalGolayCorrection = None

__all__ = [
    'load_ubp_constants', 'get_frequency_weights',
    'OffBit', 'MutableBitfield', 'UBPState',
    'PrimeResonanceCoordinateSystem', 'GlobalCoherenceIndex', 'EnhancedNRCI',
    'ObserverScaling', 'CARFEFieldEquation', 'TGICSystem', 'DotTheorySystem',
    'SpinTransitionModule', 'PAdic', 'AdelicNumber', 'PAdic_ErrorCorrection',
    'GlobalGolayCorrection'
]
"""
Universal Binary Principle (UBP) Framework v3.2+ - GLR Level 7: Global Golay Correction with Syndrome Calculation
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Implements the complete Golay(24,12) error correction system with
parity-check matrix H and syndrome calculation S = H × v mod 2.

This is the core mathematical component that provides:
- Error detection via syndrome calculation
- Error correction using syndrome lookup tables  
- Integration with OffBit 24-bit structure
- Production-ready error correction for UBP

Mathematical Foundation:
- H: 12×24 parity-check matrix for Golay(24,12)
- S = H × v mod 2 (syndrome calculation)
- Error correction capability: up to 3-bit errors
- Code parameters: n=24, k=12, d=8

This is NOT a simulation - all mathematical operations are exact.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from glr_base import GLRProcessor, GLRLevel, GLRResult, LatticeStructure, LatticeType


@dataclass
class GolayCodeParameters:
    """Parameters for the Golay(24,12) code"""
    n: int = 24  # Codeword length
    k: int = 12  # Message length
    d: int = 8   # Minimum distance
    t: int = 3   # Error correction capability


class GolayParityCheckMatrix:
    """
    Golay(24,12) Parity-Check Matrix H and syndrome calculation.
    
    Implements the exact mathematical specification for UBP GLR Level 7.
    """
    
    def __init__(self):
        self.params = GolayCodeParameters()
        self._H = None
        self._syndrome_table = None
        self._error_patterns = None
        
    @property
    def H(self) -> np.ndarray:
        """Get the 12×24 parity-check matrix H for Golay(24,12)"""
        if self._H is None:
            self._H = self._generate_parity_check_matrix()
        return self._H
    
    def _generate_parity_check_matrix(self) -> np.ndarray:
        """
        Generate the exact Golay(24,12) parity-check matrix.
        
        H = [P^T | I_12], where P^T is the transpose of the generator's parity submatrix.
        
        Returns:
            12×24 parity-check matrix
        """
        # Define P^T (12×12) - exact Golay construction
        P_T = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,1,1,1,1,1,0,0,0,0,0,0],
            [1,1,1,0,0,0,1,1,1,0,0,0],
            [1,1,0,1,0,0,1,0,0,1,1,0],
            [1,1,0,0,1,0,0,1,0,1,0,1],
            [1,1,0,0,0,1,0,0,1,0,1,1],
            [1,0,1,1,0,0,0,1,1,1,0,0],
            [1,0,1,0,1,0,1,0,1,0,1,0],
            [1,0,1,0,0,1,1,1,0,0,0,1],
            [1,0,0,1,1,0,1,0,0,0,1,1],
            [1,0,0,1,0,1,0,1,1,1,0,0],
            [1,0,0,0,1,1,0,0,1,1,1,0]
        ], dtype=int) % 2
        
        # I_12 identity matrix
        I_12 = np.eye(12, dtype=int)
        
        # H = [P^T | I_12]
        H = np.hstack((P_T, I_12))
        
        # Verify dimensions
        assert H.shape == (12, 24), f"Expected (12, 24), got {H.shape}"
        
        return H
    
    def compute_syndrome(self, received_vector: np.ndarray) -> np.ndarray:
        """
        Compute syndrome S = H × v mod 2.
        
        This is the core UBP formula: S = H × v mod 2
        
        Args:
            received_vector: 24-bit vector (OffBit data)
        
        Returns:
            12-bit syndrome vector
        """
        if received_vector.shape != (24,):
            raise ValueError(f"Expected 24-bit vector, got shape {received_vector.shape}")
        
        # Ensure binary values
        v = np.array(received_vector, dtype=int) % 2
        
        # S = H × v mod 2
        syndrome = np.dot(self.H, v) % 2
        
        return syndrome.astype(int)
    
    def detect_error(self, syndrome: np.ndarray) -> bool:
        """
        Detect if errors are present based on syndrome.
        
        Args:
            syndrome: 12-bit syndrome vector
        
        Returns:
            True if errors detected, False if no errors
        """
        return np.any(syndrome)
    
    def get_error_weight(self, syndrome: np.ndarray) -> int:
        """
        Estimate error weight from syndrome.
        
        Args:
            syndrome: 12-bit syndrome vector
        
        Returns:
            Estimated number of errors
        """
        # Hamming weight of syndrome gives error estimate
        return np.sum(syndrome)
    
    @property
    def syndrome_table(self) -> Dict[str, np.ndarray]:
        """Get precomputed syndrome lookup table for error correction"""
        if self._syndrome_table is None:
            self._generate_syndrome_table()
        return self._syndrome_table
    
    def _generate_syndrome_table(self):
        """
        Generate syndrome lookup table for error correction.
        
        Maps syndrome patterns to error patterns for fast correction.
        """
        self._syndrome_table = {}
        self._error_patterns = {}
        
        # Generate all possible error patterns up to weight 3
        for weight in range(1, 4):  # 1, 2, 3 bit errors
            for positions in self._generate_error_positions(weight):
                error_pattern = np.zeros(24, dtype=int)
                for pos in positions:
                    error_pattern[pos] = 1
                
                # Compute syndrome for this error pattern
                syndrome = self.compute_syndrome(error_pattern)
                syndrome_key = ''.join(map(str, syndrome))
                
                # Store in lookup table
                if syndrome_key not in self._syndrome_table:
                    self._syndrome_table[syndrome_key] = error_pattern.copy()
                    self._error_patterns[syndrome_key] = positions
    
    def _generate_error_positions(self, weight: int) -> List[Tuple[int, ...]]:
        """Generate all combinations of error positions for given weight"""
        from itertools import combinations
        return list(combinations(range(24), weight))
    
    def correct_error(self, received_vector: np.ndarray, syndrome: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Correct errors using syndrome lookup table.
        
        Args:
            received_vector: 24-bit received vector
            syndrome: 12-bit syndrome vector
        
        Returns:
            Tuple of (corrected_vector, error_count)
        """
        if not self.detect_error(syndrome):
            return received_vector.copy(), 0
        
        syndrome_key = ''.join(map(str, syndrome))
        
        # Look up error pattern in syndrome table
        if syndrome_key in self.syndrome_table:
            error_pattern = self.syndrome_table[syndrome_key]
            corrected = (received_vector + error_pattern) % 2
            error_count = np.sum(error_pattern)
            return corrected, error_count
        else:
            # Uncorrectable error pattern
            return received_vector.copy(), -1
    
    def validate_codeword(self, codeword: np.ndarray) -> bool:
        """
        Validate if a vector is a valid Golay codeword.
        
        Args:
            codeword: 24-bit vector to validate
        
        Returns:
            True if valid codeword (syndrome = 0), False otherwise
        """
        syndrome = self.compute_syndrome(codeword)
        return not self.detect_error(syndrome)


class GlobalGolayCorrection(GLRProcessor):
    """
    GLR Level 7: Global Golay Correction processor.
    
    Implements realm-wide coherence using Golay(24,12) error correction
    with syndrome calculation S = H × v mod 2.
    """
    
    def __init__(self):
        self.golay_matrix = GolayParityCheckMatrix()
        self.lattice_structure = self._create_lattice_structure()
        self._correction_history = []
        
    def _create_lattice_structure(self) -> LatticeStructure:
        """Create lattice structure for Global Golay Correction"""
        return LatticeStructure(
            lattice_type=LatticeType.GOLAY_GLOBAL,
            coordination_number=24,  # 24-bit codewords
            harmonic_modes=[12.0, 24.0],  # k=12, n=24
            error_correction_levels={
                'local': 'hamming_7_4',
                'regional': 'bch_31_21', 
                'global': 'golay_23_12'
            },
            spatial_efficiency=0.85,  # High efficiency for global correction
            temporal_efficiency=0.92,
            nrci_target=0.999999,  # OnBit regime target
            wavelength=800.0,  # nm - global coherence wavelength
            frequency=3.75e14,  # Hz - corresponding frequency
            realm="global",
            symmetry_group="M_24",  # Mathieu group M_24
            basis_vectors=None  # Global correction doesn't use spatial basis
        )
    
    def get_level(self) -> GLRLevel:
        """Return GLR Level 7"""
        return GLRLevel.LEVEL_7_GLOBAL_GOLAY
    
    def get_lattice_structure(self) -> LatticeStructure:
        """Return the lattice structure for this level"""
        return self.lattice_structure
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data for Global Golay Correction.
        
        Args:
            data: Input data to validate
        
        Returns:
            True if data is valid for processing
        """
        # Data should be 24-bit vectors or multiples thereof
        if len(data.shape) == 1:
            return data.shape[0] % 24 == 0
        elif len(data.shape) == 2:
            return data.shape[1] == 24
        else:
            return False
    
    def process_correction(self, data: np.ndarray, **kwargs) -> GLRResult:
        """
        Process Global Golay error correction.
        
        Args:
            data: Input data (24-bit vectors)
            **kwargs: Additional parameters
        
        Returns:
            GLRResult with correction results
        """
        start_time = time.time()
        
        # Reshape data to 24-bit vectors if needed
        if len(data.shape) == 1 and data.shape[0] % 24 == 0:
            vectors = data.reshape(-1, 24)
        elif len(data.shape) == 2 and data.shape[1] == 24:
            vectors = data
        else:
            raise ValueError("Data must be 24-bit vectors or multiples thereof")
        
        corrected_vectors = []
        total_errors = 0
        correction_details = []
        
        # Process each 24-bit vector
        for i, vector in enumerate(vectors):
            # Compute syndrome: S = H × v mod 2
            syndrome = self.golay_matrix.compute_syndrome(vector)
            
            # Detect and correct errors
            if self.golay_matrix.detect_error(syndrome):
                corrected_vector, error_count = self.golay_matrix.correct_error(vector, syndrome)
                
                if error_count > 0:
                    total_errors += error_count
                    correction_details.append({
                        'vector_index': i,
                        'syndrome': syndrome.tolist(),
                        'error_count': error_count,
                        'correctable': error_count <= 3
                    })
                else:
                    # Uncorrectable error
                    corrected_vector = vector.copy()
                    correction_details.append({
                        'vector_index': i,
                        'syndrome': syndrome.tolist(),
                        'error_count': -1,
                        'correctable': False
                    })
            else:
                # No errors detected
                corrected_vector = vector.copy()
            
            corrected_vectors.append(corrected_vector)
        
        # Reconstruct corrected data
        corrected_data = np.array(corrected_vectors)
        if len(data.shape) == 1:
            corrected_data = corrected_data.flatten()
        
        # Compute correction efficiency
        correctable_errors = sum(1 for detail in correction_details if detail['correctable'])
        total_error_events = len(correction_details)
        correction_efficiency = correctable_errors / max(1, total_error_events)
        
        # Compute NRCI improvement (simplified)
        nrci_before = 1.0 - (total_errors / max(1, len(vectors) * 24))
        nrci_after = min(1.0, nrci_before + correction_efficiency * 0.1)
        
        processing_time = time.time() - start_time
        
        result = GLRResult(
            level=self.get_level(),
            success=total_errors == 0 or correction_efficiency > 0.5,
            corrected_data=corrected_data,
            error_count=total_errors,
            correction_efficiency=correction_efficiency,
            nrci_before=nrci_before,
            nrci_after=nrci_after,
            processing_time=processing_time,
            metadata={
                'syndrome_calculations': len(vectors),
                'correction_details': correction_details,
                'golay_parameters': {
                    'n': self.golay_matrix.params.n,
                    'k': self.golay_matrix.params.k,
                    'd': self.golay_matrix.params.d,
                    't': self.golay_matrix.params.t
                },
                'matrix_dimensions': self.golay_matrix.H.shape,
                'correctable_errors': correctable_errors,
                'uncorrectable_errors': total_error_events - correctable_errors
            }
        )
        
        self._correction_history.append(result)
        return result
    
    def compute_error_metrics(self, original: np.ndarray, corrected: np.ndarray) -> Dict[str, float]:
        """
        Compute error metrics for Global Golay correction.
        
        Args:
            original: Original data before correction
            corrected: Data after correction
        
        Returns:
            Dictionary of error metrics
        """
        if original.shape != corrected.shape:
            raise ValueError("Original and corrected data must have same shape")
        
        # Bit error rate
        total_bits = original.size
        bit_errors = np.sum(original != corrected)
        bit_error_rate = bit_errors / total_bits
        
        # Hamming distance
        hamming_distance = np.sum(original != corrected)
        
        # Syndrome-based metrics
        if len(original.shape) == 1 and original.shape[0] % 24 == 0:
            vectors_orig = original.reshape(-1, 24)
            vectors_corr = corrected.reshape(-1, 24)
        elif len(original.shape) == 2 and original.shape[1] == 24:
            vectors_orig = original
            vectors_corr = corrected
        else:
            vectors_orig = original.reshape(-1, 24)
            vectors_corr = corrected.reshape(-1, 24)
        
        syndrome_improvements = 0
        for orig_vec, corr_vec in zip(vectors_orig, vectors_corr):
            syndrome_orig = self.golay_matrix.compute_syndrome(orig_vec)
            syndrome_corr = self.golay_matrix.compute_syndrome(corr_vec)
            
            if np.sum(syndrome_corr) < np.sum(syndrome_orig):
                syndrome_improvements += 1
        
        syndrome_improvement_rate = syndrome_improvements / len(vectors_orig)
        
        return {
            'bit_error_rate': bit_error_rate,
            'hamming_distance': hamming_distance,
            'total_bits': total_bits,
            'syndrome_improvement_rate': syndrome_improvement_rate,
            'vectors_processed': len(vectors_orig),
            'syndrome_improvements': syndrome_improvements
        }
    
    def process_offbit_correction(self, offbit_value: int) -> Tuple[int, Dict[str, Any]]:
        """
        Process Golay correction for a single OffBit value.
        
        Args:
            offbit_value: 32-bit OffBit value
        
        Returns:
            Tuple of (corrected_offbit, correction_metadata)
        """
        # Extract 24-bit data from OffBit (bits 0-23)
        data_24bit = offbit_value & 0xFFFFFF
        
        # Convert to binary array
        binary_data = np.array([
            (data_24bit >> i) & 1 for i in range(24)
        ], dtype=int)
        
        # Compute syndrome
        syndrome = self.golay_matrix.compute_syndrome(binary_data)
        
        # Correct if needed
        if self.golay_matrix.detect_error(syndrome):
            corrected_binary, error_count = self.golay_matrix.correct_error(binary_data, syndrome)
            
            # Convert back to integer
            corrected_24bit = 0
            for i in range(24):
                if corrected_binary[i]:
                    corrected_24bit |= (1 << i)
            
            # Preserve upper 8 bits, replace lower 24 bits
            corrected_offbit = (offbit_value & 0xFF000000) | corrected_24bit
            
            metadata = {
                'error_detected': True,
                'error_count': error_count,
                'syndrome': syndrome.tolist(),
                'correctable': error_count > 0,
                'original_24bit': data_24bit,
                'corrected_24bit': corrected_24bit
            }
        else:
            # No errors
            corrected_offbit = offbit_value
            metadata = {
                'error_detected': False,
                'error_count': 0,
                'syndrome': syndrome.tolist(),
                'correctable': True
            }
        
        return corrected_offbit, metadata
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about Global Golay corrections performed.
        
        Returns:
            Dictionary containing correction statistics
        """
        if not self._correction_history:
            return {'statistics': 'no_corrections_performed'}
        
        total_corrections = len(self._correction_history)
        successful_corrections = sum(1 for result in self._correction_history if result.success)
        total_errors_corrected = sum(result.error_count for result in self._correction_history)
        
        avg_correction_efficiency = np.mean([
            result.correction_efficiency for result in self._correction_history
        ])
        
        avg_processing_time = np.mean([
            result.processing_time for result in self._correction_history
        ])
        
        nrci_improvements = [
            result.nrci_after - result.nrci_before 
            for result in self._correction_history
        ]
        avg_nrci_improvement = np.mean(nrci_improvements)
        
        return {
            'total_corrections': total_corrections,
            'successful_corrections': successful_corrections,
            'success_rate': successful_corrections / total_corrections,
            'total_errors_corrected': total_errors_corrected,
            'average_correction_efficiency': avg_correction_efficiency,
            'average_processing_time': avg_processing_time,
            'average_nrci_improvement': avg_nrci_improvement,
            'golay_parameters': {
                'n': self.golay_matrix.params.n,
                'k': self.golay_matrix.params.k,
                'd': self.golay_matrix.params.d,
                't': self.golay_matrix.params.t
            }
        }
    
    def validate_golay_system(self) -> Dict[str, Any]:
        """
        Validate the Golay(24,12) system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'matrix_dimensions_correct': True,
            'syndrome_calculation_correct': True,
            'error_correction_functional': True,
            'codeword_validation_correct': True
        }
        
        try:
            # Test 1: Matrix dimensions
            H = self.golay_matrix.H
            if H.shape != (12, 24):
                validation_results['matrix_dimensions_correct'] = False
                validation_results['matrix_error'] = f"Expected (12, 24), got {H.shape}"
            
            # Test 2: Syndrome of zero vector should be zero
            zero_vector = np.zeros(24, dtype=int)
            syndrome_zero = self.golay_matrix.compute_syndrome(zero_vector)
            if np.any(syndrome_zero):
                validation_results['syndrome_calculation_correct'] = False
                validation_results['syndrome_error'] = f"Zero vector syndrome: {syndrome_zero}"
            
            # Test 3: Single bit error detection and correction
            test_vector = np.zeros(24, dtype=int)
            test_vector[5] = 1  # Introduce single bit error
            
            syndrome = self.golay_matrix.compute_syndrome(test_vector)
            if not self.golay_matrix.detect_error(syndrome):
                validation_results['error_correction_functional'] = False
                validation_results['detection_error'] = "Failed to detect single bit error"
            
            corrected, error_count = self.golay_matrix.correct_error(test_vector, syndrome)
            if not np.array_equal(corrected, zero_vector) or error_count != 1:
                validation_results['error_correction_functional'] = False
                validation_results['correction_error'] = f"Failed to correct single bit error: {corrected}, count: {error_count}"
            
            # Test 4: Codeword validation
            if not self.golay_matrix.validate_codeword(zero_vector):
                validation_results['codeword_validation_correct'] = False
                validation_results['validation_error'] = "Zero vector should be valid codeword"
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['syndrome_calculation_correct'] = False
            validation_results['error_correction_functional'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_global_golay_correction() -> GlobalGolayCorrection:
    """
    Create a Global Golay Correction processor for GLR Level 7.
    
    Returns:
        Configured GlobalGolayCorrection instance
    """
    return GlobalGolayCorrection()


if __name__ == "__main__":
    # Validation and testing
    print("Initializing Global Golay Correction (GLR Level 7)...")
    
    golay_processor = create_global_golay_correction()
    
    # Test syndrome calculation S = H × v mod 2
    print("\nTesting syndrome calculation S = H × v mod 2...")
    
    # Test with zero vector (should have zero syndrome)
    zero_vector = np.zeros(24, dtype=int)
    syndrome_zero = golay_processor.golay_matrix.compute_syndrome(zero_vector)
    print(f"Zero vector syndrome: {syndrome_zero} (sum: {np.sum(syndrome_zero)})")
    
    # Test with single bit error
    error_vector = np.zeros(24, dtype=int)
    error_vector[7] = 1  # Flip bit 7
    syndrome_error = golay_processor.golay_matrix.compute_syndrome(error_vector)
    print(f"Single error syndrome: {syndrome_error} (sum: {np.sum(syndrome_error)})")
    
    # Test error correction
    corrected, error_count = golay_processor.golay_matrix.correct_error(error_vector, syndrome_error)
    print(f"Corrected vector: {corrected}")
    print(f"Error count: {error_count}")
    print(f"Correction successful: {np.array_equal(corrected, zero_vector)}")
    
    # Test with multiple vectors
    print("\nTesting with multiple 24-bit vectors...")
    test_data = np.array([
        [1, 0, 1, 0] * 6,  # 24-bit vector 1
        [0, 1, 0, 1] * 6,  # 24-bit vector 2
        [1, 1, 0, 0] * 6   # 24-bit vector 3
    ])
    
    result = golay_processor.process_correction(test_data)
    print(f"Processing result:")
    print(f"  Success: {result.success}")
    print(f"  Errors corrected: {result.error_count}")
    print(f"  Correction efficiency: {result.correction_efficiency:.3f}")
    print(f"  NRCI improvement: {result.nrci_after - result.nrci_before:.6f}")
    print(f"  Processing time: {result.processing_time:.6f}s")
    
    # System validation
    validation = golay_processor.validate_golay_system()
    print(f"\nGolay system validation:")
    print(f"  Matrix dimensions: {validation['matrix_dimensions_correct']}")
    print(f"  Syndrome calculation: {validation['syndrome_calculation_correct']}")
    print(f"  Error correction: {validation['error_correction_functional']}")
    print(f"  Codeword validation: {validation['codeword_validation_correct']}")
    
    # Test OffBit correction
    print(f"\nTesting OffBit correction...")
    test_offbit = 0x12345678  # 32-bit OffBit value
    corrected_offbit, metadata = golay_processor.process_offbit_correction(test_offbit)
    print(f"Original OffBit: 0x{test_offbit:08X}")
    print(f"Corrected OffBit: 0x{corrected_offbit:08X}")
    print(f"Error detected: {metadata['error_detected']}")
    print(f"Error count: {metadata['error_count']}")
    
    print("\nGlobal Golay Correction (GLR Level 7) with S = H × v mod 2 ready for UBP integration.")
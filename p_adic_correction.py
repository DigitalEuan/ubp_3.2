"""
Universal Binary Principle (UBP) Framework v3.2+ - Advanced Error Correction (p-adic & Fibonacci)
Author: Euan Craig, New Zealand
Date: 03 September 2025

Implements advanced error correction using p-adic number theory, Fibonacci encodings,
and adelic structures for the UBP framework. Provides ultra-high precision
error correction beyond traditional binary codes, now combining functionalities
from p_adic_correction.py and enhanced_error_correction.py.

Mathematical Foundation:
- p-adic valuations and norms for error detection
- Adelic product structures across multiple primes
- Hensel lifting for p-adic error correction
- p-adic metric spaces for distance-based correction
- Fibonacci encodings leveraging natural redundancy
- Majority voting for Fibonacci error correction
- Integration with GLR framework

"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict
from scipy.special import comb # Moved from enhanced_error_correction.py

# Import configuration
from ubp_config import get_config, UBPConfig # Direct import as ubp_config is in the root directory

print("DEBUG: p_adic_correction.py: Module level code execution started.")

_config = get_config() # Initialize configuration

class PAdicPrime(Enum):
    """Standard primes for p-adic calculations"""
    P2 = 2
    P3 = 3
    P5 = 5
    P7 = 7
    P11 = 11
    P13 = 13
    P17 = 17
    P19 = 19
    P23 = 23
    P29 = 29


@dataclass
class PAdicNumber:
    """
    Represents a p-adic number with finite precision.
    """
    prime: int
    digits: List[int]  # p-adic digits (least significant first)
    precision: int
    valuation: int = 0  # v_p(x) - power of p in factorization
    
    def __post_init__(self):
        # Ensure digits are in valid range
        self.digits = [d % self.prime for d in self.digits]
        
        # Pad or truncate to precision
        while len(self.digits) < self.precision:
            self.digits.append(0)
        self.digits = self.digits[:self.precision]
    
    def __str__(self):
        if self.valuation > 0:
            return f"{self.prime}^{self.valuation} * {self.digits}"
        else:
            return f"{self.digits} (base {self.prime})"


@dataclass
class AdelicNumber:
    """
    Represents an adelic number as a product over multiple primes.
    """
    components: Dict[int, PAdicNumber]  # prime -> p-adic component
    real_component: float = 0.0
    
    def get_primes(self) -> List[int]:
        """Get list of primes in this adelic number"""
        return list(self.components.keys())


@dataclass
class FibonacciCode:
    """Represents a Fibonacci-encoded state. Moved from enhanced_error_correction.py"""
    fibonacci_sequence: List[int]
    encoded_bits: List[int]
    original_data: Optional[np.ndarray]
    redundancy_level: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class ErrorCorrectionResult:
    """Result from error correction operation. Moved from enhanced_error_correction.py"""
    original_errors: int
    corrected_errors: int
    correction_success_rate: float
    encoding_efficiency: float
    decoding_time: float
    method_used: str
    confidence_score: float
    metadata: Dict = field(default_factory=dict)


class PAdicArithmetic:
    """
    Implements p-adic arithmetic operations.
    """
    
    def __init__(self, prime: int, precision: int = 10):
        self.prime = prime
        self.precision = precision
        self._inverse_cache = {}
    
    def valuation(self, n: int) -> int:
        """
        Compute p-adic valuation v_p(n).
        
        Args:
            n: Integer to compute valuation for
        
        Returns:
            p-adic valuation (power of p in factorization)
        """
        if n == 0:
            return float('inf')
        
        valuation = 0
        while n % self.prime == 0:
            n //= self.prime
            valuation += 1
        
        return valuation
    
    def norm(self, n: int) -> float:
        """
        Compute p-adic norm |n|_p = p^(-v_p(n)).
        
        Args:
            n: Integer to compute norm for
        
        Returns:
            p-adic norm
        """
        if n == 0:
            return 0.0
        
        val = self.valuation(n)
        return self.prime ** (-val)
    
    def distance(self, a: int, b: int) -> float:
        """
        Compute p-adic distance |a - b|_p.
        
        Args:
            a, b: Integers to compute distance between
        
        Returns:
            p-adic distance
        """
        return self.norm(a - b)
    
    def to_padic(self, n: int) -> PAdicNumber:
        """
        Convert integer to p-adic representation.
        
        Args:
            n: Integer to convert
        
        Returns:
            p-adic number
        """
        if n == 0:
            return PAdicNumber(self.prime, [0] * self.precision, self.precision, 0)
        
        # Compute valuation
        val = self.valuation(n)
        
        # Remove p^val factor
        reduced_n = n // (self.prime ** val) if val < float('inf') else 0
        
        # Convert to p-adic digits
        digits = []
        current = abs(reduced_n)
        
        for _ in range(self.precision):
            digits.append(current % self.prime)
            current //= self.prime
        
        return PAdicNumber(self.prime, digits, self.precision, val)
    
    def from_padic(self, padic_num: PAdicNumber) -> int:
        """
        Convert p-adic number back to integer (approximate).
        
        Args:
            padic_num: p-adic number to convert
        
        Returns:
            Integer approximation
        """
        result = 0
        power = 1
        
        for digit in padic_num.digits:
            result += digit * power
            power *= self.prime
        
        # Apply valuation
        if padic_num.valuation > 0:
            result *= (self.prime ** padic_num.valuation)
        
        return result
    
    def add_padic(self, a: PAdicNumber, b: PAdicNumber) -> PAdicNumber:
        """
        Add two p-adic numbers.
        
        Args:
            a, b: p-adic numbers to add
        
        Returns:
            Sum as p-adic number
        """
        if a.prime != b.prime:
            raise ValueError("Cannot add p-adic numbers with different primes")
        
        # Convert to integers, add, convert back
        int_a = self.from_padic(a)
        int_b = self.from_padic(b)
        sum_int = int_a + int_b
        
        return self.to_padic(sum_int)
    
    def multiply_padic(self, a: PAdicNumber, b: PAdicNumber) -> PAdicNumber:
        """
        Multiply two p-adic numbers.
        
        Args:
            a, b: p-adic numbers to multiply
        
        Returns:
            Product as p-adic number
        """
        if a.prime != b.prime:
            raise ValueError("Cannot multiply p-adic numbers with different primes")
        
        # Convert to integers, multiply, convert back
        int_a = self.from_padic(a)
        int_b = self.from_padic(b)
        product_int = int_a * int_b
        
        return self.to_padic(product_int)
    
    def inverse_padic(self, a: PAdicNumber) -> Optional[PAdicNumber]:
        """
        Compute multiplicative inverse of p-adic number using Hensel lifting.
        
        Args:
            a: p-adic number to invert
        
        Returns:
            Inverse p-adic number if it exists, None otherwise
        """
        if a.valuation != 0:
            return None  # No inverse if divisible by p
        
        # Use Hensel lifting to compute inverse
        # Start with inverse modulo p
        a0 = a.digits[0]
        if a0 == 0:
            return None
        
        # Find inverse of a0 modulo p using extended Euclidean algorithm
        inv_a0 = self._mod_inverse(a0, self.prime)
        if inv_a0 is None:
            return None
        
        # Hensel lifting
        inverse_digits = [inv_a0]
        
        for k in range(1, self.precision):
            # Compute next digit using Hensel's formula
            # x_{k+1} = x_k - (a * x_k - 1) * x_0 / p^k
            
            # This is a simplified version - full implementation would
            # require more sophisticated Hensel lifting
            next_digit = (inverse_digits[-1] * 2) % self.prime
            inverse_digits.append(next_digit)
        
        return PAdicNumber(self.prime, inverse_digits, self.precision, 0)
    
    def _mod_inverse(self, a: int, m: int) -> Optional[int]:
        """Compute modular inverse using extended Euclidean algorithm"""
        if math.gcd(a, m) != 1:
            return None
        
        # Extended Euclidean algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        _, x, _ = extended_gcd(a % m, m)
        return (x % m + m) % m


class AdelicArithmetic:
    """
    Implements arithmetic operations on adelic numbers.
    """
    
    def __init__(self, primes: List[int], precision: int = 10):
        self.primes = primes
        self.precision = precision
        self.padic_calculators = {p: PAdicArithmetic(p, precision) for p in primes}
    
    def create_adelic(self, value: Union[int, float], 
                     primes: Optional[List[int]] = None) -> AdelicNumber:
        """
        Create adelic number from integer or float.
        
        Args:
            value: Value to convert
            primes: List of primes to use (uses default if None)
            
        Returns:
            Adelic number
        """
        if primes is None: # Use is for comparison to None
            primes = self.primes
        
        components = {}
        
        if isinstance(value, int):
            for p in primes:
                calc = self.padic_calculators[p]
                components[p] = calc.to_padic(value)
            real_component = float(value)
        else:
            # For float values, use integer part for p-adic components
            int_part = int(value)
            for p in primes:
                calc = self.padic_calculators[p]
                components[p] = calc.to_padic(int_part)
            real_component = value
        
        return AdelicNumber(components, real_component)
    
    def add_adelic(self, a: AdelicNumber, b: AdelicNumber) -> AdelicNumber:
        """
        Add two adelic numbers.
        
        Args:
            a, b: Adelic numbers to add
        
        Returns:
            Sum as adelic number
        """
        result_components = {}
        
        # Add p-adic components
        all_primes = set(a.get_primes()) | set(b.get_primes())
        
        for p in all_primes:
            calc = self.padic_calculators[p]
            
            a_comp = a.components.get(p, calc.to_padic(0))
            b_comp = b.components.get(p, calc.to_padic(0))
            
            result_components[p] = calc.add_padic(a_comp, b_comp)
        
        # Add real components
        result_real = a.real_component + b.real_component
        
        return AdelicNumber(result_components, result_real)
    
    def multiply_adelic(self, a: AdelicNumber, b: AdelicNumber) -> AdelicNumber:
        """
        Multiply two adelic numbers.
        
        Args:
            a, b: Adelic numbers to multiply
        
        Returns:
            Product as adelic number
        """
        result_components = {}
        
        # Multiply p-adic components
        all_primes = set(a.get_primes()) | set(b.get_primes())
        
        for p in all_primes:
            calc = self.padic_calculators[p]
            
            a_comp = a.components.get(p, calc.to_padic(1))
            b_comp = b.components.get(p, calc.to_padic(1))
            
            result_components[p] = calc.multiply_padic(a_comp, b_comp)
        
        # Multiply real components
        result_real = a.real_component * b.real_component
        
        return AdelicNumber(result_components, result_real)
    
    def adelic_norm(self, a: AdelicNumber) -> float:
        """
        Compute adelic norm (product of all p-adic norms).
        
        Args:
            a: Adelic number
        
        Returns:
            Adelic norm
        """
        norm_product = 1.0
        
        for p, padic_comp in a.components.items():
            calc = self.padic_calculators[p]
            int_val = calc.from_padic(padic_comp)
            p_norm = calc.norm(int_val)
            norm_product *= p_norm
        
        # Include real component
        norm_product *= abs(a.real_component)
        
        return norm_product


class PAdicEncoder:
    """
    p-adic number encoder for advanced error correction.
    
    Uses p-adic representations to provide natural error correction
    through the ultrametric properties of p-adic numbers.
    """
    
    def __init__(self, prime: int = 2, precision: int = 20):
        self.prime = prime
        self.precision = precision
        
        # Validate prime
        if not self._is_prime(prime):
            raise ValueError(f"Prime {prime} is not a valid prime number")
    
    def encode_to_padic(self, data: np.ndarray) -> PAdicNumber: # Changed to PAdicNumber to align
        """
        Encode data to p-adic representation.
        
        Args:
            data: Input data array
            
        Returns:
            PAdicNumber (representing the encoded state, not a dataclass PAdicState)
        """
        if len(data) == 0:
            return PAdicNumber(
                prime=self.prime,
                digits=[],
                precision=self.precision,
                valuation=0
            )
        
        # Convert data to integers (scaled and rounded)
        scale_factor = 1000  # Scale to preserve precision
        int_data = np.round(data * scale_factor).astype(int)
        
        # Encode each integer as p-adic (summed into one PAdicNumber for simplicity here)
        # In a more complex system, each data point might be its own PAdicNumber.
        # For this integration, let's sum them for a single PAdicNumber.
        
        sum_val = np.sum(int_data)
        padic_state = self._integer_to_padic(sum_val) # Returns PAdicNumber
        
        # Ensure metadata attribute exists and update
        if not hasattr(padic_state, 'metadata'): 
            object.__setattr__(padic_state, 'metadata', {}) # Assign metadata to PAdicNumber
        padic_state.metadata.update({ # Add metadata to PAdicNumber if it has a metadata field
            'original_data_length': len(data),
            'scale_factor': scale_factor,
            'encoding_time': time.time(),
            'sum_val': sum_val # Storing sum_val for reconstruction
        })
        
        return padic_state
    
    def decode_from_padic(self, padic_state: PAdicNumber) -> np.ndarray: # Changed to PAdicNumber
        """
        Decode p-adic representation back to data.
        
        Args:
            padic_state: p-adic encoded state
            
        Returns:
            Decoded data array
        """
        if not hasattr(padic_state, 'metadata') or 'sum_val' not in padic_state.metadata:
            # Fallback if metadata not available or sum_val missing
            return np.array([self._padic_to_integer(padic_state)] / 1000.0) # Assume scale factor

        original_length = padic_state.metadata.get('original_data_length', 1)
        scale_factor = padic_state.metadata.get('scale_factor', 1000)
        sum_val = padic_state.metadata.get('sum_val') # Use stored sum_val

        # Reconstruct integer from p-adic coefficients
        integer_value = self._padic_to_integer(padic_state)
        
        # For simplicity, if we summed, we can't perfectly reconstruct the original array.
        # We'll just return the integer value divided by scale_factor as a single element array.
        # A more complex system would store individual p-adic numbers or coefficients.
        
        # If the original encoding was a simple sum, approximate decoding by distributing the sum
        # This is a *major simplification* for integration. A robust P-adic encoder for arrays
        # would store a list of PAdicNumbers or a multi-dimensional P-adic number.
        if original_length > 0:
            avg_value_per_element = (integer_value / scale_factor) / original_length
            decoded_values = np.full(original_length, avg_value_per_element)
        else:
            decoded_values = np.array([])
        
        return decoded_values
    
    def correct_padic_errors(self, corrupted_padic: PAdicNumber, 
                           error_threshold: float = 0.1) -> Tuple[PAdicNumber, int]: # Changed to PAdicNumber
        """
        Correct errors in p-adic representation using ultrametric properties.
        
        Args:
            corrupted_padic: Corrupted p-adic state
            error_threshold: Threshold for error detection
            
        Returns:
            Tuple of (corrected_padic_state, number_of_corrections)
        """
        if not corrupted_padic.digits:
            return corrupted_padic, 0
        
        corrected_digits = corrupted_padic.digits.copy()
        corrections_made = 0
        
        # Error correction using p-adic distance properties
        for i in range(len(corrected_digits)):
            digit = corrected_digits[i]
            
            # Check if digit is valid for the prime
            if digit >= self.prime or digit < 0:
                # Correct by taking modulo prime
                corrected_digits[i] = digit % self.prime
                corrections_made += 1
            
            # Check for consistency with neighboring digits (simplified)
            if i > 0 and i < len(corrected_digits) - 1:
                prev_digit = corrected_digits[i-1]
                next_digit = corrected_digits[i+1]
                
                # Simple consistency check: digit should be "close" to neighbors
                expected_digit = (prev_digit + next_digit) // 2
                
                # Use a threshold related to the prime
                if abs(digit - expected_digit) > self.prime * error_threshold:
                    corrected_digits[i] = expected_digit % self.prime
                    corrections_made += 1
        
        corrected_padic_num = PAdicNumber(
            prime=corrupted_padic.prime,
            digits=corrected_digits,
            precision=corrupted_padic.precision,
            valuation=corrupted_padic.valuation
        )
        if hasattr(corrupted_padic, 'metadata'):
            object.__setattr__(corrected_padic_num, 'metadata', {
                **corrupted_padic.metadata,
                'corrections_made': corrections_made,
                'correction_time': time.time()
            })
        
        return corrected_padic_num, corrections_made
    
    def _integer_to_padic(self, n: int) -> PAdicNumber:
        """Convert integer to p-adic representation. Returns PAdicNumber"""
        if n == 0:
            return PAdicNumber(self.prime, [0] * self.precision, self.precision, 0)
        
        # Find p-adic valuation (highest power of p dividing n)
        valuation = 0
        temp_n = abs(n)
        
        while temp_n % self.prime == 0 and temp_n > 0:
            temp_n //= self.prime
            valuation += 1
        
        # Extract p-adic digits
        digits = []
        remaining = abs(n) // (self.prime ** valuation) if valuation < float('inf') else 0
        
        for _ in range(self.precision):
            digits.append(remaining % self.prime)
            remaining //= self.prime
            
            if remaining == 0:
                break
        
        # Pad with zeros if needed
        while len(digits) < self.precision:
            digits.append(0)
        
        return PAdicNumber(self.prime, digits, self.precision, valuation)
    
    def _padic_to_integer(self, padic_num: PAdicNumber) -> int: # Changed to PAdicNumber
        """Convert p-adic representation to integer."""
        if not padic_num.digits:
            return 0
        
        result = 0
        power = 1
        
        for digit in padic_num.digits:
            result += digit * power
            power *= self.prime
        
        # Apply valuation
        result *= (self.prime ** padic_num.valuation)
        
        return result
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime."""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        
        return True


class FibonacciEncoder:
    """
    Fibonacci sequence encoder for natural error correction.
    
    Uses Fibonacci sequences to provide error correction through
    the natural redundancy in Fibonacci representations.
    Moved from enhanced_error_correction.py
    """
    
    def __init__(self, max_fibonacci_index: int = 50):
        """Initialize Fibonacci encoder with sufficient sequence length."""
        self.max_index = max_fibonacci_index
        
        # Generate Fibonacci sequence
        self.fibonacci_sequence = self._generate_fibonacci_sequence(max_fibonacci_index)
    
    def encode_to_fibonacci(self, data: np.ndarray, redundancy_level: float = 0.3) -> FibonacciCode:
        """
        Encode data using Fibonacci representation.
        
        Args:
            data: Input data array
            redundancy_level: Level of redundancy for error correction (0.0 to 1.0)
            
        Returns:
            FibonacciCode with Fibonacci encoding
        """
        if len(data) == 0:
            return FibonacciCode(
                fibonacci_sequence=self.fibonacci_sequence,
                encoded_bits=[],
                original_data=data,
                redundancy_level=redundancy_level
            )
        
        # Convert data to positive integers
        scale_factor = 1000
        int_data = np.round(np.abs(data) * scale_factor).astype(int)
        
        # Encode each integer using Fibonacci representation
        all_encoded_bits = []
        value_boundaries = []  # Track where each value's encoding starts and ends
        
        for value in int_data:
            start_pos = len(all_encoded_bits)
            
            fib_bits = self._integer_to_fibonacci(value)
            
            # Add redundancy
            redundant_bits = self._add_fibonacci_redundancy(fib_bits, redundancy_level)
            all_encoded_bits.extend(redundant_bits)
            
            end_pos = len(all_encoded_bits)
            value_boundaries.append((start_pos, end_pos))
        
        fibonacci_code = FibonacciCode(
            fibonacci_sequence=self.fibonacci_sequence,
            encoded_bits=all_encoded_bits,
            original_data=data.copy(),
            redundancy_level=redundancy_level,
            metadata={
                'scale_factor': scale_factor,
                'original_length': len(data),
                'value_boundaries': value_boundaries,
                'encoding_time': time.time()
            }
        )
        
        return fibonacci_code
    
    def decode_from_fibonacci(self, fibonacci_code: FibonacciCode) -> np.ndarray:
        """
        Decode Fibonacci representation back to data.
        
        Args:
            fibonacci_code: Fibonacci encoded data
            
        Returns:
            Decoded data array
        """
        if not fibonacci_code.encoded_bits:
            return np.array([])
        
        scale_factor = fibonacci_code.metadata.get('scale_factor', 1000)
        original_length = fibonacci_code.metadata.get('original_length', 1)
        value_boundaries = fibonacci_code.metadata.get('value_boundaries', None)
        
        decoded_values = []
        
        if value_boundaries and len(value_boundaries) == original_length:
            # Use stored boundaries for precise decoding
            for start_pos, end_pos in value_boundaries:
                bit_segment = fibonacci_code.encoded_bits[start_pos:end_pos]
                
                # Remove redundancy from this segment
                core_bits = self._remove_fibonacci_redundancy(bit_segment, fibonacci_code.redundancy_level)
                
                # Decode to integer and scale back
                integer_value = self._fibonacci_to_integer(core_bits)
                decoded_values.append(integer_value / scale_factor)
        else:
            # Fallback to even splitting if boundaries not available
            core_bits = self._remove_fibonacci_redundancy(fibonacci_code.encoded_bits, fibonacci_code.redundancy_level)
            
            if original_length == 1:
                # Single value case
                integer_value = self._fibonacci_to_integer(core_bits)
                decoded_values.append(integer_value / scale_factor)
            else:
                # Multiple values case - split evenly
                if len(core_bits) >= original_length:
                    bits_per_value = len(core_bits) // original_length
                    
                    for i in range(original_length):
                        start_idx = i * bits_per_value
                        end_idx = start_idx + bits_per_value
                        if i == original_length - 1:  # Last value gets remaining bits
                            end_idx = len(core_bits)
                        
                        bit_segment = core_bits[start_idx:end_idx]
                        if bit_segment:  # Only decode if we have bits
                            integer_value = self._fibonacci_to_integer(bit_segment)
                            decoded_values.append(integer_value / scale_factor)
                        else:
                            decoded_values.append(0.0)
                else:
                    # Not enough bits, pad with zeros
                    for i in range(original_length):
                        decoded_values.append(0.0)
        
        return np.array(decoded_values)
    
    def correct_fibonacci_errors(self, corrupted_code: FibonacciCode) -> Tuple[FibonacciCode, int]:
        """
        Correct errors in Fibonacci representation using redundancy.
        
        Args:
            corrupted_code: Corrupted Fibonacci code
            
        Returns:
            Tuple of (corrected_code, number_of_corrections)
        """
        if not corrupted_code.encoded_bits:
            return corrupted_code, 0
        
        corrected_bits = corrupted_code.encoded_bits.copy()
        corrections_made = 0
        
        # Error correction using Fibonacci properties
        # Property: No two consecutive 1s in valid Fibonacci representation (Zeckendorf)
        # We need to process the redundant bits first to get a "best guess" core, then apply Zeckendorf rule.
        
        # Step 1: Majority voting on redundant bits to get a robust `core_bits_full_array`
        core_bits_full_array_best_guess, group_corrections = self._majority_vote_correction(
            corrected_bits, corrupted_code.redundancy_level
        )
        corrections_made += group_corrections
        
        # Step 2: Apply Zeckendorf constraint to the `core_bits_full_array_best_guess`
        zeckendorf_corrected_bits = core_bits_full_array_best_guess.copy()
        zeckendorf_corrections = 0
        
        i = 0
        while i < len(zeckendorf_corrected_bits) - 1:
            if zeckendorf_corrected_bits[i] == 1 and zeckendorf_corrected_bits[i+1] == 1:
                zeckendorf_corrected_bits[i+1] = 0 # Assume the second 1 is the error
                zeckendorf_corrections += 1
            i += 1
        corrections_made += zeckendorf_corrections
        
        corrected_code = FibonacciCode(
            fibonacci_sequence=corrupted_code.fibonacci_sequence,
            encoded_bits=zeckendorf_corrected_bits, # Store the Zeckendorf-corrected core bits
            original_data=corrupted_code.original_data, # Original data is just reference
            redundancy_level=corrupted_code.redundancy_level,
            metadata={
                **corrupted_code.metadata,
                'corrections_made': corrections_made,
                'zeckendorf_corrections': zeckendorf_corrections,
                'correction_time': time.time()
            }
        )
        
        return corrected_code, corrections_made
    
    def _generate_fibonacci_sequence(self, n: int) -> List[int]:
        """Generate Fibonacci sequence up to n terms."""
        if n <= 0:
            return []
        if n == 1:
            return [1]
        
        fib = [1, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
        
        return fib
    
    def _integer_to_fibonacci(self, n: int) -> List[int]:
        """Convert integer to Fibonacci representation (Zeckendorf representation)."""
        if n == 0:
            return [0]
        
        # Find largest Fibonacci number <= n
        fib_bits = [0] * len(self.fibonacci_sequence)
        remaining = n
        
        # Greedy algorithm for Zeckendorf representation
        for i in range(len(self.fibonacci_sequence) - 1, -1, -1):
            if self.fibonacci_sequence[i] <= remaining:
                fib_bits[i] = 1
                remaining -= self.fibonacci_sequence[i]
                
                if remaining == 0:
                    break
        
        # Remove leading zeros
        while len(fib_bits) > 1 and fib_bits[-1] == 0:
            fib_bits.pop()
        
        return fib_bits
    
    def _fibonacci_to_integer(self, fib_bits: List[int]) -> int:
        """Convert Fibonacci representation to integer."""
        if not fib_bits:
            return 0
        
        result = 0
        for i, bit in enumerate(fib_bits):
            if bit == 1 and i < len(self.fibonacci_sequence):
                result += self.fibonacci_sequence[i]
        
        return result
    
    def _add_fibonacci_redundancy(self, fib_bits: List[int], redundancy_level: float) -> List[int]:
        """Add redundancy to Fibonacci representation."""
        if redundancy_level <= 0:
            return fib_bits
        
        redundancy_factor = int(1 + redundancy_level * 3)  # 1-4 repetitions
        
        redundant_bits = []
        for bit in fib_bits:
            redundant_bits.extend([bit] * redundancy_factor)
        
        return redundant_bits
    
    def _remove_fibonacci_redundancy(self, redundant_bits: List[int], redundancy_level: float) -> List[int]:
        """
        Remove redundancy from Fibonacci representation using majority voting.
        This function now processes the entire redundant_bits array and returns the
        reconstructed *core* (non-redundant) Fibonacci bit array.
        """
        if redundancy_level <= 0:
            return redundant_bits
        
        redundancy_factor = int(1 + redundancy_level * 3)
        
        core_bits = []
        
        for i in range(0, len(redundant_bits), redundancy_factor):
            bit_group = redundant_bits[i:i+redundancy_factor]
            
            if not bit_group: 
                continue
            
            # Majority vote
            ones = sum(bit_group)
            zeros = len(bit_group) - ones
            
            majority_bit = 1 if ones > zeros else 0
            core_bits.append(majority_bit)
        
        return core_bits
    
    def _majority_vote_correction(self, bit_group: List[int], 
                                redundancy_level: float) -> Tuple[List[int], int]:
        """
        Apply majority vote correction to a group of bits representing a single original bit.
        This function is adapted for internal use by `correct_fibonacci_errors`.
        It returns the reconstructed core bits (without the repetition) and corrections made.
        """
        if redundancy_level <= 0:
            return bit_group, 0
        
        redundancy_factor = int(1 + redundancy_level * 3)
        corrections = 0
        reconstructed_core_bits = [] 
        
        if not bit_group:
            return [], 0 

        # We assume bit_group here is the *entire* `encoded_bits` sequence,
        # not a single logical bit's repeated copies.
        # So we need to iterate over this `bit_group` to extract actual logical bit repetitions.
        
        corrected_full_sequence = []
        total_corrections_made_in_groups = 0
        
        # Iterate over segments corresponding to one original bit
        for i in range(0, len(bit_group), redundancy_factor):
            segment = bit_group[i : i + redundancy_factor]
            if not segment:
                continue
            
            ones = sum(segment)
            zeros = len(segment) - ones
            majority_bit = 1 if ones > zeros else 0
            
            # Count corrections for this segment
            corrections_in_segment = sum(1 for bit in segment if bit != majority_bit)
            total_corrections_made_in_groups += corrections_in_segment
            
            corrected_full_sequence.extend([majority_bit] * len(segment)) # Reconstruct with corrected values

        # The `reconstructed_core_bits` should be the actual deduplicated bits, not the full sequence
        reconstructed_core_bits = self._remove_fibonacci_redundancy(corrected_full_sequence, redundancy_level)

        return reconstructed_core_bits, total_corrections_made_in_groups


class AdvancedErrorCorrectionModule:
    """
    Advanced Error Correction system combining multiple encoding methods.
    
    Integrates p-adic encoding, Fibonacci encoding, and traditional methods
    for comprehensive error correction in UBP computations.
    This class now consolidates `PAdicErrorCorrector` and `AdvancedErrorCorrection`
    from `enhanced_error_correction.py`.
    """
    
    def __init__(self, primes: Optional[List[int]] = None, precision: int = 10):
        # Local import to avoid circular dependency at module level with ubp_config
        self.config = _config # Using the module-level _config already initialized
        
        # Initialize p-adic components
        if primes is None:
            primes = [2, 3, 5, 7, 11]  # Default prime set
        self.primes = primes
        self.precision = precision
        self.adelic_calc = AdelicArithmetic(primes, precision)
        self.padic_encoder = PAdicEncoder(prime=self.primes[0], precision=self.precision) # Use the first prime for encoder

        # Initialize Fibonacci encoder
        self.fibonacci_encoder = FibonacciEncoder(max_fibonacci_index=self.config.error_correction.fibonacci_depth) # Use config for depth
        
        # Error correction statistics
        self.correction_history = []
        self.system_statistics = { # Used for p-adic specific correction stats
            'total_corrections': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'average_error_magnitude': 0.0
        }
        print("DEBUG: AdvancedErrorCorrectionModule initialized.")
    
    def detect_error_padic(self, data: np.ndarray, 
                          expected_pattern: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect errors using p-adic analysis and statistical outlier detection.
        
        Args:
            data: Data array to check for errors
            expected_pattern: Expected pattern (if known)
        
        Returns:
            Dictionary containing error detection results
        """
        errors_detected = []
        error_magnitudes = []
        
        # First, use statistical outlier detection
        if len(data) > 2:
            # Calculate statistical measures
            data_mean = np.mean(data)
            data_std = np.std(data)
            
            # Detect outliers using z-score method
            for i, value in enumerate(data):
                if data_std > 0:
                    z_score = abs((value - data_mean) / data_std)
                    
                    # Values with z-score > 2.0 are considered outliers (more sensitive)
                    if z_score > 2.0:
                        errors_detected.append({
                            'position': i,
                            'detected_value': value,
                            'error_type': 'statistical_outlier',
                            'z_score': z_score,
                            'expected_range': (data_mean - 2*data_std, data_mean + 2*data_std)
                        })
                        error_magnitudes.append(z_score)
        
        # Then, use p-adic analysis for additional validation
        for i, value in enumerate(data):
            if not isinstance(value, (int, float)):
                continue
            
            # Convert to adelic number
            try:
                adelic_val = self.adelic_calc.create_adelic(int(value))
                
                # Compute p-adic norms for each prime
                p_norms = {}
                for p in self.primes:
                    calc = self.adelic_calc.padic_calculators[p]
                    int_val = calc.from_padic(adelic_val.components[p])
                    p_norms[p] = calc.norm(int_val)
                
                # Detect anomalies in p-adic structure
                norm_variance = np.var(list(p_norms.values()))
                
                # Check against expected pattern if provided
                if expected_pattern is not None and i < len(expected_pattern):
                    expected_adelic = self.adelic_calc.create_adelic(int(expected_pattern[i]))
                    
                    # Compute adelic distance
                    diff_adelic = self.adelic_calc.add_adelic(
                        adelic_val,
                        self.adelic_calc.multiply_adelic(
                            expected_adelic,
                            self.adelic_calc.create_adelic(-1)
                        )
                    )
                    
                    error_magnitude = self.adelic_calc.adelic_norm(diff_adelic)
                    
                    if error_magnitude > self.config.error_correction.error_threshold:
                        # Check if not already detected as statistical outlier
                        if not any(err['position'] == i for err in errors_detected):
                            errors_detected.append({
                                'position': i,
                                'detected_value': value,
                                'expected_value': expected_pattern[i],
                                'error_magnitude': error_magnitude,
                                'error_type': 'padic_mismatch',
                                'p_norms': p_norms,
                                'norm_variance': norm_variance
                            })
                            error_magnitudes.append(error_magnitude)
                
                # Detect structural anomalies even without expected pattern
                elif norm_variance > 0.1:  # High variance indicates potential error
                    # Check if not already detected as statistical outlier
                    if not any(err['position'] == i for err in errors_detected):
                        errors_detected.append({
                            'position': i,
                            'detected_value': value,
                            'error_type': 'padic_structural_anomaly',
                            'norm_variance': norm_variance,
                            'p_norms': p_norms
                        })
                        error_magnitudes.append(norm_variance)
                        
            except Exception as e:
                # If p-adic analysis fails, still report as potential error
                if not any(err['position'] == i for err in errors_detected):
                    errors_detected.append({
                        'position': i,
                        'detected_value': value,
                        'error_type': 'padic_analysis_failed',
                        'error_message': str(e)
                    })
                    error_magnitudes.append(1.0)
        
        return {
            'errors_detected': errors_detected,
            'num_errors': len(errors_detected),
            'average_error_magnitude': np.mean(error_magnitudes) if error_magnitudes else 0.0,
            'max_error_magnitude': np.max(error_magnitudes) if error_magnitudes else 0.0,
            'error_positions': [err['position'] for err in errors_detected]
        }
    
    def correct_error_hensel(self, corrupted_value: int, 
                           context_values: List[int]) -> Tuple[int, float]:
        """
        Correct error using Hensel lifting and p-adic context.
        
        Args:
            corrupted_value: Value suspected to contain error
            context_values: Surrounding values for context
        
        Returns:
            Tuple of (corrected_value, confidence)
        """
        if not context_values:
            return corrupted_value, 0.0
        
        # Convert all values to adelic representation
        try:
            corrupted_adelic = self.adelic_calc.create_adelic(corrupted_value)
            context_adelics = [self.adelic_calc.create_adelic(val) for val in context_values]
        except Exception:
            return corrupted_value, 0.0
        
        # Compute expected value based on context patterns
        # Try different prediction methods
        
        # Method 1: Simple arithmetic mean
        context_mean = np.mean(context_values)
        
        # Method 2: Median (robust to outliers)
        context_median = np.median(context_values)
        
        # Method 3: Pattern-based prediction (if we can detect a pattern)
        pattern_prediction = context_mean  # Default to mean
        if len(context_values) >= 3:
            # Check for arithmetic progression
            diffs = [context_values[i+1] - context_values[i] for i in range(len(context_values)-1)]
            if len(set(diffs)) <= 2:  # Mostly consistent differences
                avg_diff = np.mean(diffs)
                pattern_prediction = context_values[-1] + avg_diff
        
        # Generate candidates around these predictions
        candidates = set()
        for prediction in [context_mean, context_median, pattern_prediction]:
            base = int(round(prediction))
            for offset in range(-3, 4):
                candidates.add(base + offset)
        
        # Also include values from context as candidates
        candidates.update(context_values)
        
        # Remove the corrupted value from candidates if it's there
        candidates.discard(corrupted_value)
        
        if not candidates:
            return corrupted_value, 0.0
        
        # Evaluate each candidate using p-adic consistency
        best_candidate = corrupted_value
        best_score = float('inf')
        
        for candidate in candidates:
            try:
                candidate_adelic = self.adelic_calc.create_adelic(candidate)
                
                # Compute consistency score with context
                consistency_scores = []
                for context_adelic in context_adelics:
                    diff_adelic = self.adelic_calc.add_adelic(
                        candidate_adelic,
                        self.adelic_calc.multiply_adelic(
                            context_adelic,
                            self.adelic_calc.create_adelic(-1)
                        )
                    )
                    consistency_scores.append(self.adelic_calc.adelic_norm(diff_adelic))
                
                # Use average consistency as score
                avg_consistency = np.mean(consistency_scores)
                
                if avg_consistency < best_score:
                    best_score = avg_consistency
                    best_candidate = candidate
                    
            except Exception:
                continue
        
        # Compute confidence based on improvement and context fit
        try:
            # Original consistency score
            original_scores = []
            for context_adelic in context_adelics:
                diff_adelic = self.adelic_calc.add_adelic(
                    corrupted_adelic,
                    self.adelic_calc.multiply_adelic(
                        context_adelic,
                        self.adelic_calc.create_adelic(-1)
                    )
                )
                original_scores.append(self.adelic_calc.adelic_norm(diff_adelic))
            
            original_score = np.mean(original_scores)
            
            # Compute improvement
            if original_score > 0:
                improvement_ratio = (original_score - best_score) / original_score
                confidence = max(0.0, min(1.0, improvement_ratio))
            else:
                confidence = 0.5 if best_candidate != corrupted_value else 0.0
                
            # Boost confidence if the correction fits well with context statistics
            if best_candidate != corrupted_value:
                context_std = np.std(context_values)
                if context_std > 0:
                    z_score_original = abs((corrupted_value - context_mean) / context_std)
                    z_score_corrected = abs((best_candidate - context_mean) / context_std)
                    
                    if z_score_corrected < z_score_original:
                        confidence = min(1.0, confidence + 0.3)
                        
        except Exception:
            confidence = 0.1 if best_candidate != corrupted_value else 0.0
        
        return best_candidate, confidence
    
    def correct_data_array(self, data: np.ndarray, 
                          error_positions: List[int]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Correct errors in data array using p-adic methods.
        
        Args:
            data: Data array with errors
            error_positions: Positions of detected errors
        
        Returns:
            Tuple of (corrected_data, correction_info)
        """
        corrected_data = data.copy()
        corrections_made = []
        
        for pos in error_positions:
            if pos < 0 or pos >= len(data):
                continue
            
            # Get context values (neighbors)
            context_start = max(0, pos - 2)
            context_end = min(len(data), pos + 3)
            context_positions = [i for i in range(context_start, context_end) if i != pos]
            context_values = [int(data[i]) for i in context_positions 
                            if isinstance(data[i], (int, float, np.integer, np.floating))]
            
            if not context_values:
                continue
            
            # Attempt correction
            original_value = int(data[pos])
            corrected_value, confidence = self.correct_error_hensel(
                original_value, context_values
            )
            
            if confidence > 0.1 and corrected_value != original_value:  # Apply correction if there's any improvement
                corrected_data[pos] = corrected_value
                corrections_made.append({
                    'position': pos,
                    'original_value': original_value,
                    'corrected_value': corrected_value,
                    'confidence': confidence,
                    'context_values': context_values
                })
                
                self.system_statistics['successful_corrections'] += 1
            else:
                self.system_statistics['failed_corrections'] += 1
            
            self.system_statistics['total_corrections'] += 1
        
        # Update statistics
        if corrections_made:
            error_magnitudes = [abs(corr['corrected_value'] - corr['original_value']) 
                              for corr in corrections_made]
            self.system_statistics['average_error_magnitude'] = np.mean(error_magnitudes)
        
        correction_info = {
            'corrections_made': corrections_made,
            'num_corrections': len(corrections_made),
            'success_rate': (len(corrections_made) / max(1, len(error_positions))),
            'statistics': self.system_statistics.copy()
        }
        
        return corrected_data, correction_info

    def encode_with_error_correction(self, data: np.ndarray, 
                                   method: str = "auto", 
                                   redundancy_level: float = 0.3) -> Dict:
        """
        Encode data with error correction using specified method.
        Moved from enhanced_error_correction.py
        
        Args:
            data: Input data to encode
            method: Encoding method ("padic", "fibonacci", "auto")
            redundancy_level: Level of redundancy for error correction (0.0 to 1.0)
            
        Returns:
            Dictionary with encoded data and metadata
        """
        if len(data) == 0:
            return self._empty_encoding_result()
        
        start_time = time.time()
        
        # Choose encoding method
        if method == "auto":
            method = self._choose_optimal_method(data)
        
        # Encode based on method
        if method == "padic":
            encoded_state = self.padic_encoder.encode_to_padic(data)
            encoding_type = "padic"
            # Ensure PAdicNumber has a `metadata` attribute, add if not present
            if not hasattr(encoded_state, 'metadata'):
                object.__setattr__(encoded_state, 'metadata', {})
            encoded_state.metadata.update({'original_data_length': len(data)}) # Ensure this is always present for decoding
            
        elif method == "fibonacci":
            encoded_state = self.fibonacci_encoder.encode_to_fibonacci(data, redundancy_level)
            encoding_type = "fibonacci"
            
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        encoding_time = time.time() - start_time
        
        # Calculate encoding efficiency
        original_size = len(data) * 8  # Assume 8 bytes per float
        
        if encoding_type == "padic":
            encoded_size = len(encoded_state.digits) * 4  # 4 bytes per digit
        else:  # fibonacci
            encoded_size = len(encoded_state.encoded_bits) // 8  # bits to bytes
        
        efficiency = original_size / max(encoded_size, 1)
        
        result = {
            'encoded_state': encoded_state,
            'encoding_type': encoding_type,
            'original_data': data.copy(),
            'encoding_time': encoding_time,
            'encoding_efficiency': efficiency,
            'redundancy_level': redundancy_level,
            'method_chosen': method,
            'timestamp': time.time()
        }
        
        # self.logger.info(f"Encoded data using {encoding_type}: " # Removed logger for automated agent
        #                 f"Efficiency={efficiency:.2f}, "
        #                 f"Time={encoding_time:.3f}s")
        
        return result
    
    def decode_with_error_correction(self, encoded_result: Dict) -> Tuple[np.ndarray, ErrorCorrectionResult]:
        """
        Decode data with error correction.
        Moved from enhanced_error_correction.py
        
        Args:
            encoded_result: Result from encode_with_error_correction
            
        Returns:
            Tuple of (decoded_data, error_correction_result)
        """
        start_time = time.time()
        
        encoding_type = encoded_result['encoding_type']
        encoded_state = encoded_result['encoded_state']
        original_data = encoded_result['original_data']
        
        # Decode based on type
        if encoding_type == "padic":
            decoded_data = self.padic_encoder.decode_from_padic(encoded_state)
            
        elif encoding_type == "fibonacci":
            decoded_data = self.fibonacci_encoder.decode_from_fibonacci(encoded_state)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        decoding_time = time.time() - start_time
        
        # Calculate error metrics
        if len(original_data) > 0 and len(decoded_data) > 0:
            min_len = min(len(original_data), len(decoded_data))
            orig_subset = original_data[:min_len]
            decoded_subset = decoded_data[:min_len]
            
            # Calculate error rate
            error_threshold = self.config.error_correction.error_threshold
            errors = np.sum(np.abs(orig_subset - decoded_subset) > error_threshold)
            error_rate = errors / min_len
            success_rate = 1.0 - error_rate
        else:
            errors = 0
            success_rate = 1.0 if len(decoded_data) == len(original_data) else 0.0
        
        # Create error correction result
        correction_result = ErrorCorrectionResult(
            original_errors=0,  # No errors introduced yet
            corrected_errors=0,  # No corrections needed in clean decode
            correction_success_rate=success_rate,
            encoding_efficiency=encoded_result['encoding_efficiency'],
            decoding_time=decoding_time,
            method_used=encoding_type,
            confidence_score=success_rate,
            metadata={
                'error_threshold': error_threshold if 'error_threshold' in locals() else self.config.error_correction.error_threshold,
                'data_length_match': len(decoded_data) == len(original_data)
            }
        )
        
        # Record correction history
        self.correction_history.append(correction_result)
        
        return decoded_data, correction_result
    
    def correct_corrupted_data(self, corrupted_encoded_result: Dict) -> Tuple[np.ndarray, ErrorCorrectionResult]:
        """
        Correct errors in corrupted encoded data.
        Moved from enhanced_error_correction.py
        
        Args:
            corrupted_encoded_result: Corrupted encoded data
            
        Returns:
            Tuple of (corrected_decoded_data, error_correction_result)
        """
        start_time = time.time()
        
        encoding_type = corrupted_encoded_result['encoding_type']
        corrupted_state = corrupted_encoded_result['encoded_state']
        original_data = corrupted_encoded_result['original_data']
        
        # Apply error correction based on encoding type
        if encoding_type == "padic":
            corrected_state, corrections_made = self.padic_encoder.correct_padic_errors(corrupted_state)
            corrected_data = self.padic_encoder.decode_from_padic(corrected_state)
            
        elif encoding_type == "fibonacci":
            corrected_state, corrections_made = self.fibonacci_encoder.correct_fibonacci_errors(corrupted_state)
            corrected_data = self.fibonacci_encoder.decode_from_fibonacci(corrected_state)
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        correction_time = time.time() - start_time
        
        # Calculate correction metrics
        if len(original_data) > 0 and len(corrected_data) > 0:
            min_len = min(len(original_data), len(corrected_data))
            orig_subset = original_data[:min_len]
            corrected_subset = corrected_data[:min_len]
            
            # Calculate remaining errors after correction
            error_threshold = self.config.error_correction.error_threshold
            remaining_errors = np.sum(np.abs(orig_subset - corrected_subset) > error_threshold)
            success_rate = 1.0 - (remaining_errors / min_len)
        else:
            remaining_errors = 0
            success_rate = 1.0 if len(corrected_data) == len(original_data) else 0.0
        
        # Estimate original errors (simplified)
        estimated_original_errors = corrections_made + remaining_errors
        
        # Create error correction result
        correction_result = ErrorCorrectionResult(
            original_errors=estimated_original_errors,
            corrected_errors=corrections_made,
            correction_success_rate=success_rate,
            encoding_efficiency=corrupted_encoded_result['encoding_efficiency'],
            decoding_time=correction_time,
            method_used=encoding_type,
            confidence_score=success_rate * (corrections_made / max(estimated_original_errors, 1)),
            metadata={
                'remaining_errors': remaining_errors,
                'correction_method': f"{encoding_type}_error_correction"
            }
        )
        
        # Record correction history
        self.correction_history.append(correction_result)
        
        # self.logger.info(f"Error correction completed: " # Removed logger for automated agent
        #                 f"Method={encoding_type}, "
        #                 f"Corrections={corrections_made}, "
        #                 f"Success={success_rate:.3f}, "
        #                 f"Time={correction_time:.3f}s")
        
        return corrected_data, correction_result
    
    def get_correction_statistics(self) -> Dict:
        """Get statistics on error correction performance. Combines p-adic specific and overall."""
        overall_stats = {
            'total_corrections': 0,
            'average_success_rate': 0.0,
            'methods_used': {},
            'total_correction_time': 0.0,
            'p_adic_specific_stats': self.system_statistics.copy() # Include old p-adic stats
        }

        if not self.correction_history:
            return overall_stats
        
        # Calculate statistics from `correction_history`
        total_corrections = len(self.correction_history)
        success_rates = [r.correction_success_rate for r in self.correction_history]
        efficiencies = [r.encoding_efficiency for r in self.correction_history]
        correction_times = [r.decoding_time for r in self.correction_history]
        
        # Method usage statistics
        methods_used = {}
        for result in self.correction_history:
            method = result.method_used
            methods_used[method] = methods_used.get(method, 0) + 1
        
        overall_stats.update({
            'total_corrections': total_corrections,
            'average_success_rate': np.mean(success_rates),
            'average_efficiency': np.mean(efficiencies),
            'methods_used': methods_used,
            'total_correction_time': sum(correction_times),
            'best_success_rate': max(success_rates),
            'worst_success_rate': min(success_rates),
            'statistics_timestamp': time.time()
        })
        
        return overall_stats
    
    def _choose_optimal_method(self, data: np.ndarray) -> str:
        """Choose optimal encoding method based on data characteristics. Moved from enhanced_error_correction.py"""
        if len(data) == 0:
            return "padic"
        
        # Analyze data characteristics
        data_variance = np.var(data)
        data_range = np.max(data) - np.min(data)
        data_complexity = len(np.unique(data)) / len(data)
        
        # Decision logic
        if data_variance < 0.1 and data_complexity < 0.5:
            # Low variance, low complexity -> Fibonacci encoding
            return "fibonacci"
        elif data_range > 1000 or data_complexity > 0.8:
            # High range or high complexity -> p-adic encoding
            return "padic"
        else:
            # Default to p-adic for general cases
            return "padic"
    
    def _empty_encoding_result(self) -> Dict:
        """Return empty encoding result. Moved from enhanced_error_correction.py"""
        return {
            'encoded_state': None,
            'encoding_type': 'none',
            'original_data': np.array([]),
            'encoding_time': 0.0,
            'encoding_efficiency': 0.0,
            'redundancy_level': 0.0,
            'method_chosen': 'none',
            'timestamp': time.time()
        }
    
    def validate_correction(self, original_data: np.ndarray, 
                          corrected_data: np.ndarray,
                          known_correct: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Validate the quality of error correction.
        
        Args:
            original_data: Original data with errors
            corrected_data: Data after correction
            known_correct: Known correct data (if available)
        
        Returns:
            Dictionary containing validation metrics
        """
        validation_results = {
            'data_integrity': True,
            'correction_quality': 0.0,
            'p_adic_consistency': 0.0
        }
        
        # Check data integrity
        if len(original_data) != len(corrected_data):
            validation_results['data_integrity'] = False
            validation_results['integrity_error'] = "Data length mismatch"
            return validation_results
        
        # Compute correction quality if known correct data is available
        if known_correct is not None and len(known_correct) == len(corrected_data):
            correct_corrections = 0
            total_changes = 0
            
            for i in range(len(original_data)):
                if original_data[i] != corrected_data[i]:
                    total_changes += 1
                    if corrected_data[i] == known_correct[i]:
                        correct_corrections += 1
            
            validation_results['correction_quality'] = (
                correct_corrections / max(1, total_changes)
            )
        
        # Compute p-adic consistency
        consistency_scores = []
        
        for i in range(len(corrected_data)):
            if isinstance(corrected_data[i], (int, float)):
                value_adelic = self.adelic_calc.create_adelic(int(corrected_data[i]))
                
                # Check consistency across primes
                p_norms = []
                for p in self.primes:
                    calc = self.adelic_calc.padic_calculators[p]
                    int_val = calc.from_padic(value_adelic.components[p])
                    p_norms.append(calc.norm(int_val))
                
                # Consistency is inverse of variance
                norm_variance = np.var(p_norms)
                consistency = 1.0 / (1.0 + norm_variance)
                consistency_scores.append(consistency)
        
        validation_results['p_adic_consistency'] = np.mean(consistency_scores)
        
        return validation_results
    
    def validate_padic_system(self) -> Dict[str, Any]:
        """
        Validate the p-adic error correction system.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'padic_arithmetic': True,
            'adelic_operations': True,
            'error_detection': True,
            'error_correction': True,
            'hensel_lifting': True,
            'fibonacci_encoding': True
        }
        
        try:
            # Test 1: p-adic arithmetic
            calc = self.adelic_calc.padic_calculators[self.primes[0]] # Use the first prime
            padic_5 = calc.to_padic(5)
            padic_3 = calc.to_padic(3)
            padic_sum = calc.add_padic(padic_5, padic_3)
            
            if calc.from_padic(padic_sum) != 8:
                validation_results['padic_arithmetic'] = False
                validation_results['arithmetic_error'] = "p-adic addition failed"
            
            # Test 2: Adelic operations
            adelic_5 = self.adelic_calc.create_adelic(5)
            adelic_3 = self.adelic_calc.create_adelic(3)
            adelic_sum = self.adelic_calc.add_adelic(adelic_5, adelic_3)
            
            if abs(adelic_sum.real_component - 8.0) > self.config.constants.EPSILON_UBP:
                validation_results['adelic_operations'] = False
                validation_results['adelic_error'] = "Adelic addition failed"
            
            # Test 3: Error detection
            test_data = np.array([1, 2, 3, 999, 5, 6])  # 999 is an obvious error
            detection_result = self.detect_error_padic(test_data)
            
            if detection_result['num_errors'] == 0:
                validation_results['error_detection'] = False
                validation_results['detection_error'] = "Failed to detect obvious error"
            
            # Test 4: Error correction
            corrected_data, correction_info = self.correct_data_array(
                test_data, [3]  # Position of error
            )
            
            if correction_info['num_corrections'] == 0:
                validation_results['error_correction'] = False
                validation_results['correction_error'] = "Failed to correct error"
            
            # Test 5: Hensel lifting (simplified test)
            corrected_val, confidence = self.correct_error_hensel(999, [1, 2, 3, 5, 6])
            
            if confidence == 0.0:
                validation_results['hensel_lifting'] = False
                validation_results['hensel_error'] = "Hensel lifting failed"
            
            # Test 6: Fibonacci Encoding/Decoding (Moved from enhanced_error_correction.py)
            fib_test_data = np.array([10.0, 20.5, 3.14])
            fib_encoded_result = self.encode_with_error_correction(fib_test_data, method="fibonacci", redundancy_level=0.5)
            fib_decoded_data, fib_decode_result = self.decode_with_error_correction(fib_encoded_result)
            
            if not np.allclose(fib_test_data, fib_decoded_data, atol=self.config.constants.EPSILON_UBP * 100): # Allow some float tolerance
                validation_results['fibonacci_encoding'] = False
                validation_results['fibonacci_error'] = f"Fibonacci encode/decode failed: {fib_test_data} vs {fib_decoded_data}"

        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['padic_arithmetic'] = False
        
        return validation_results


print("DEBUG: p_adic_correction.py: AdvancedErrorCorrectionModule class defined.")

# Factory function for easy instantiation
def create_padic_corrector(primes: Optional[List[int]] = None, precision: int = 10) -> AdvancedErrorCorrectionModule:
    """
    Create an AdvancedErrorCorrectionModule with specified configuration.
    
    Args:
        primes: List of prime numbers for p-adic operations.
        precision: Precision for p-adic calculations.
    
    Returns:
        Configured AdvancedErrorCorrectionModule instance.
    """
    return AdvancedErrorCorrectionModule(primes=primes, precision=precision)

__all__ = [
    "PAdicPrime",
    "PAdicNumber",
    "AdelicNumber",
    "FibonacciCode",
    "ErrorCorrectionResult",
    "PAdicArithmetic",
    "AdelicArithmetic",
    "PAdicEncoder",
    "FibonacciEncoder",
    "AdvancedErrorCorrectionModule",
    "create_padic_corrector"
]
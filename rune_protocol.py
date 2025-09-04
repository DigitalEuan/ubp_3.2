"""
UBP Framework v3.0 - Rune Protocol
Author: Euan Craig, New Zealand
Date: 13 August 2025

Rune Protocol provides Glyph operations with self-reference capability for the UBP system.
This module implements the advanced symbolic computation and recursive feedback mechanisms
that enable the UBP to achieve higher-order coherence and self-organization.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from enum import Enum
from abc import ABC, abstractmethod

# Import configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from ubp_config import get_config

class GlyphType(Enum):
    """Types of Glyphs in the Rune Protocol."""
    QUANTIFY = "quantify"
    CORRELATE = "correlate"
    SELF_REFERENCE = "self_reference"
    TRANSFORM = "transform"
    RESONANCE = "resonance"
    COHERENCE = "coherence"
    FEEDBACK = "feedback"
    EMERGENCE = "emergence"

@dataclass
class GlyphState:
    """Represents the state of a Glyph."""
    glyph_id: str
    glyph_type: GlyphType
    activation_level: float
    coherence_pressure: float
    self_reference_depth: int
    resonance_frequency: float
    state_vector: np.ndarray
    metadata: Dict = field(default_factory=dict)

@dataclass
class RuneOperationResult:
    """Result from a Rune Protocol operation."""
    operation_type: str
    input_glyphs: List[str]
    output_glyph: Optional[str]
    coherence_change: float
    self_reference_loops: int
    emergence_detected: bool
    operation_time: float
    nrci_score: float
    metadata: Dict = field(default_factory=dict)

@dataclass
class CoherencePressureState:
    """State of coherence pressure in the system."""
    current_pressure: float
    target_pressure: float
    pressure_gradient: float
    stability_index: float
    mitigation_active: bool
    pressure_history: List[float] = field(default_factory=list)

class GlyphOperator(ABC):
    """Abstract base class for Glyph operators."""
    
    @abstractmethod
    def operate(self, glyph_state: GlyphState, *args, **kwargs) -> GlyphState:
        """Perform the glyph operation."""
        pass
    
    @abstractmethod
    def get_operation_type(self) -> str:
        """Get the operation type name."""
        pass

class QuantifyOperator(GlyphOperator):
    """
    Quantify Glyph Operator: Q(G, state) = Σ G_i(state)
    
    Quantifies the state of a Glyph by summing its components.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def operate(self, glyph_state: GlyphState, target_state: Optional[np.ndarray] = None) -> GlyphState:
        """
        Quantify operation on a Glyph state.
        
        Args:
            glyph_state: Input Glyph state
            target_state: Optional target state for quantification
            
        Returns:
            Updated GlyphState
        """
        if target_state is None:
            target_state = glyph_state.state_vector
        
        # Quantification: sum of state components weighted by activation
        quantified_value = np.sum(glyph_state.state_vector * glyph_state.activation_level)
        
        # Create new state vector with quantified value
        new_state_vector = np.array([quantified_value])
        
        # Update coherence based on quantification quality
        state_coherence = 1.0 - np.var(glyph_state.state_vector) / (np.mean(glyph_state.state_vector)**2 + 1e-10)
        new_coherence_pressure = glyph_state.coherence_pressure * (1.0 + state_coherence * 0.1)
        
        # Create updated state
        updated_state = GlyphState(
            glyph_id=glyph_state.glyph_id,
            glyph_type=glyph_state.glyph_type,
            activation_level=min(1.0, glyph_state.activation_level * 1.1),
            coherence_pressure=new_coherence_pressure,
            self_reference_depth=glyph_state.self_reference_depth,
            resonance_frequency=glyph_state.resonance_frequency,
            state_vector=new_state_vector,
            metadata={
                **glyph_state.metadata,
                'quantified_value': quantified_value,
                'quantification_time': time.time()
            }
        )
        
        return updated_state
    
    def get_operation_type(self) -> str:
        return "quantify"

class CorrelateOperator(GlyphOperator):
    """
    Correlate Glyph Operator: C(G, R_i, R_j) = P(R_i) * P(R_j) / P(R_i ∩ R_j)
    
    Correlates Glyph states across different realms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def operate(self, glyph_state: GlyphState, other_glyph: GlyphState, 
               realm_i: str = "quantum", realm_j: str = "electromagnetic") -> GlyphState:
        """
        Correlate operation between two Glyph states.
        
        Args:
            glyph_state: First Glyph state
            other_glyph: Second Glyph state
            realm_i: First realm
            realm_j: Second realm
            
        Returns:
            Updated GlyphState with correlation information
        """
        # Calculate state probabilities
        state1_norm = np.linalg.norm(glyph_state.state_vector)
        state2_norm = np.linalg.norm(other_glyph.state_vector)
        
        if state1_norm == 0 or state2_norm == 0:
            correlation = 0.0
        else:
            # Normalize states
            norm_state1 = glyph_state.state_vector / state1_norm
            norm_state2 = other_glyph.state_vector / state2_norm
            
            # Ensure same length for correlation
            min_len = min(len(norm_state1), len(norm_state2))
            if min_len == 0:
                correlation = 0.0
            else:
                s1 = norm_state1[:min_len]
                s2 = norm_state2[:min_len]
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(s1, s2)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
        
        # Calculate realm intersection probability (simplified)
        realm_intersection = self._calculate_realm_intersection(realm_i, realm_j)
        
        # Correlation formula: P(R_i) * P(R_j) / P(R_i ∩ R_j)
        p_ri = glyph_state.activation_level
        p_rj = other_glyph.activation_level
        p_intersection = realm_intersection
        
        if p_intersection > 1e-10:
            correlation_value = (p_ri * p_rj) / p_intersection
        else:
            correlation_value = 0.0
        
        # Create correlated state vector
        correlated_state = np.array([correlation, correlation_value])
        
        # Update coherence pressure based on correlation strength
        coherence_boost = abs(correlation) * 0.2
        new_coherence_pressure = glyph_state.coherence_pressure * (1.0 + coherence_boost)
        
        # Create updated state
        updated_state = GlyphState(
            glyph_id=glyph_state.glyph_id,
            glyph_type=glyph_state.glyph_type,
            activation_level=min(1.0, glyph_state.activation_level + abs(correlation) * 0.1),
            coherence_pressure=new_coherence_pressure,
            self_reference_depth=glyph_state.self_reference_depth,
            resonance_frequency=glyph_state.resonance_frequency,
            state_vector=correlated_state,
            metadata={
                **glyph_state.metadata,
                'correlation_coefficient': correlation,
                'correlation_value': correlation_value,
                'correlated_with': other_glyph.glyph_id,
                'realm_i': realm_i,
                'realm_j': realm_j,
                'correlation_time': time.time()
            }
        )
        
        return updated_state
    
    def _calculate_realm_intersection(self, realm_i: str, realm_j: str) -> float:
        """Calculate intersection probability between two realms."""
        # Simplified realm intersection calculation
        # In practice, this would involve complex physics
        
        realm_overlaps = {
            ('quantum', 'electromagnetic'): 0.8,
            ('electromagnetic', 'optical'): 0.9,
            ('quantum', 'nuclear'): 0.7,
            ('gravitational', 'cosmological'): 0.6,
            ('biological', 'quantum'): 0.5,
            ('nuclear', 'optical'): 0.3,
        }
        
        # Check both directions
        overlap = realm_overlaps.get((realm_i, realm_j), 
                                   realm_overlaps.get((realm_j, realm_i), 0.1))
        
        return overlap
    
    def get_operation_type(self) -> str:
        return "correlate"

class SelfReferenceOperator(GlyphOperator):
    """
    Self-Reference Glyph Operator with recursive feedback.
    
    Implements recursive feedback mechanisms with coherence pressure mitigation.
    """
    
    def __init__(self, max_depth: int = 10):
        self.logger = logging.getLogger(__name__)
        self.max_depth = max_depth
        self.coherence_pressure_threshold = 0.8
    
    def operate(self, glyph_state: GlyphState, feedback_strength: float = 0.1) -> GlyphState:
        """
        Self-reference operation with recursive feedback.
        
        Args:
            glyph_state: Input Glyph state
            feedback_strength: Strength of recursive feedback
            
        Returns:
            Updated GlyphState with self-reference applied
        """
        # Check if we've reached maximum recursion depth
        if glyph_state.self_reference_depth >= self.max_depth:
            self.logger.warning(f"Maximum self-reference depth reached for {glyph_state.glyph_id}")
            return glyph_state
        
        # Check coherence pressure
        if glyph_state.coherence_pressure > self.coherence_pressure_threshold:
            # Apply coherence pressure mitigation
            mitigated_state = self._apply_coherence_pressure_mitigation(glyph_state)
            return mitigated_state
        
        # Apply self-reference transformation
        self_ref_state = self._apply_self_reference_transform(glyph_state, feedback_strength)
        
        return self_ref_state
    
    def _apply_self_reference_transform(self, glyph_state: GlyphState, 
                                      feedback_strength: float) -> GlyphState:
        """Apply self-reference transformation to Glyph state."""
        # Self-reference: state becomes a function of itself
        current_state = glyph_state.state_vector
        
        # Recursive feedback: new_state = f(current_state, previous_states)
        if len(current_state) > 0:
            # Simple self-reference: weighted sum of current state with itself
            self_feedback = np.convolve(current_state, current_state[::-1], mode='same')
            
            # Normalize to prevent explosion
            if np.linalg.norm(self_feedback) > 0:
                self_feedback = self_feedback / np.linalg.norm(self_feedback)
            
            # Combine with original state
            new_state = (1.0 - feedback_strength) * current_state + feedback_strength * self_feedback
        else:
            new_state = current_state
        
        # Update coherence pressure (self-reference increases pressure)
        pressure_increase = feedback_strength * 0.1
        new_coherence_pressure = glyph_state.coherence_pressure + pressure_increase
        
        # Increase self-reference depth
        new_depth = glyph_state.self_reference_depth + 1
        
        # Update resonance frequency based on self-reference
        frequency_modulation = 1.0 + feedback_strength * np.sin(2 * np.pi * new_depth / 10.0)
        new_frequency = glyph_state.resonance_frequency * frequency_modulation
        
        # Create updated state
        updated_state = GlyphState(
            glyph_id=glyph_state.glyph_id,
            glyph_type=glyph_state.glyph_type,
            activation_level=min(1.0, glyph_state.activation_level * (1.0 + feedback_strength * 0.05)),
            coherence_pressure=new_coherence_pressure,
            self_reference_depth=new_depth,
            resonance_frequency=new_frequency,
            state_vector=new_state,
            metadata={
                **glyph_state.metadata,
                'self_reference_applied': True,
                'feedback_strength': feedback_strength,
                'self_reference_time': time.time()
            }
        )
        
        return updated_state
    
    def _apply_coherence_pressure_mitigation(self, glyph_state: GlyphState) -> GlyphState:
        """Apply coherence pressure mitigation to prevent system instability."""
        self.logger.info(f"Applying coherence pressure mitigation to {glyph_state.glyph_id}")
        
        # Reduce coherence pressure through state normalization
        current_state = glyph_state.state_vector
        
        if len(current_state) > 0 and np.linalg.norm(current_state) > 0:
            # Normalize state to unit length
            normalized_state = current_state / np.linalg.norm(current_state)
            
            # Apply smoothing to reduce high-frequency components
            if len(normalized_state) > 2:
                smoothed_state = np.convolve(normalized_state, [0.25, 0.5, 0.25], mode='same')
            else:
                smoothed_state = normalized_state
        else:
            smoothed_state = current_state
        
        # Reduce coherence pressure
        mitigated_pressure = glyph_state.coherence_pressure * 0.7
        
        # Reset self-reference depth if pressure was too high
        reset_depth = max(0, glyph_state.self_reference_depth - 2)
        
        # Create mitigated state
        mitigated_state = GlyphState(
            glyph_id=glyph_state.glyph_id,
            glyph_type=glyph_state.glyph_type,
            activation_level=glyph_state.activation_level * 0.9,
            coherence_pressure=mitigated_pressure,
            self_reference_depth=reset_depth,
            resonance_frequency=glyph_state.resonance_frequency,
            state_vector=smoothed_state,
            metadata={
                **glyph_state.metadata,
                'coherence_pressure_mitigated': True,
                'mitigation_time': time.time(),
                'original_pressure': glyph_state.coherence_pressure
            }
        )
        
        return mitigated_state
    
    def get_operation_type(self) -> str:
        return "self_reference"

class EmergenceDetector:
    """
    Detects emergent properties in Glyph interactions.
    
    Monitors for spontaneous organization and higher-order patterns
    that emerge from Glyph operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Emergence detection parameters
        self.complexity_threshold = 0.7
        self.coherence_threshold = 0.9
        self.pattern_memory = []
        self.max_memory_size = 100
    
    def detect_emergence(self, glyph_states: List[GlyphState]) -> Dict:
        """
        Detect emergent properties in a collection of Glyph states.
        
        Args:
            glyph_states: List of Glyph states to analyze
            
        Returns:
            Dictionary with emergence analysis results
        """
        if not glyph_states:
            return self._empty_emergence_result()
        
        # Analyze complexity
        complexity_score = self._calculate_system_complexity(glyph_states)
        
        # Analyze coherence
        coherence_score = self._calculate_system_coherence(glyph_states)
        
        # Detect patterns
        patterns = self._detect_patterns(glyph_states)
        
        # Check for emergence criteria
        emergence_detected = (
            complexity_score > self.complexity_threshold and
            coherence_score > self.coherence_threshold and
            len(patterns) > 0
        )
        
        # Analyze emergence type
        emergence_type = self._classify_emergence_type(patterns, complexity_score, coherence_score)
        
        # Calculate emergence strength
        emergence_strength = self._calculate_emergence_strength(
            complexity_score, coherence_score, patterns
        )
        
        # Update pattern memory
        self._update_pattern_memory(patterns)
        
        result = {
            'emergence_detected': emergence_detected,
            'emergence_type': emergence_type,
            'emergence_strength': emergence_strength,
            'complexity_score': complexity_score,
            'coherence_score': coherence_score,
            'detected_patterns': patterns,
            'glyph_count': len(glyph_states),
            'analysis_time': time.time()
        }
        
        if emergence_detected:
            self.logger.info(f"Emergence detected: Type={emergence_type}, "
                           f"Strength={emergence_strength:.3f}, "
                           f"Patterns={len(patterns)}")
        
        return result
    
    def _calculate_system_complexity(self, glyph_states: List[GlyphState]) -> float:
        """Calculate overall system complexity."""
        if not glyph_states:
            return 0.0
        
        # Complexity based on state diversity and interactions
        state_vectors = [g.state_vector for g in glyph_states if len(g.state_vector) > 0]
        
        if not state_vectors:
            return 0.0
        
        # Calculate entropy of state distributions
        all_values = np.concatenate(state_vectors)
        if len(all_values) == 0:
            return 0.0
        
        # Discretize values for entropy calculation
        hist, _ = np.histogram(all_values, bins=20)
        hist = hist + 1e-10  # Avoid log(0)
        probabilities = hist / np.sum(hist)
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        # Normalize entropy to [0, 1]
        max_entropy = np.log2(len(probabilities))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Factor in interaction complexity
        interaction_complexity = self._calculate_interaction_complexity(glyph_states)
        
        # Combined complexity
        total_complexity = (normalized_entropy + interaction_complexity) / 2.0
        
        return min(1.0, total_complexity)
    
    def _calculate_system_coherence(self, glyph_states: List[GlyphState]) -> float:
        """Calculate overall system coherence."""
        if not glyph_states:
            return 0.0
        
        # Coherence based on synchronization of Glyph states
        coherence_pressures = [g.coherence_pressure for g in glyph_states]
        activation_levels = [g.activation_level for g in glyph_states]
        
        # Coherence from pressure synchronization
        pressure_coherence = 1.0 - np.var(coherence_pressures) / (np.mean(coherence_pressures)**2 + 1e-10)
        
        # Coherence from activation synchronization
        activation_coherence = 1.0 - np.var(activation_levels) / (np.mean(activation_levels)**2 + 1e-10)
        
        # Combined coherence
        total_coherence = (pressure_coherence + activation_coherence) / 2.0
        
        return min(1.0, max(0.0, total_coherence))
    
    def _calculate_interaction_complexity(self, glyph_states: List[GlyphState]) -> float:
        """Calculate complexity of Glyph interactions."""
        if len(glyph_states) < 2:
            return 0.0
        
        # Complexity based on correlation between Glyph states
        correlations = []
        
        for i, glyph1 in enumerate(glyph_states):
            for j, glyph2 in enumerate(glyph_states):
                if i < j and len(glyph1.state_vector) > 0 and len(glyph2.state_vector) > 0:
                    # Calculate correlation between state vectors
                    min_len = min(len(glyph1.state_vector), len(glyph2.state_vector))
                    if min_len > 1:
                        s1 = glyph1.state_vector[:min_len]
                        s2 = glyph2.state_vector[:min_len]
                        
                        corr = np.corrcoef(s1, s2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
        
        if not correlations:
            return 0.0
        
        # Interaction complexity based on correlation diversity
        correlation_entropy = -np.sum([c * np.log2(c + 1e-10) for c in correlations])
        
        # Normalize
        max_entropy = len(correlations) * np.log2(len(correlations)) if len(correlations) > 1 else 1.0
        normalized_complexity = correlation_entropy / max_entropy
        
        return min(1.0, normalized_complexity)
    
    def _detect_patterns(self, glyph_states: List[GlyphState]) -> List[Dict]:
        """Detect patterns in Glyph state collection."""
        patterns = []
        
        if len(glyph_states) < 2:
            return patterns
        
        # Pattern 1: Synchronization patterns
        sync_pattern = self._detect_synchronization_pattern(glyph_states)
        if sync_pattern:
            patterns.append(sync_pattern)
        
        # Pattern 2: Resonance patterns
        resonance_pattern = self._detect_resonance_pattern(glyph_states)
        if resonance_pattern:
            patterns.append(resonance_pattern)
        
        # Pattern 3: Hierarchical patterns
        hierarchy_pattern = self._detect_hierarchy_pattern(glyph_states)
        if hierarchy_pattern:
            patterns.append(hierarchy_pattern)
        
        return patterns
    
    def _detect_synchronization_pattern(self, glyph_states: List[GlyphState]) -> Optional[Dict]:
        """Detect synchronization patterns in Glyph states."""
        activation_levels = [g.activation_level for g in glyph_states]
        
        # Check for synchronization (low variance in activation levels)
        activation_var = np.var(activation_levels)
        activation_mean = np.mean(activation_levels)
        
        if activation_mean > 0 and activation_var / (activation_mean**2) < 0.1:
            return {
                'type': 'synchronization',
                'strength': 1.0 - activation_var / (activation_mean**2),
                'participants': [g.glyph_id for g in glyph_states],
                'sync_level': activation_mean
            }
        
        return None
    
    def _detect_resonance_pattern(self, glyph_states: List[GlyphState]) -> Optional[Dict]:
        """Detect resonance patterns in Glyph frequencies."""
        frequencies = [g.resonance_frequency for g in glyph_states]
        
        # Look for harmonic relationships
        for i, freq1 in enumerate(frequencies):
            for j, freq2 in enumerate(frequencies):
                if i < j and freq1 > 0 and freq2 > 0:
                    ratio = freq2 / freq1
                    
                    # Check if ratio is close to a simple harmonic (2, 3, 1.5, etc.)
                    simple_ratios = [0.5, 1.5, 2.0, 3.0, 4.0]
                    for simple_ratio in simple_ratios:
                        if abs(ratio - simple_ratio) < 0.1:
                            return {
                                'type': 'resonance',
                                'strength': 1.0 - abs(ratio - simple_ratio) / 0.1,
                                'participants': [glyph_states[i].glyph_id, glyph_states[j].glyph_id],
                                'frequency_ratio': ratio,
                                'harmonic_ratio': simple_ratio
                            }
        
        return None
    
    def _detect_hierarchy_pattern(self, glyph_states: List[GlyphState]) -> Optional[Dict]:
        """Detect hierarchical patterns in Glyph organization."""
        # Sort by self-reference depth
        sorted_glyphs = sorted(glyph_states, key=lambda g: g.self_reference_depth)
        
        # Check for clear hierarchy (different depth levels)
        depths = [g.self_reference_depth for g in sorted_glyphs]
        unique_depths = len(set(depths))
        
        if unique_depths > 1 and unique_depths < len(glyph_states):
            return {
                'type': 'hierarchy',
                'strength': unique_depths / len(glyph_states),
                'participants': [g.glyph_id for g in sorted_glyphs],
                'depth_levels': unique_depths,
                'max_depth': max(depths)
            }
        
        return None
    
    def _classify_emergence_type(self, patterns: List[Dict], 
                                complexity: float, coherence: float) -> str:
        """Classify the type of emergence detected."""
        if not patterns:
            return "none"
        
        pattern_types = [p['type'] for p in patterns]
        
        # Classification based on dominant patterns and metrics
        if 'hierarchy' in pattern_types and complexity > 0.8:
            return "hierarchical_emergence"
        elif 'synchronization' in pattern_types and coherence > 0.9:
            return "coherent_emergence"
        elif 'resonance' in pattern_types:
            return "resonant_emergence"
        elif len(patterns) > 1:
            return "complex_emergence"
        else:
            return "simple_emergence"
    
    def _calculate_emergence_strength(self, complexity: float, 
                                    coherence: float, patterns: List[Dict]) -> float:
        """Calculate overall emergence strength."""
        if not patterns:
            return 0.0
        
        # Base strength from complexity and coherence
        base_strength = (complexity + coherence) / 2.0
        
        # Pattern contribution
        pattern_strength = np.mean([p.get('strength', 0.5) for p in patterns])
        
        # Number of patterns bonus
        pattern_bonus = min(0.2, len(patterns) * 0.05)
        
        # Combined strength
        total_strength = base_strength * pattern_strength + pattern_bonus
        
        return min(1.0, total_strength)
    
    def _update_pattern_memory(self, patterns: List[Dict]):
        """Update pattern memory for learning."""
        for pattern in patterns:
            self.pattern_memory.append({
                'pattern': pattern,
                'timestamp': time.time()
            })
        
        # Limit memory size
        if len(self.pattern_memory) > self.max_memory_size:
            self.pattern_memory = self.pattern_memory[-self.max_memory_size:]
    
    def _empty_emergence_result(self) -> Dict:
        """Return empty emergence detection result."""
        return {
            'emergence_detected': False,
            'emergence_type': 'none',
            'emergence_strength': 0.0,
            'complexity_score': 0.0,
            'coherence_score': 0.0,
            'detected_patterns': [],
            'glyph_count': 0,
            'analysis_time': time.time()
        }

class RuneProtocol:
    """
    Main Rune Protocol engine for UBP Framework v3.0.
    
    Provides Glyph operations with self-reference capability and emergence detection.
    Implements the symbolic computation layer that enables higher-order UBP operations.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        
        # Initialize operators
        self.operators = {
            'quantify': QuantifyOperator(),
            'correlate': CorrelateOperator(),
            'self_reference': SelfReferenceOperator()
        }
        
        # Initialize emergence detector
        self.emergence_detector = EmergenceDetector()
        
        # Glyph registry
        self.glyphs = {}
        self.operation_history = []
        
        # Coherence pressure monitoring
        self.coherence_pressure_state = CoherencePressureState(
            current_pressure=0.0,
            target_pressure=0.8,
            pressure_gradient=0.0,
            stability_index=1.0,
            mitigation_active=False
        )
    
    def create_glyph(self, glyph_id: str, glyph_type: GlyphType, 
                    initial_state: Optional[np.ndarray] = None) -> GlyphState:
        """
        Create a new Glyph with specified type and initial state.
        
        Args:
            glyph_id: Unique identifier for the Glyph
            glyph_type: Type of Glyph to create
            initial_state: Initial state vector (random if None)
            
        Returns:
            Created GlyphState
        """
        if initial_state is None:
            initial_state = np.random.random(10) * 0.1  # Small random initial state
        
        # Create Glyph state
        glyph_state = GlyphState(
            glyph_id=glyph_id,
            glyph_type=glyph_type,
            activation_level=0.1,
            coherence_pressure=0.0,
            self_reference_depth=0,
            resonance_frequency=1e12,  # Default 1 THz
            state_vector=initial_state,
            metadata={
                'creation_time': time.time(),
                'creator': 'RuneProtocol'
            }
        )
        
        # Register Glyph
        self.glyphs[glyph_id] = glyph_state
        
        self.logger.info(f"Created Glyph: {glyph_id} (type: {glyph_type.value})")
        
        return glyph_state
    
    def execute_operation(self, operation_type: str, glyph_id: str, 
                         **kwargs) -> RuneOperationResult:
        """
        Execute a Rune Protocol operation on a Glyph.
        
        Args:
            operation_type: Type of operation to execute
            glyph_id: Target Glyph ID
            **kwargs: Additional operation parameters
            
        Returns:
            RuneOperationResult with operation results
        """
        if glyph_id not in self.glyphs:
            raise ValueError(f"Glyph {glyph_id} not found")
        
        if operation_type not in self.operators:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        start_time = time.time()
        glyph_state = self.glyphs[glyph_id]
        operator = self.operators[operation_type]
        
        # Record initial state
        initial_coherence = glyph_state.coherence_pressure
        initial_loops = glyph_state.self_reference_depth
        
        # Execute operation
        try:
            updated_state = operator.operate(glyph_state, **kwargs)
            self.glyphs[glyph_id] = updated_state
            
            # Calculate changes
            coherence_change = updated_state.coherence_pressure - initial_coherence
            self_reference_loops = updated_state.self_reference_depth - initial_loops
            
            # Check for emergence
            emergence_result = self.emergence_detector.detect_emergence([updated_state])
            emergence_detected = emergence_result['emergence_detected']
            
            # Calculate NRCI
            nrci_score = self._calculate_operation_nrci(glyph_state, updated_state)
            
            # Update coherence pressure state
            self._update_coherence_pressure_state(updated_state.coherence_pressure)
            
            operation_time = time.time() - start_time
            
            # Create result
            result = RuneOperationResult(
                operation_type=operation_type,
                input_glyphs=[glyph_id],
                output_glyph=glyph_id,
                coherence_change=coherence_change,
                self_reference_loops=self_reference_loops,
                emergence_detected=emergence_detected,
                operation_time=operation_time,
                nrci_score=nrci_score,
                metadata={
                    'emergence_result': emergence_result,
                    'initial_state_norm': np.linalg.norm(glyph_state.state_vector),
                    'final_state_norm': np.linalg.norm(updated_state.state_vector)
                }
            )
            
            # Record operation
            self.operation_history.append(result)
            
            self.logger.info(f"Operation {operation_type} on {glyph_id}: "
                           f"NRCI={nrci_score:.6f}, "
                           f"Emergence={emergence_detected}, "
                           f"Time={operation_time:.3f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Operation {operation_type} failed on {glyph_id}: {e}")
            raise
    
    def execute_multi_glyph_operation(self, operation_type: str, 
                                    glyph_ids: List[str], **kwargs) -> RuneOperationResult:
        """
        Execute operation involving multiple Glyphs.
        
        Args:
            operation_type: Type of operation
            glyph_ids: List of Glyph IDs to operate on
            **kwargs: Additional parameters
            
        Returns:
            RuneOperationResult
        """
        if not glyph_ids:
            raise ValueError("No Glyph IDs provided")
        
        # Validate all Glyphs exist
        for glyph_id in glyph_ids:
            if glyph_id not in self.glyphs:
                raise ValueError(f"Glyph {glyph_id} not found")
        
        start_time = time.time()
        
        if operation_type == "correlate" and len(glyph_ids) >= 2:
            # Correlate operation between two Glyphs
            glyph1 = self.glyphs[glyph_ids[0]]
            glyph2 = self.glyphs[glyph_ids[1]]
            
            operator = self.operators['correlate']
            updated_state = operator.operate(glyph1, glyph2, **kwargs)
            
            # Update first Glyph with correlation result
            self.glyphs[glyph_ids[0]] = updated_state
            
            # Check for emergence across all involved Glyphs
            all_states = [self.glyphs[gid] for gid in glyph_ids]
            emergence_result = self.emergence_detector.detect_emergence(all_states)
            
            # Calculate NRCI
            nrci_score = self._calculate_operation_nrci(glyph1, updated_state)
            
            operation_time = time.time() - start_time
            
            result = RuneOperationResult(
                operation_type=operation_type,
                input_glyphs=glyph_ids,
                output_glyph=glyph_ids[0],
                coherence_change=updated_state.coherence_pressure - glyph1.coherence_pressure,
                self_reference_loops=0,
                emergence_detected=emergence_result['emergence_detected'],
                operation_time=operation_time,
                nrci_score=nrci_score,
                metadata={'emergence_result': emergence_result}
            )
            
            self.operation_history.append(result)
            return result
        
        else:
            raise ValueError(f"Multi-Glyph operation {operation_type} not supported")
    
    def get_system_state(self) -> Dict:
        """Get current state of the Rune Protocol system."""
        glyph_states = list(self.glyphs.values())
        
        # System-wide emergence analysis
        emergence_result = self.emergence_detector.detect_emergence(glyph_states)
        
        # Calculate system metrics
        total_coherence_pressure = sum(g.coherence_pressure for g in glyph_states)
        avg_activation = np.mean([g.activation_level for g in glyph_states]) if glyph_states else 0.0
        max_self_ref_depth = max([g.self_reference_depth for g in glyph_states]) if glyph_states else 0
        
        return {
            'glyph_count': len(self.glyphs),
            'total_coherence_pressure': total_coherence_pressure,
            'average_activation': avg_activation,
            'max_self_reference_depth': max_self_ref_depth,
            'coherence_pressure_state': {
                'current': self.coherence_pressure_state.current_pressure,
                'target': self.coherence_pressure_state.target_pressure,
                'stability': self.coherence_pressure_state.stability_index,
                'mitigation_active': self.coherence_pressure_state.mitigation_active
            },
            'emergence_status': emergence_result,
            'operation_count': len(self.operation_history),
            'system_time': time.time()
        }
    
    def _calculate_operation_nrci(self, initial_state: GlyphState, 
                                final_state: GlyphState) -> float:
        """Calculate NRCI for a Rune operation."""
        # NRCI based on coherence preservation and enhancement
        initial_coherence = 1.0 - np.var(initial_state.state_vector) / (np.mean(initial_state.state_vector)**2 + 1e-10)
        final_coherence = 1.0 - np.var(final_state.state_vector) / (np.mean(final_state.state_vector)**2 + 1e-10)
        
        # Information preservation
        if len(initial_state.state_vector) > 0 and len(final_state.state_vector) > 0:
            min_len = min(len(initial_state.state_vector), len(final_state.state_vector))
            if min_len > 1:
                initial_norm = initial_state.state_vector[:min_len]
                final_norm = final_state.state_vector[:min_len]
                
                if np.linalg.norm(initial_norm) > 0 and np.linalg.norm(final_norm) > 0:
                    initial_norm = initial_norm / np.linalg.norm(initial_norm)
                    final_norm = final_norm / np.linalg.norm(final_norm)
                    
                    information_preservation = abs(np.dot(initial_norm, final_norm))
                else:
                    information_preservation = 0.0
            else:
                information_preservation = 0.5
        else:
            information_preservation = 0.0
        
        # Combined NRCI
        nrci = (initial_coherence + final_coherence + information_preservation) / 3.0
        
        return min(1.0, max(0.0, nrci))
    
    def _update_coherence_pressure_state(self, new_pressure: float):
        """Update coherence pressure monitoring state."""
        # Update current pressure
        old_pressure = self.coherence_pressure_state.current_pressure
        self.coherence_pressure_state.current_pressure = new_pressure
        
        # Calculate gradient
        self.coherence_pressure_state.pressure_gradient = new_pressure - old_pressure
        
        # Update history
        self.coherence_pressure_state.pressure_history.append(new_pressure)
        if len(self.coherence_pressure_state.pressure_history) > 100:
            self.coherence_pressure_state.pressure_history = self.coherence_pressure_state.pressure_history[-100:]
        
        # Calculate stability index
        if len(self.coherence_pressure_state.pressure_history) > 10:
            recent_pressures = self.coherence_pressure_state.pressure_history[-10:]
            pressure_variance = np.var(recent_pressures)
            pressure_mean = np.mean(recent_pressures)
            
            if pressure_mean > 0:
                stability = 1.0 - pressure_variance / (pressure_mean**2)
                self.coherence_pressure_state.stability_index = max(0.0, min(1.0, stability))
        
        # Check if mitigation should be active
        self.coherence_pressure_state.mitigation_active = (
            new_pressure > self.coherence_pressure_state.target_pressure
        )


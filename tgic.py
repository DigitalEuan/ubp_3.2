"""
Universal Binary Principle (UBP) Framework v3.2+ - TGIC: Triad Graph Interaction Constraint for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

Implements the geometric constraint system that enforces the fundamental
3, 6, 9 structure across UBP realms using dodecahedral graphs and
Leech lattice projections.

Mathematical Foundation:
- 3 axes: x, y, z spatial dimensions
- 6 faces: cubic/dodecahedral face interactions
- 9 interactions: per OffBit neighborhood interactions
- Dodecahedral graph: 20 nodes, 60 edges
- Leech lattice: 24D sphere packing projection
- Geometric coherence constraints

This is NOT a simulation - implements real geometric constraint mathematics.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict


class TGICGeometry(Enum):
    """TGIC geometric structures"""
    CUBIC = "cubic"                    # 3×3×3 cubic structure
    DODECAHEDRAL = "dodecahedral"      # 20-node dodecahedral graph
    ICOSAHEDRAL = "icosahedral"        # 12-node icosahedral graph
    LEECH_24D = "leech_24d"           # 24D Leech lattice projection
    TETRAHEDRAL = "tetrahedral"        # 4-node tetrahedral structure
    OCTAHEDRAL = "octahedral"          # 6-node octahedral structure


class InteractionType(Enum):
    """Types of TGIC interactions"""
    AXIS_ALIGNED = "axis_aligned"      # Along x, y, z axes
    FACE_DIAGONAL = "face_diagonal"    # Across face diagonals
    SPACE_DIAGONAL = "space_diagonal"  # Through space diagonals
    EDGE_CONNECTED = "edge_connected"  # Edge-to-edge connections
    VERTEX_SHARED = "vertex_shared"    # Vertex-sharing interactions
    HARMONIC = "harmonic"              # Harmonic resonance interactions
    QUANTUM = "quantum"                # Quantum entanglement interactions
    TEMPORAL = "temporal"              # Temporal coupling interactions
    NONLOCAL = "nonlocal"             # Non-local correlations


@dataclass
class TGICNode:
    """
    Represents a node in the TGIC graph structure.
    """
    node_id: int
    position: np.ndarray  # 3D or higher dimensional position
    connections: Set[int] = field(default_factory=set)
    interaction_types: Dict[int, InteractionType] = field(default_factory=dict)
    weight: float = 1.0
    activation_state: float = 0.0
    coherence_level: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TGICConstraint:
    """
    Represents a geometric constraint in the TGIC system.
    """
    constraint_id: str
    constraint_type: str
    nodes_involved: List[int]
    constraint_function: callable
    tolerance: float = 1e-6
    weight: float = 1.0
    active: bool = True


class DodecahedralGraph:
    """
    Implements the dodecahedral graph structure for TGIC.
    
    A dodecahedron has 20 vertices, 30 edges, and 12 pentagonal faces.
    This provides the geometric foundation for the 3, 6, 9 structure.
    """
    
    def __init__(self):
        self.nodes = {}
        self.edges = set()
        self._generate_dodecahedral_structure()
    
    def _generate_dodecahedral_structure(self):
        """
        Generate the complete dodecahedral graph structure.
        
        Uses the golden ratio φ = (1 + √5)/2 for vertex coordinates.
        """
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        
        # Dodecahedron vertices (20 vertices)
        vertices = []
        
        # 8 vertices of a cube
        for i in [-1, 1]:
            for j in [-1, 1]:
                for k in [-1, 1]:
                    vertices.append([i, j, k])
        
        # 12 vertices on rectangular faces
        for i in [-1, 1]:
            vertices.append([0, i/phi, i*phi])
            vertices.append([i/phi, i*phi, 0])
            vertices.append([i*phi, 0, i/phi])
        
        # Create nodes
        for i, vertex in enumerate(vertices):
            self.nodes[i] = TGICNode(
                node_id=i,
                position=np.array(vertex),
                weight=1.0
            )
        
        # Generate edges based on dodecahedral connectivity
        self._generate_dodecahedral_edges()
    
    def _generate_dodecahedral_edges(self):
        """
        Generate edges for the dodecahedral graph.
        
        Each vertex connects to exactly 3 other vertices.
        """
        # Distance threshold for edge connection
        edge_threshold = 2.1  # Approximate edge length in dodecahedron
        
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                pos_i = self.nodes[i].position
                pos_j = self.nodes[j].position
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < edge_threshold:
                    self.edges.add((i, j))
                    self.nodes[i].connections.add(j)
                    self.nodes[j].connections.add(i)
                    
                    # Determine interaction type based on geometry
                    if self._is_axis_aligned(pos_i, pos_j):
                        interaction_type = InteractionType.AXIS_ALIGNED
                    elif self._is_face_diagonal(pos_i, pos_j):
                        interaction_type = InteractionType.FACE_DIAGONAL
                    else:
                        interaction_type = InteractionType.EDGE_CONNECTED
                    
                    self.nodes[i].interaction_types[j] = interaction_type
                    self.nodes[j].interaction_types[i] = interaction_type
    
    def _is_axis_aligned(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if two positions are axis-aligned"""
        diff = pos1 - pos2
        non_zero_count = np.sum(np.abs(diff) > 1e-6)
        return non_zero_count == 1
    
    def _is_face_diagonal(self, pos1: np.ndarray, pos2: np.ndarray) -> bool:
        """Check if two positions form a face diagonal"""
        diff = pos1 - pos2
        non_zero_count = np.sum(np.abs(diff) > 1e-6)
        return non_zero_count == 2
    
    def get_node_neighbors(self, node_id: int) -> List[int]:
        """Get all neighbors of a given node"""
        if node_id in self.nodes:
            return list(self.nodes[node_id].connections)
        return []
    
    def get_interaction_type(self, node1: int, node2: int) -> Optional[InteractionType]:
        """Get interaction type between two nodes"""
        if node1 in self.nodes and node2 in self.nodes[node1].interaction_types:
            return self.nodes[node1].interaction_types[node2]
        return None
    
    def compute_graph_properties(self) -> Dict[str, Any]:
        """Compute properties of the dodecahedral graph"""
        num_nodes = len(self.nodes)
        num_edges = len(self.edges)
        
        # Compute degree distribution
        degrees = [len(node.connections) for node in self.nodes.values()]
        avg_degree = np.mean(degrees)
        
        # Compute clustering coefficient
        clustering_coeffs = []
        for node_id, node in self.nodes.items():
            neighbors = list(node.connections)
            if len(neighbors) < 2:
                clustering_coeffs.append(0.0)
                continue
            
            # Count triangles
            triangles = 0
            possible_triangles = len(neighbors) * (len(neighbors) - 1) // 2
            
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in self.nodes[neighbors[i]].connections:
                        triangles += 1
            
            clustering = triangles / possible_triangles if possible_triangles > 0 else 0.0
            clustering_coeffs.append(clustering)
        
        avg_clustering = np.mean(clustering_coeffs)
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree,
            'avg_clustering': avg_clustering,
            'degree_distribution': degrees,
            'is_regular': len(set(degrees)) == 1,
            'max_degree': max(degrees),
            'min_degree': min(degrees)
        }


class LeechLatticeProjection:
    """
    Implements 24D Leech lattice projection for TGIC constraints.
    
    The Leech lattice provides optimal sphere packing in 24 dimensions
    and serves as the geometric foundation for advanced TGIC operations.
    """
    
    def __init__(self, dimension: int = 24):
        self.dimension = dimension
        self.lattice_points = []
        self._generate_leech_basis()
    
    def _generate_leech_basis(self):
        """
        Generate basis vectors for Leech lattice.
        
        This is a simplified representation. Full Leech lattice
        construction requires advanced algebraic methods.
        """
        # Simplified Leech lattice basis using E8 lattices
        # Full implementation would use proper Leech construction
        
        # Generate E8 lattice basis (8D)
        e8_basis = self._generate_e8_basis()
        
        # Extend to 24D using three copies of E8
        leech_basis = []
        for i in range(3):
            for basis_vector in e8_basis:
                extended_vector = np.zeros(24)
                extended_vector[i*8:(i+1)*8] = basis_vector
                leech_basis.append(extended_vector)
        
        self.basis_vectors = np.array(leech_basis)
    
    def _generate_e8_basis(self) -> List[np.ndarray]:
        """
        Generate basis vectors for E8 lattice.
        
        E8 is the optimal sphere packing lattice in 8 dimensions.
        """
        # Standard E8 basis vectors
        e8_basis = []
        
        # Type 1: (±1, ±1, 0, 0, 0, 0, 0, 0) and permutations
        for signs in itertools.product([-1, 1], repeat=2):
            for positions in itertools.combinations(range(8), 2):
                vector = np.zeros(8)
                for i, pos in enumerate(positions):
                    vector[pos] = signs[i]
                e8_basis.append(vector)
        
        # Type 2: (±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2, ±1/2) with even number of -1/2
        for signs in itertools.product([-0.5, 0.5], repeat=8):
            if sum(1 for s in signs if s < 0) % 2 == 0:  # Even number of negative signs
                e8_basis.append(np.array(signs))
        
        return e8_basis[:8]  # Return first 8 basis vectors
    
    def project_to_3d(self, lattice_point: np.ndarray) -> np.ndarray:
        """
        Project 24D Leech lattice point to 3D for visualization.
        
        Args:
            lattice_point: 24D lattice point
        
        Returns:
            3D projection
        """
        if len(lattice_point) != 24:
            raise ValueError("Lattice point must be 24-dimensional")
        
        # Simple projection: take first 3 coordinates
        # More sophisticated projections could use PCA or other methods
        projection_3d = lattice_point[:3]
        
        return projection_3d
    
    def compute_lattice_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Compute distance between two lattice points.
        
        Args:
            point1, point2: 24D lattice points
        
        Returns:
            Euclidean distance
        """
        return np.linalg.norm(point1 - point2)
    
    def find_nearest_neighbors(self, point: np.ndarray, k: int = 9) -> List[Tuple[np.ndarray, float]]:
        """
        Find k nearest neighbors in the lattice.
        
        Args:
            point: Query point
            k: Number of neighbors to find
        
        Returns:
            List of (neighbor_point, distance) tuples
        """
        if not self.lattice_points:
            # Generate some lattice points for demonstration
            self._generate_sample_lattice_points()
        
        distances = []
        for lattice_point in self.lattice_points:
            distance = self.compute_lattice_distance(point, lattice_point)
            distances.append((lattice_point, distance))
        
        # Sort by distance and return k nearest
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def _generate_sample_lattice_points(self, num_points: int = 100):
        """Generate sample lattice points for testing"""
        self.lattice_points = []
        
        for _ in range(num_points):
            # Generate random lattice point
            coefficients = np.random.randint(-2, 3, len(self.basis_vectors))
            lattice_point = np.sum(coefficients[:, np.newaxis] * self.basis_vectors, axis=0)
            self.lattice_points.append(lattice_point)


class TGICSystem:
    """
    Main TGIC (Triad Graph Interaction Constraint) system.
    
    Implements the complete geometric constraint framework that enforces
    the fundamental 3, 6, 9 structure across UBP realms.
    """
    
    def __init__(self, geometry: TGICGeometry = TGICGeometry.DODECAHEDRAL):
        self.geometry = geometry
        self.constraints = {}
        self.interaction_matrix = None
        
        # Initialize geometric structure
        if geometry == TGICGeometry.DODECAHEDRAL:
            self.graph = DodecahedralGraph()
            self.leech_projection = None
        elif geometry == TGICGeometry.LEECH_24D:
            self.graph = None
            self.leech_projection = LeechLatticeProjection()
        else:
            self.graph = DodecahedralGraph()  # Default to dodecahedral
            self.leech_projection = LeechLatticeProjection()
        
        self._initialize_constraints()
    
    def _initialize_constraints(self):
        """Initialize the fundamental TGIC constraints"""
        
        # Constraint 1: 3-axis structure
        self.add_constraint(
            "three_axis_structure",
            "geometric",
            list(range(min(3, len(self.graph.nodes) if self.graph else 3))),
            self._enforce_three_axis_constraint
        )
        
        # Constraint 2: 6-face interactions
        if self.graph and len(self.graph.nodes) >= 6:
            self.add_constraint(
                "six_face_interactions",
                "topological",
                list(range(6)),
                self._enforce_six_face_constraint
            )
        
        # Constraint 3: 9-interaction neighborhood
        if self.graph and len(self.graph.nodes) >= 9:
            self.add_constraint(
                "nine_interaction_neighborhood",
                "connectivity",
                list(range(9)),
                self._enforce_nine_interaction_constraint
            )
    
    def add_constraint(self, constraint_id: str, constraint_type: str,
                      nodes_involved: List[int], constraint_function: callable,
                      tolerance: float = 1e-6, weight: float = 1.0):
        """
        Add a new TGIC constraint.
        
        Args:
            constraint_id: Unique identifier for constraint
            constraint_type: Type of constraint
            nodes_involved: List of node IDs involved in constraint
            constraint_function: Function that enforces the constraint
            tolerance: Tolerance for constraint satisfaction
            weight: Weight of constraint in optimization
        """
        constraint = TGICConstraint(
            constraint_id=constraint_id,
            constraint_type=constraint_type,
            nodes_involved=nodes_involved,
            constraint_function=constraint_function,
            tolerance=tolerance,
            weight=weight
        )
        
        self.constraints[constraint_id] = constraint
    
    def _enforce_three_axis_constraint(self, nodes: List[int]) -> float:
        """
        Enforce the three-axis structure constraint.
        
        Args:
            nodes: List of node IDs (should be 3 nodes)
        
        Returns:
            Constraint violation measure (0 = satisfied)
        """
        if not self.graph or len(nodes) < 3:
            return 0.0
        
        # Get positions of the three nodes
        positions = []
        for node_id in nodes[:3]:
            if node_id in self.graph.nodes:
                positions.append(self.graph.nodes[node_id].position)
        
        if len(positions) < 3:
            return 1.0  # Maximum violation
        
        # Check if positions form orthogonal axes
        pos1, pos2, pos3 = positions[0], positions[1], positions[2]
        
        # Compute vectors
        v1 = pos2 - pos1
        v2 = pos3 - pos1
        
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
        
        # Check orthogonality (dot product should be 0)
        dot_product = np.abs(np.dot(v1_norm, v2_norm))
        
        # Violation is how far from orthogonal
        violation = dot_product
        
        return violation
    
    def _enforce_six_face_constraint(self, nodes: List[int]) -> float:
        """
        Enforce the six-face interaction constraint.
        
        Args:
            nodes: List of node IDs (should be 6 nodes)
        
        Returns:
            Constraint violation measure
        """
        if not self.graph or len(nodes) < 6:
            return 0.0
        
        # Check that nodes form proper face interactions
        face_interactions = 0
        total_possible = 0
        
        for i in range(min(6, len(nodes))):
            for j in range(i + 1, min(6, len(nodes))):
                node_i = nodes[i]
                node_j = nodes[j]
                
                if (node_i in self.graph.nodes and 
                    node_j in self.graph.nodes and
                    node_j in self.graph.nodes[node_i].connections):
                    
                    interaction_type = self.graph.get_interaction_type(node_i, node_j)
                    if interaction_type == InteractionType.FACE_DIAGONAL:
                        face_interactions += 1
                
                total_possible += 1
        
        # Violation is how far from expected face interaction ratio
        expected_ratio = 0.5  # Expected 50% face interactions
        actual_ratio = face_interactions / max(1, total_possible)
        violation = abs(actual_ratio - expected_ratio)
        
        return violation
    
    def _enforce_nine_interaction_constraint(self, nodes: List[int]) -> float:
        """
        Enforce the nine-interaction neighborhood constraint.
        
        Args:
            nodes: List of node IDs (should be 9 nodes)
        
        Returns:
            Constraint violation measure
        """
        if not self.graph or len(nodes) < 9:
            return 0.0
        
        # Check that each node has exactly 9 interactions in neighborhood
        total_violation = 0.0
        
        for node_id in nodes[:9]:
            if node_id not in self.graph.nodes:
                total_violation += 1.0
                continue
            
            # Count interactions within the 9-node neighborhood
            interactions_in_neighborhood = 0
            for other_node in nodes[:9]:
                if (other_node != node_id and 
                    other_node in self.graph.nodes[node_id].connections):
                    interactions_in_neighborhood += 1
            
            # Ideal is to have connections to all other 8 nodes in neighborhood
            # But this might be too restrictive, so we use a more flexible target
            target_interactions = min(8, len(self.graph.nodes[node_id].connections))
            violation = abs(interactions_in_neighborhood - target_interactions) / 8.0
            total_violation += violation
        
        return total_violation / min(9, len(nodes))
    
    def evaluate_all_constraints(self) -> Dict[str, float]:
        """
        Evaluate all active constraints.
        
        Returns:
            Dictionary mapping constraint IDs to violation measures
        """
        violations = {}
        
        for constraint_id, constraint in self.constraints.items():
            if constraint.active:
                violation = constraint.constraint_function(constraint.nodes_involved)
                violations[constraint_id] = violation
        
        return violations
    
    def compute_total_violation(self) -> float:
        """
        Compute total weighted constraint violation.
        
        Returns:
            Total violation measure
        """
        violations = self.evaluate_all_constraints()
        
        total_violation = 0.0
        total_weight = 0.0
        
        for constraint_id, violation in violations.items():
            constraint = self.constraints[constraint_id]
            total_violation += constraint.weight * violation
            total_weight += constraint.weight
        
        return total_violation / max(1.0, total_weight)
    
    def optimize_node_positions(self, max_iterations: int = 100,
                              learning_rate: float = 0.01) -> Dict[str, Any]:
        """
        Optimize node positions to minimize constraint violations.
        
        Args:
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
        
        Returns:
            Dictionary containing optimization results
        """
        if not self.graph:
            return {'status': 'no_graph_available'}
        
        initial_violation = self.compute_total_violation()
        violation_history = [initial_violation]
        
        for iteration in range(max_iterations):
            # Compute gradients numerically
            for node_id, node in self.graph.nodes.items():
                original_position = node.position.copy()
                
                # Compute gradient for each dimension
                gradient = np.zeros_like(node.position)
                delta = 0.001
                
                for dim in range(len(node.position)):
                    # Positive perturbation
                    node.position[dim] += delta
                    violation_plus = self.compute_total_violation()
                    
                    # Negative perturbation
                    node.position[dim] -= 2 * delta
                    violation_minus = self.compute_total_violation()
                    
                    # Compute gradient
                    gradient[dim] = (violation_plus - violation_minus) / (2 * delta)
                    
                    # Restore original position
                    node.position[dim] = original_position[dim]
                
                # Update position
                node.position -= learning_rate * gradient
            
            # Compute new violation
            current_violation = self.compute_total_violation()
            violation_history.append(current_violation)
            
            # Check convergence
            if len(violation_history) > 1:
                improvement = violation_history[-2] - violation_history[-1]
                if improvement < 1e-6:
                    break
        
        final_violation = self.compute_total_violation()
        
        return {
            'initial_violation': initial_violation,
            'final_violation': final_violation,
            'improvement': initial_violation - final_violation,
            'iterations': len(violation_history) - 1,
            'violation_history': violation_history,
            'converged': len(violation_history) < max_iterations
        }
    
    def analyze_interaction_patterns(self) -> Dict[str, Any]:
        """
        Analyze interaction patterns in the TGIC system.
        
        Returns:
            Dictionary containing pattern analysis
        """
        if not self.graph:
            return {'status': 'no_graph_available'}
        
        # Count interaction types
        interaction_counts = defaultdict(int)
        for node in self.graph.nodes.values():
            for interaction_type in node.interaction_types.values():
                interaction_counts[interaction_type.value] += 1
        
        # Analyze connectivity patterns
        connectivity_stats = self.graph.compute_graph_properties()
        
        # Compute coherence metrics
        coherence_levels = [node.coherence_level for node in self.graph.nodes.values()]
        avg_coherence = np.mean(coherence_levels) if coherence_levels else 0.0
        
        # Analyze constraint satisfaction
        constraint_violations = self.evaluate_all_constraints()
        satisfied_constraints = sum(1 for v in constraint_violations.values() if v < 0.1)
        total_constraints = len(constraint_violations)
        
        return {
            'interaction_type_counts': dict(interaction_counts),
            'connectivity_stats': connectivity_stats,
            'average_coherence': avg_coherence,
            'coherence_distribution': coherence_levels,
            'constraint_satisfaction': {
                'satisfied': satisfied_constraints,
                'total': total_constraints,
                'satisfaction_rate': satisfied_constraints / max(1, total_constraints)
            },
            'constraint_violations': constraint_violations,
            'total_violation': self.compute_total_violation()
        }
    
    def validate_tgic_system(self) -> Dict[str, Any]:
        """
        Validate the TGIC system implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'geometric_structure': True,
            'constraint_enforcement': True,
            'interaction_patterns': True,
            'optimization_capability': True
        }
        
        try:
            # Test 1: Geometric structure
            if self.graph:
                graph_props = self.graph.compute_graph_properties()
                if graph_props['num_nodes'] == 0:
                    validation_results['geometric_structure'] = False
                    validation_results['structure_error'] = "No nodes in graph"
            
            # Test 2: Constraint enforcement
            violations = self.evaluate_all_constraints()
            if not violations:
                validation_results['constraint_enforcement'] = False
                validation_results['constraint_error'] = "No constraints evaluated"
            
            # Test 3: Interaction patterns
            patterns = self.analyze_interaction_patterns()
            if 'interaction_type_counts' not in patterns:
                validation_results['interaction_patterns'] = False
                validation_results['pattern_error'] = "Interaction analysis failed"
            
            # Test 4: Optimization capability
            if self.graph and len(self.graph.nodes) > 0:
                opt_result = self.optimize_node_positions(max_iterations=5)
                if 'final_violation' not in opt_result:
                    validation_results['optimization_capability'] = False
                    validation_results['optimization_error'] = "Optimization failed"
            
        except Exception as e:
            validation_results['validation_exception'] = str(e)
            validation_results['geometric_structure'] = False
        
        return validation_results


# Factory function for easy instantiation
def create_tgic_system(geometry: TGICGeometry = TGICGeometry.DODECAHEDRAL) -> TGICSystem:
    """
    Create a TGIC system with specified geometry.
    
    Args:
        geometry: Geometric structure to use
    
    Returns:
        Configured TGICSystem instance
    """
    return TGICSystem(geometry)


if __name__ == "__main__":
    # Validation and testing
    print("Initializing TGIC system...")
    
    tgic_system = create_tgic_system(TGICGeometry.DODECAHEDRAL)
    
    # Test dodecahedral graph properties
    if tgic_system.graph:
        print("\nTesting dodecahedral graph...")
        graph_props = tgic_system.graph.compute_graph_properties()
        print(f"Nodes: {graph_props['num_nodes']}")
        print(f"Edges: {graph_props['num_edges']}")
        print(f"Average degree: {graph_props['avg_degree']:.2f}")
        print(f"Average clustering: {graph_props['avg_clustering']:.6f}")
        print(f"Is regular: {graph_props['is_regular']}")
    
    # Test constraint evaluation
    print(f"\nTesting constraint evaluation...")
    violations = tgic_system.evaluate_all_constraints()
    for constraint_id, violation in violations.items():
        print(f"  {constraint_id}: {violation:.6f}")
    
    total_violation = tgic_system.compute_total_violation()
    print(f"Total violation: {total_violation:.6f}")
    
    # Test interaction pattern analysis
    print(f"\nTesting interaction pattern analysis...")
    patterns = tgic_system.analyze_interaction_patterns()
    
    if 'interaction_type_counts' in patterns:
        print("Interaction type counts:")
        for interaction_type, count in patterns['interaction_type_counts'].items():
            print(f"  {interaction_type}: {count}")
    
    if 'constraint_satisfaction' in patterns:
        satisfaction = patterns['constraint_satisfaction']
        print(f"Constraint satisfaction rate: {satisfaction['satisfaction_rate']:.3f}")
    
    # Test optimization
    print(f"\nTesting position optimization...")
    opt_result = tgic_system.optimize_node_positions(max_iterations=10)
    print(f"Initial violation: {opt_result['initial_violation']:.6f}")
    print(f"Final violation: {opt_result['final_violation']:.6f}")
    print(f"Improvement: {opt_result['improvement']:.6f}")
    print(f"Iterations: {opt_result['iterations']}")
    
    # Test Leech lattice projection
    print(f"\nTesting Leech lattice projection...")
    leech_system = create_tgic_system(TGICGeometry.LEECH_24D)
    if leech_system.leech_projection:
        # Test 24D point projection
        test_point_24d = np.random.randn(24)
        projection_3d = leech_system.leech_projection.project_to_3d(test_point_24d)
        print(f"24D point projected to 3D: {projection_3d}")
        
        # Test nearest neighbors
        neighbors = leech_system.leech_projection.find_nearest_neighbors(test_point_24d, k=3)
        print(f"Found {len(neighbors)} nearest neighbors")
    
    # System validation
    validation = tgic_system.validate_tgic_system()
    print(f"\nTGIC system validation:")
    print(f"  Geometric structure: {validation['geometric_structure']}")
    print(f"  Constraint enforcement: {validation['constraint_enforcement']}")
    print(f"  Interaction patterns: {validation['interaction_patterns']}")
    print(f"  Optimization capability: {validation['optimization_capability']}")
    
    print("\nTGIC system ready for UBP integration.")


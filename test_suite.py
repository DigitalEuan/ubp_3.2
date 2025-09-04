"""
Universal Binary Principle (UBP) Framework v3.2+ - Test Suite
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================
"""
import os
import sys
import numpy as np
import math
import time
import json
from typing import Dict, Any, List, Optional, Tuple
import logging # Import logging

# Adjust sys.path to ensure all modules are discoverable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all UBP modules that might have validation functions or need testing
from ubp_config import get_config, reset_config, UBPConfig, RealmConfig
from hex_dictionary import HexDictionary
from global_coherence import GlobalCoherenceIndex, create_global_coherence_system
from enhanced_nrci import EnhancedNRCI, create_enhanced_nrci_system, CoherenceRegime
from observer_scaling import ObserverScaling, create_observer_scaling_system, ObserverState, ObserverIntentType, ScaleRegime
from carfe import CARFEFieldEquation, create_carfe_system, FieldState, FieldTopology, CARFEMode
from dot_theory import DotTheorySystem, create_dot_theory_system, DotState, PurposeType, ConsciousnessLevel, DotGeometry
from spin_transition import SpinTransitionSystem, create_spin_transition_system, SpinRealm
from p_adic_correction import AdvancedErrorCorrectionModule, create_padic_corrector # Corrected import: new class name
from glr_base import GLRFramework, create_glr_framework, GLRLevel, HammingCode, BCHCode, GolayCode
from level_7_global_golay import GlobalGolayCorrection, create_global_golay_correction
from prime_resonance import PrimeResonanceCoordinateSystem, create_prime_resonance_system
from tgic import TGICSystem, create_tgic_system, TGICGeometry
from hardware_emulation import HardwareEmulationSystem, create_hardware_system, CPUEmulator, MemoryEmulator, Instruction, InstructionType, create_cpu_emulator, create_memory_emulator # Import factory functions explicitly
from ubp_lisp import UBPLispInterpreter, create_ubp_lisp_interpreter, UBPValue, UBPType
from crv_database import EnhancedCRVDatabase, CRVProfile, SubCRV
from enhanced_crv_selector import AdaptiveCRVSelector, CRVSelectionResult
from htr_engine import HTREngine
from ubp_256_study_evolution import UBP256Evolution
from ubp_pattern_analysis import UBPPatternAnalyzer
from ubp_pattern_generator_1 import run_ubp_simulation as run_basic_pattern_generation_test
from ubp_pattern_integrator import UBPPatternIntegrator
from runtime import Runtime, SimulationState, SimulationResult
from energy import energy, resonance_strength, structural_optimality, observer_effect_factor, cosmic_constant, spin_information_factor, calculate_energy_for_realm, weighted_toggle_matrix_sum
from kernels import resonance_kernel, coherence, normalized_coherence, global_coherence_invariant, calculate_weighted_frequency_average, generate_oscillating_signal
from metrics import nrci, coherence_pressure_spatial, fractal_dimension, calculate_system_coherence_score
from toggle_ops import toggle_and, toggle_xor, toggle_or, resonance_toggle, entanglement_toggle, superposition_toggle, hybrid_xor_resonance, spin_transition, apply_tgic_constraint
from state import OffBit, MutableBitfield, UBPState


class UBPTestSuite:
    def __init__(self, output_dir: str = "./output/test_results/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.results: Dict[str, Any] = {}
        
        # Configure logging for debug output
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Initialize UBPConfig first, and pass to other modules as needed
        reset_config() # Ensure a clean slate for the config
        self.config = get_config(environment="testing") # Use 'testing' environment for consistent tests
        
        # Initialize HexDictionary first, and pass to modules that use it
        self.hex_dict = HexDictionary()
        # Ensure HexDictionary is clean for the test run to avoid interference
        self.hex_dict.clear_all() 

    def run_all_tests(self):
        print("\n--- Running UBP System-Wide Validation Test Suite ---")
        self.results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.results['overall_status'] = 'PENDING'
        
        test_methods = [
            self.test_ubp_config,
            self.test_hex_dictionary,
            self.test_global_coherence,
            self.test_enhanced_nrci,
            self.test_observer_scaling,
            self.test_carfe,
            self.test_dot_theory,
            self.test_spin_transition,
            self.test_p_adic_correction,
            self.test_glr_framework_base,
            self.test_level_7_global_golay,
            self.test_prime_resonance,
            self.test_tgic_system,
            self.test_hardware_emulation,
            self.test_ubp_lisp,
            self.test_crv_database,
            self.test_enhanced_crv_selector,
            self.test_htr_engine,
            self.test_ubp_256_study_evolution,
            self.test_ubp_pattern_analysis,
            self.test_ubp_pattern_generator_1,
            self.test_ubp_pattern_integrator,
            self.test_runtime,
            self.test_energy_module,
            self.test_kernels_module,
            self.test_metrics_module,
            self.test_toggle_ops_module,
            self.test_state_module,
        ]

        total_passed = 0
        total_failed = 0
        
        for test_func in test_methods:
            module_name = test_func.__name__.replace('test_', '')
            print(f"\n🧪 Running test for: {module_name}...")
            try:
                test_func()
                status = self.results.get(module_name, {}).get('status', 'PASSED')
                if status == 'PASSED':
                    total_passed += 1
                else:
                    total_failed += 1
                print(f"  Status: {status}")
            except Exception as e:
                self.results[module_name] = {'status': 'FAILED', 'exception': str(e)}
                total_failed += 1
                print(f"  Status: FAILED due to unhandled exception: {e}")
        
        self.results['summary'] = {
            'total_tests_run': len(test_methods),
            'total_passed': total_passed,
            'total_failed': total_failed
        }
        self.results['overall_status'] = 'PASSED' if total_failed == 0 else 'FAILED'

        print("\n--- UBP System-Wide Validation Test Suite Complete ---")
        print(f"Overall Status: {self.results['overall_status']}")
        print(f"Passed: {total_passed}, Failed: {total_failed}")

        results_filepath = os.path.join(self.output_dir, "ubp_test_suite_results.json")
        with open(results_filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Detailed results saved to {results_filepath}")
        
        # Clean up HexDictionary after tests
        self.hex_dict.clear_all()

    def _record_test_result(self, module: str, status: str, details: Optional[Dict[str, Any]] = None):
        if details is None: # Use 'is' for comparison to None
            details = {}
        self.results[module] = {'status': status, 'details': details}

    # --- Individual Test Methods ---
    
    def test_ubp_config(self):
        module_name = 'ubp_config'
        try:
            # Check if constants are loaded
            assert self.config.constants.PI == math.pi
            assert self.config.get_realm_config('quantum') is not None
            self._record_test_result(module_name, 'PASSED', self.config.get_summary())
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_hex_dictionary(self):
        module_name = 'hex_dictionary'
        try:
            # Test store/retrieve
            test_data = "hello world"
            test_hash = self.hex_dict.store(test_data, 'str', metadata={'test': 'true'})
            retrieved_data = self.hex_dict.retrieve(test_hash)
            assert retrieved_data == test_data # Use == for comparison to test_data
            
            # Test metadata
            meta = self.hex_dict.get_metadata(test_hash)
            # The structure of metadata changed in ubp_lisp, ensure this test reflects hex_dictionary directly.
            # HexDictionary stores metadata directly, not nested under 'ubp_lisp_type' and 'original_lisp_metadata'
            # unless ubp_lisp is the one storing it. Here, we're calling hex_dict directly.
            assert meta['test'] == 'true' # Use == for comparison to 'true'

            # Test deletion
            self.hex_dict.delete(test_hash)
            assert self.hex_dict.retrieve(test_hash) == None # Use == for comparison to None

            self._record_test_result(module_name, 'PASSED', self.hex_dict.get_metadata_stats())
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_global_coherence(self):
        module_name = 'global_coherence'
        try:
            gci_system = create_global_coherence_system()
            validation_results = gci_system.validate_system()
            assert validation_results['f_avg_calculation'] is True
            assert validation_results['p_gci_calculation'] is True
            assert validation_results['p_gci_range_valid'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_enhanced_nrci(self):
        module_name = 'enhanced_nrci'
        try:
            nrci_system = create_enhanced_nrci_system()
            validation_results = nrci_system.validate_system()
            assert validation_results['mathematical_validation'] is True
            assert validation_results['regime_classification'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_observer_scaling(self):
        module_name = 'observer_scaling'
        try:
            observer_system = create_observer_scaling_system()
            validation_results = observer_system.validate_observer_scaling()
            assert validation_results['formula_implementation'] is True
            assert validation_results['purpose_tensor_calculation'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_carfe(self):
        module_name = 'carfe'
        try:
            carfe_system = create_carfe_system()
            validation_results = carfe_system.validate_carfe_system()
            assert validation_results['recursive_evolution'] is True
            assert validation_results['expansive_dynamics'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})
            
    def test_dot_theory(self):
        module_name = 'dot_theory'
        try:
            dot_system = create_dot_theory_system()
            validation_results = dot_system.validate_dot_theory_system()
            assert validation_results['dot_creation'] is True
            assert validation_results['purpose_tensor_calculation'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_spin_transition(self):
        module_name = 'spin_transition'
        try:
            spin_system = create_spin_transition_system(SpinRealm.QUANTUM)
            validation_results = spin_system.validate_spin_transition_system()
            assert validation_results['spin_system_creation'] is True
            assert validation_results['transition_calculation'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_p_adic_correction(self):
        module_name = 'p_adic_correction'
        try:
            # Use the new class name AdvancedErrorCorrectionModule
            padic_corrector = create_padic_corrector() # Factory returns AdvancedErrorCorrectionModule
            validation_results = padic_corrector.validate_padic_system()
            
            assert validation_results['padic_arithmetic'] is True
            assert validation_results['adelic_operations'] is True
            assert validation_results['error_detection'] is True
            assert validation_results['error_correction'] is True
            assert validation_results['hensel_lifting'] is True
            assert validation_results['fibonacci_encoding'] is True # New: Assert Fibonacci test

            # Additional test for Fibonacci encoding/decoding specifically.
            test_data_fib = np.array([10.0, 20.5, 3.14])
            encoded_fib = padic_corrector.encode_with_error_correction(test_data_fib, method="fibonacci", redundancy_level=0.5)
            decoded_fib, _ = padic_corrector.decode_with_error_correction(encoded_fib)
            
            # Use allclose with a small tolerance for floating point comparisons
            assert np.allclose(test_data_fib, decoded_fib, atol=1e-2) # Adjusted tolerance for floating point

            # Test corrupted Fibonacci data
            corrupted_encoded_fib = encoded_fib.copy()
            # Introduce an error: flip a bit in the encoded sequence
            if corrupted_encoded_fib['encoded_state'].encoded_bits:
                corrupted_encoded_fib['encoded_state'].encoded_bits[0] = 1 - corrupted_encoded_fib['encoded_state'].encoded_bits[0]
            
            corrected_decoded_fib, correction_info_fib = padic_corrector.correct_corrupted_data(corrupted_encoded_fib)
            assert correction_info_fib.corrected_errors > 0 or correction_info_fib.correction_success_rate == 1.0 # Assert corrections or already perfect
            
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': f"Unhandled exception: {str(e)}"})


    def test_glr_framework_base(self):
        module_name = 'glr_base'
        try:
            glr_framework = create_glr_framework()
            # Need to register at least one processor for comprehensive validation
            # Using GlobalGolayCorrection as an example.
            glr_framework.register_processor(create_global_golay_correction()) 
            validation_results = glr_framework.validate_framework()
            assert validation_results['framework_functional'] is True
            # Expecting 'all_levels_covered' to be False if only one is registered
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_level_7_global_golay(self):
        module_name = 'level_7_global_golay'
        try:
            golay_processor = create_global_golay_correction()
            validation_results = golay_processor.validate_golay_system()
            assert validation_results['matrix_dimensions_correct'] is True
            assert validation_results['syndrome_calculation_correct'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_prime_resonance(self):
        module_name = 'prime_resonance'
        try:
            pr_system = create_prime_resonance_system()
            validation_results = pr_system.validate_system()
            assert validation_results['coordinate_mapping_test'] is True
            assert validation_results['resonance_calculation_test'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_tgic_system(self):
        module_name = 'tgic'
        try:
            tgic_system = create_tgic_system(TGICGeometry.DODECAHEDRAL)
            validation_results = tgic_system.validate_tgic_system()
            assert validation_results['geometric_structure'] is True
            assert validation_results['constraint_enforcement'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_hardware_emulation(self):
        module_name = 'hardware_emulation'
        try:
            hw_system = create_hardware_system("test_hw_system", self.config) # Pass config
            cpu = create_cpu_emulator("test_cpu", self.config) # Pass config
            memory = create_memory_emulator("test_mem", self.config) # Pass config
            hw_system.add_component(cpu)
            hw_system.add_component(memory)
            validation_results = hw_system.validate_hardware_emulation()
            assert validation_results['cpu_emulation'] is True
            assert validation_results['memory_emulation'] is True
            assert validation_results['ubp_integration'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_ubp_lisp(self):
        module_name = 'ubp_lisp'
        try:
            # Need to pass self.hex_dict here.
            ubp_lisp_interpreter = create_ubp_lisp_interpreter(hex_dict_instance=self.hex_dict)
            validation_results = ubp_lisp_interpreter.validate_ubp_lisp_system()
            assert validation_results['parser_functionality'] is True
            assert validation_results['basic_evaluation'] is True
            assert validation_results['ubp_operations'] is True
            assert validation_results['bitbase_integration'] is True
            assert validation_results['function_definition'] is True
            self._record_test_result(module_name, 'PASSED', validation_results)
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_crv_database(self):
        module_name = 'crv_database'
        try:
            crv_db = EnhancedCRVDatabase()
            # Test initialization from config
            assert len(crv_db.crv_profiles) > 0
            
            # Test getting a profile
            em_profile = crv_db.get_crv_profile('electromagnetic')
            assert em_profile is not None
            assert em_profile.main_crv == self.config.get_realm_config('electromagnetic').main_crv # Use == for comparison to main_crv
            
            # Test optimal CRV selection (simplified data_characteristics)
            data_chars = {'frequency': 2.45e9, 'complexity': 0.5, 'noise_level': 0.1, 'target_nrci': 0.99}
            optimal_crv, reason = crv_db.get_optimal_crv('electromagnetic', data_chars)
            assert optimal_crv is not None
            assert isinstance(optimal_crv, float)
            self._record_test_result(module_name, 'PASSED', {'optimal_crv': optimal_crv, 'reason': reason})
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})
            
    def test_enhanced_crv_selector(self):
        module_name = 'enhanced_crv_selector'
        try:
            crv_selector = AdaptiveCRVSelector()
            data_chars = {'frequency': 2.45e9, 'complexity': 0.6, 'noise_level': 0.05, 'target_nrci': 0.99, 'speed_priority': True}
            
            result = crv_selector.select_optimal_crv('electromagnetic', data_chars)
            assert isinstance(result, CRVSelectionResult)
            assert result.selected_crv is not None
            assert result.confidence_score > 0.5
            self._record_test_result(module_name, 'PASSED', result.to_dict())
        except Exception as e: # Catch all exceptions as select_optimal_crv can raise ValueError
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_htr_engine(self):
        module_name = 'htr_engine'
        try:
            htr_engine = HTREngine(realm_name='quantum')
            # Create a simple lattice (e.g., 5 atoms in a line)
            lattice_coords = np.array([[0,0,0], [1,0,0], [2,0,0], [3,0,0], [4,0,0]], dtype=float)
            htr_results = htr_engine.process_with_htr(lattice_coords=lattice_coords * 1e-9, realm='quantum') # Scale to nm
            
            assert 'energy' in htr_results
            assert 'nrci' in htr_results
            assert htr_results['energy'] > 0
            assert htr_results['nrci'] > 0
            self._record_test_result(module_name, 'PASSED', htr_results)
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_ubp_256_study_evolution(self):
        module_name = 'ubp_256_study_evolution'
        try:
            study_evolution = UBP256Evolution(resolution=64, config=self.config) # Use smaller resolution for speed
            results = study_evolution.run_comprehensive_study(output_dir=os.path.join(self.output_dir, "256_study_output"))
            
            assert 'patterns' in results
            assert len(results['patterns']) > 0
            
            # Check a sample pattern's analysis
            sample_key = list(results['patterns'].keys())[0]
            assert 'coherence_score' in results['patterns'][sample_key]['analysis']
            assert results['patterns'][sample_key]['analysis']['coherence_score'] >= 0
            self._record_test_result(module_name, 'PASSED', {'sample_pattern_analysis': results['patterns'][sample_key]['analysis']})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_ubp_pattern_analysis(self):
        module_name = 'ubp_pattern_analysis'
        try:
            analyzer = UBPPatternAnalyzer(config=self.config)
            test_pattern = analyzer.generate_harmonic_test_patterns(size=64)['fundamental'] # Smaller size for speed
            analysis = analyzer.analyze_coherence_pressure(test_pattern)
            
            assert 'coherence_score' in analysis
            assert analysis['coherence_score'] >= 0
            assert 'harmonic_ratios' in analysis
            self._record_test_result(module_name, 'PASSED', analysis)
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_ubp_pattern_generator_1(self):
        module_name = 'ubp_pattern_generator_1'
        try:
            test_frequencies = [self.config.realms['electromagnetic'].main_crv]
            test_realm_names = ['electromagnetic']
            
            output_gen_dir = os.path.join(self.output_dir, "pattern_gen_output")
            os.makedirs(output_gen_dir, exist_ok=True)

            results = run_basic_pattern_generation_test(
                test_frequencies, 
                test_realm_names, 
                output_gen_dir, 
                self.config, 
                resolution=32 # Smaller resolution for speed
            )
            
            assert len(results) == 1 # Use == for comparison to 1
            assert 'pattern_data' in results[0]
            assert 'nrci_from_htr' in results[0]
            assert results[0]['nrci_from_htr'] > 0
            self._record_test_result(module_name, 'PASSED', {'sample_result': results[0]})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_ubp_pattern_integrator(self):
        module_name = 'ubp_pattern_integrator'
        try:
            integrator = UBPPatternIntegrator(hex_dictionary_instance=self.hex_dict, config=self.config)
            
            # Generate and store a basic pattern
            stored_info = integrator.generate_and_store_patterns(
                pattern_generation_method='basic_simulation',
                frequencies_or_crv_keys=[self.config.realms['quantum'].main_crv],
                realm_contexts=['quantum'],
                resolution=32 # Very small for speed
            )
            assert len(stored_info) == 1 # Use == for comparison to 1
            sample_key = list(stored_info.keys())[0] # Get the generated key like 'basic_pattern_0'
            sample_hash = stored_info[sample_key]['hash'] # Access the hash using the key
            
            # Retrieve the pattern
            retrieved_pattern_dict = integrator.get_pattern_by_hash(sample_hash)
            assert retrieved_pattern_dict is not None
            assert 'pattern_array' in retrieved_pattern_dict
            
            # Explicitly assert the type - this should now pass
            assert isinstance(retrieved_pattern_dict['pattern_array'], np.ndarray)

            # Search for the pattern
            search_criteria = {"data_type": "ubp_pattern_basic_simulation"}
            found_patterns = integrator.search_patterns_by_metadata(search_criteria, limit=1)
            assert len(found_patterns) == 1 # Use == for comparison to 1
            assert found_patterns[0]['hash'] == sample_hash # Use == for comparison to sample_hash
            
            self._record_test_result(module_name, 'PASSED', {'stored_info': stored_info})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_runtime(self):
        module_name = 'runtime'
        try:
            runtime = Runtime(hardware_profile='desktop_8gb') # Use a specific profile
            runtime.set_realm('electromagnetic')
            runtime.initialize_bitfield(pattern='sparse_random', density=0.01, seed=42)
            
            sim_result = runtime.run_simulation(steps=5, operations_per_step=2, record_timeline=True)
            
            assert isinstance(sim_result, SimulationResult)
            assert sim_result.final_state.nrci_value >= 0
            assert len(sim_result.timeline) == 6 # Initial + 5 steps # Use == for comparison to 6
            self._record_test_result(module_name, 'PASSED', sim_result.final_state.to_dict())
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_energy_module(self):
        module_name = 'energy'
        try:
            # Test energy calculation
            M_val = 100
            R_val = resonance_strength(0.9, 0.1)
            # `structural_optimality` requires lists of floats, max_distance.
            # Using placeholder values for distances, active_bits (0-11)
            S_opt_val = structural_optimality([1.0,2.0,3.0], 5.0, [1,1,0,1,0,0,0,0,0,0,0,0])
            P_GCI_val = global_coherence_invariant(1e9, self.config.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD_DEFAULT) # Using kernels func
            O_observer_val = observer_effect_factor("intentional")
            # `spin_information_factor` requires list of probabilities
            I_spin_val = spin_information_factor([0.5, 0.5])
            w_sum_val = weighted_toggle_matrix_sum([0.5,0.5], [0.1,0.2])
            
            total_energy = energy(M_val, R=R_val, S_opt=S_opt_val, P_GCI=P_GCI_val, O_observer=O_observer_val, I_spin=I_spin_val, w_sum=w_sum_val)
            assert total_energy > 0
            
            # Test realm energy calculation
            realm_energy = calculate_energy_for_realm('quantum', M_val)
            assert realm_energy > 0
            self._record_test_result(module_name, 'PASSED', {'total_energy': total_energy, 'realm_energy': realm_energy})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_kernels_module(self):
        module_name = 'kernels'
        try:
            res_kernel = resonance_kernel(1.0)
            assert 0 < res_kernel <= 1
            
            sig1 = [1.0, 0.5, -1.0]
            sig2 = [1.0, 0.5, -1.0]
            coh = coherence(sig1, sig2)
            assert coh > 0
            
            norm_coh = normalized_coherence(sig1, sig2)
            assert norm_coh == 1.0 # Perfect match # Use == for comparison to 1.0
            
            f_avg = calculate_weighted_frequency_average()
            gci = global_coherence_invariant(f_avg)
            assert -1 <= gci <= 1
            
            signal_gen = generate_oscillating_signal(100.0, 0.0, 1.0)
            assert len(signal_gen) == 1000 # Use == for comparison to 1000
            
            self._record_test_result(module_name, 'PASSED', {'resonance_kernel': res_kernel, 'gci': gci})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_metrics_module(self):
        module_name = 'metrics'
        try:
            sim_data = [1.0, 2.0, 3.0, 4.0]
            target_data = [1.0, 2.0, 3.0, 4.0]
            nrci_val = nrci(sim_data, target_data)
            assert nrci_val == 1.0 # Use == for comparison to 1.0
            
            # `coherence_pressure_spatial` expects a list of distances, max_distances and active_bits (0-11)
            cp_spatial = coherence_pressure_spatial([1.0, 2.0], [5.0, 5.0], [1,1,1,0,0,0,0,0,0,0,0,0])
            assert cp_spatial >= 0
            
            fractal_dim = fractal_dimension(4)
            assert fractal_dim == math.log(4) / math.log(2) # Use == for comparison to math.log(4) / math.log(2)
            
            coherence_score = calculate_system_coherence_score(0.9, 0.1, 2.0, 0.8, 0.9)
            assert 0 <= coherence_score <= 1
            self._record_test_result(module_name, 'PASSED', {'nrci': nrci_val, 'coherence_score': coherence_score})
        except Exception as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_toggle_ops_module(self):
        module_name = 'toggle_ops'
        try:
            offbit1 = OffBit(0b101) # 5
            offbit2 = OffBit(0b011) # 3
            
            and_res = toggle_and(offbit1, offbit2)
            assert and_res.value == 0b001 # 1 # Use == for comparison to 0b001
            
            xor_res = toggle_xor(offbit1, offbit2)
            assert xor_res.value == 0b110 # 6 # Use == for comparison to 0b110
            
            or_res = toggle_or(offbit1, offbit2)
            assert or_res.value == 0b111 # 7 # Use == for comparison to 0b111
            
            res_res = resonance_toggle(offbit1, 100.0, 0.01)
            assert isinstance(res_res, OffBit)
            
            ent_res = entanglement_toggle(offbit1, offbit2, 0.98)
            assert isinstance(ent_res, OffBit)
            
            sup_res = superposition_toggle([offbit1, offbit2], [0.5, 0.5])
            # The exact result of superposition_toggle is int((val1 * w1 + val2 * w2)),
            # so (5 * 0.5 + 3 * 0.5) = 2.5 + 1.5 = 4.0.
            assert sup_res.value == 4 # Use == for comparison to 4
            
            htr_res = hybrid_xor_resonance(offbit1, offbit2, 1.0)
            assert isinstance(htr_res, OffBit)
            
            spin_res = spin_transition(offbit1, 0.5)
            assert isinstance(spin_res, OffBit)
            
            # Test TGIC constraint function as well (this module exports it)
            tgic_res = apply_tgic_constraint(True, True, False, offbit1, offbit2, frequency=1000.0, time=0.01)
            assert isinstance(tgic_res, OffBit) # Should call resonance_toggle
            
            self._record_test_result(module_name, 'PASSED', {'and_result': and_res.value, 'xor_result': xor_res.value})
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})

    def test_state_module(self):
        module_name = 'state'
        try:
            offbit = OffBit(0x123456)
            assert offbit.value == 0x123456 # Use == for comparison to 0x123456
            assert offbit.active_bits == 9 # Use == for comparison to 9
            
            bitfield = MutableBitfield(size=10)
            bitfield.set_offbit(0, offbit)
            assert bitfield.get_offbit(0).value == offbit.value # Use == for comparison to offbit.value
            assert bitfield.active_count == 1 # Use == for comparison to 1
            
            bitfield.toggle_offbit(0)
            assert bitfield.get_offbit(0).value == (0x123456 ^ 0xFFFFFF) # Use == for comparison to (0x123456 ^ 0xFFFFFF)
            
            ubp_state = UBPState(bitfield=bitfield, realm='quantum')
            ubp_state.update_coherence()
            assert ubp_state.coherence > 0
            ubp_state.compute_energy()
            assert ubp_state.energy > 0
            self._record_test_result(module_name, 'PASSED', {'offbit_value': offbit.value, 'ubp_state_energy': ubp_state.energy})
        except AssertionError as e:
            self._record_test_result(module_name, 'FAILED', {'error': str(e)})


def main():
    test_suite = UBPTestSuite()
    test_suite.run_all_tests()

if __name__ == '__main__':
    main()
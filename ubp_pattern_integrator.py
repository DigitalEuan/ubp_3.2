"""
Universal Binary Principle (UBP) Framework v3.2+ - UBP Pattern Integrator
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================

"""
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Import UBP core components
from ubp_config import get_config, UBPConfig, RealmConfig
from hex_dictionary import HexDictionary

# Import the adapted pattern generation and analysis modules
from ubp_pattern_generator_1 import run_ubp_simulation as run_basic_pattern_generation
from ubp_256_study_evolution import UBP256Evolution
from ubp_pattern_analysis import UBPPatternAnalyzer

class UBPPatternIntegrator:
    """
    Integrates UBP pattern generation, analysis, and persistent storage.
    Provides an API for the AI Realm Architect to manage cymatic pattern data.
    """
    
    def __init__(self, hex_dictionary_instance: Optional[HexDictionary] = None, config: Optional[UBPConfig] = None):
        self.config = config if config else get_config()
        self.hex_dict = hex_dictionary_instance if hex_dictionary_instance else HexDictionary()
        self.output_dir = "./output/ubp_patterns/" # Use local directory
        os.makedirs(self.output_dir, exist_ok=True)
        print("✅ UBPPatternIntegrator Initialized.")
        print(f"   HexDictionary storage path: {self.hex_dict.storage_dir}")
        print(f"   Temporary pattern images output to: {self.output_dir}")

    def _create_standard_metadata(self, 
                                  data_type: str, 
                                  unique_id: str, 
                                  realm_context: str, 
                                  description: str, 
                                  source_module: str,
                                  tags: Optional[List[str]] = None,
                                  hashtags: Optional[List[str]] = None,
                                  source_metadata: Optional[Dict[str, Any]] = None,
                                  associated_patterns: Optional[List[str]] = None,
                                  additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Creates a standardized metadata dictionary for HexDictionary entries.
        This function is flexible enough to accommodate various data types,
        including structured data like the Periodic Table, by placing specific
        fields or schemas within 'additional_metadata'.
        """
        timestamp = datetime.now().isoformat()
        
        # Ensure tags and hashtags are lists
        tags = tags if tags is not None else []
        hashtags = hashtags if hashtags is not None else []
        
        # Add common tags derived from core fields
        tags.extend([data_type, realm_context, source_module.replace('.py', '')])
        hashtags.extend([f"#{data_type.upper().replace(' ', '_')}", f"#{realm_context.upper().replace(' ', '_')}", f"#{source_module.replace('.py', '').upper().replace(' ', '_')}"])

        # Remove duplicates
        tags = list(set(tags))
        hashtags = list(set(hashtags))

        standard_meta = {
            "ubp_version": "3.1.1", # Hardcode for now, or get from config
            "timestamp": timestamp,
            "data_type": data_type,
            "unique_id": unique_id,
            "realm_context": realm_context,
            "description": description,
            "source_module": source_module,
            "tags": tags,
            "hashtags": hashtags,
            "source_metadata": source_metadata if source_metadata is not None else {},
            "associated_patterns": associated_patterns if associated_patterns is not None else [],
        }
        
        if additional_metadata:
            standard_meta.update(additional_metadata)
            
        return standard_meta

    def store_pattern_data(self, pattern_array: np.ndarray, analysis_results: Dict[str, Any], 
                           pattern_metadata: Dict[str, Any]) -> str:
        """
        Stores a pattern (NumPy array) and its analysis results as metadata in HexDictionary,
        adhering to the new standardized metadata structure.
        """
        
        # Extract core fields for standard metadata
        data_type = pattern_metadata.pop("data_type", "ubp_pattern")
        unique_id = pattern_metadata.pop("unique_id", f"pattern_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
        realm_context = pattern_metadata.pop("realm_context", "universal") # Patterns can be universal
        description = pattern_metadata.pop("description", "UBP generated cymatic pattern.")
        source_module = pattern_metadata.pop("source_module", "ubp_pattern_integrator.py")
        
        # Combine remaining pattern_metadata into additional_metadata
        full_metadata = self._create_standard_metadata(
            data_type=data_type,
            unique_id=unique_id,
            realm_context=realm_context,
            description=description,
            source_module=source_module,
            additional_metadata={"pattern_details": pattern_metadata, "analysis_results": analysis_results}
        )
        
        # Store the NumPy array directly, specifying 'array' data type for HexDictionary
        pattern_hash = self.hex_dict.store(pattern_array, 'array', metadata=full_metadata)
        
        # Make the print statement robust to coherence_score being 'N/A' or another type
        coherence_score_for_print = analysis_results.get('coherence_score', 'N/A')
        if isinstance(coherence_score_for_print, (float, int)):
            coherence_score_str = f"{float(coherence_score_for_print):.3f}"
        else:
            coherence_score_str = str(coherence_score_for_print)

        print(f"📦 Stored pattern (hash: {pattern_hash[:8]}...) with metadata. Coherence: {coherence_score_str}")
        
        return pattern_hash

    def generate_and_store_patterns(self, 
                                     pattern_generation_method: str = '256_study',
                                     frequencies_or_crv_keys: Optional[List[float]] = None, # Expects frequencies directly
                                     realm_contexts: Optional[List[str]] = None, # New parameter for realm context
                                     resolution: int = 256,
                                     num_basic_patterns_to_store: int = 5) -> Dict[str, Any]:
        """
        Generates patterns using specified methods, analyzes them, and stores them in HexDictionary.
        
        Args:
            pattern_generation_method: '256_study' or 'basic_simulation'.
            frequencies_or_crv_keys: List of frequencies (for basic) or CRV keys (for 256 study).
            realm_contexts: Optional list of realm names, corresponding to frequencies.
            resolution: Resolution for pattern generation (e.g., 256 for 256_study).
            num_basic_patterns_to_store: How many patterns to generate for 'basic_simulation'.
            
        Returns:
            A dictionary containing generated pattern hashes and analysis results.
        """
        print(f"\n⚙️ Generating and storing patterns using method: {pattern_generation_method}...")
        stored_patterns_info = {}
        
        if pattern_generation_method == '256_study':
            # This method works with CRV keys (strings), not raw frequencies.
            # So, frequencies_or_crv_keys should be a list of strings like ['CRV_BASE', 'CRV_PHI'].
            if not isinstance(frequencies_or_crv_keys, list) or not all(isinstance(x, str) for x in frequencies_or_crv_keys):
                print("Warning: For '256_study', 'frequencies_or_crv_keys' should be a list of CRV key strings.")
                # Fallback to default CRV keys if input is incorrect for 256_study
                study_evolution = UBP256Evolution(resolution=resolution, config=self.config)
                frequencies_or_crv_keys = list(study_evolution.crv_constants.keys())


            study_evolution = UBP256Evolution(resolution=resolution, config=self.config)
            study_results = study_evolution.run_comprehensive_study(output_dir=self.output_dir)
            
            # Store all patterns generated by the 256 study
            for key, pattern_data in study_results['patterns'].items():
                pattern_array = np.array(pattern_data['pattern_data_array']) # Ensure it's a numpy array
                analysis_results = pattern_data['analysis']
                
                # Construct pattern_metadata for the new standard
                unique_id = f"pattern_256study_{key}_{datetime.now().strftime('%f')}"
                realm_context = pattern_data.get('realm', 'universal') # If 256 study specifies realm
                description = f"UBP 256 study pattern for CRV {pattern_data['base_crv']} with {pattern_data['removal_type']} removal."
                
                # Pass specific pattern details and analysis results separately
                pattern_details = {
                    "crv_key": pattern_data['base_crv'],
                    "removal_type": pattern_data['removal_type'],
                    "resolution": resolution,
                    "pattern_type": "crv_harmonic_filtered"
                }

                pattern_hash = self.store_pattern_data(
                    pattern_array=pattern_array,
                    analysis_results=analysis_results,
                    pattern_metadata={
                        "data_type": "ubp_pattern_256study",
                        "unique_id": unique_id,
                        "realm_context": realm_context,
                        "description": description,
                        "source_module": "ubp_256_study_evolution.py",
                        "pattern_details": pattern_details # This will go into 'additional_metadata'
                    }
                )
                stored_patterns_info[key] = {"hash": pattern_hash, "analysis_summary": analysis_results['pattern_classification']}
            
            # Also visualize the results for the 256 study
            study_evolution.visualize_results(study_results, output_dir=self.output_dir)

        elif pattern_generation_method == 'basic_simulation':
            # Frequencies_or_crv_keys should be a list of floats (actual frequencies).
            # realm_contexts should be a list of strings (matching the frequencies).

            frequencies_to_use = frequencies_or_crv_keys if frequencies_or_crv_keys is not None else []
            realms_for_sim = realm_contexts if realm_contexts is not None else ['universal'] * len(frequencies_to_use)

            if not frequencies_to_use:
                # Fallback: Use predefined CRVs from config for various realms
                all_realm_crvs = list(self.config.realms.items())
                for i in range(min(num_basic_patterns_to_store, len(all_realm_crvs))):
                    realm_name, realm_cfg = all_realm_crvs[i]
                    frequencies_to_use.append(realm_cfg.main_crv)
                    realms_for_sim.append(realm_name)
            
            if len(frequencies_to_use) != len(realms_for_sim):
                print(f"Warning: Number of frequencies ({len(frequencies_to_use)}) does not match number of realms ({len(realms_for_sim)}). Using 'universal' for unmatched realms.")
                # Pad realms_for_sim with 'universal' if needed
                if len(realms_for_sim) < len(frequencies_to_use):
                    realms_for_sim.extend(['universal'] * (len(frequencies_to_use) - len(realms_for_sim)))
                # Or truncate if too many
                realms_for_sim = realms_for_sim[:len(frequencies_to_use)]


            # Pass the realm_names list to the basic pattern generation
            simulation_results = run_basic_pattern_generation(
                frequencies_to_use, 
                realms_for_sim, # Pass the list of realm names here
                output_dir=self.output_dir, 
                config=self.config,
                resolution=resolution
            )
            
            analyzer = UBPPatternAnalyzer(config=self.config) # Initialize analyzer for basic patterns

            for i, sim_info in enumerate(simulation_results):
                pattern_array = sim_info['pattern_data']
                analysis_results = analyzer.analyze_coherence_pressure(pattern_array)
                
                # Construct pattern_metadata for the new standard
                unique_id = f"pattern_basic_{sim_info['frequency']:.2f}_{datetime.now().strftime('%f')}"
                realm_context = sim_info.get('realm_context', 'universal') # Get realm_context from sim_info
                description = f"UBP basic simulation pattern for frequency {sim_info['frequency']:.2f} Hz."
                
                pattern_details = {
                    "frequency": sim_info['frequency'],
                    "emergent_energy_htr": sim_info.get('final_emergent_energy', 'N/A'), # Renamed for clarity
                    "nrci_from_htr": sim_info.get('nrci_from_htr', 'N/A'),
                    "characteristic_length_scale_nm": sim_info.get('characteristic_length_scale_nm', 'N/A'),
                    "bitfield_dimensions": sim_info['bitfield_dimensions'],
                    "resolution": sim_info['resolution']
                }

                pattern_hash = self.store_pattern_data(
                    pattern_array=pattern_array,
                    analysis_results=analysis_results,
                    pattern_metadata={
                        "data_type": "ubp_pattern_basic_simulation",
                        "unique_id": unique_id,
                        "realm_context": realm_context,
                        "description": description,
                        "source_module": "ubp_pattern_generator_1.py",
                        "pattern_details": pattern_details
                    }
                )
                stored_patterns_info[f"basic_pattern_{i}"] = {"hash": pattern_hash, "analysis_summary": analysis_results.get('pattern_classification', 'N/A')}
        
        else:
            print(f"❌ Unknown pattern generation method: {pattern_generation_method}")
        
        print(f"Total patterns stored in HexDictionary: {len(self.hex_dict)}")
        return stored_patterns_info

    def search_patterns_by_metadata(self, search_criteria: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """
        Searches HexDictionary for patterns matching specific metadata criteria.
        Updated to handle the new nested metadata structure.
        
        Args:
            search_criteria: A dictionary where keys are metadata fields and values are desired matches.
                             Supports exact matches for most fields, and range checks for numeric scores.
                             Examples: {'analysis_results.coherence_score_min': 0.8, 'pattern_details.pattern_type': 'crv_harmonic_filtered'}
            limit: Maximum number of matching patterns to return.
            
        Returns:
            A list of dictionaries, each containing {'hash': str, 'metadata': Dict, 'pattern_preview_file': str}.
        """
        print(f"\n🔍 Searching HexDictionary for patterns matching: {search_criteria}...")
        matching_patterns = []
        
        # Flatten search criteria for easier comparison with stored metadata
        # e.g., {'analysis.coherence_score_min': 0.8}
        flattened_search_criteria = {}
        for k, v in search_criteria.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flattened_search_criteria[f"{k}.{sub_k}"] = sub_v
            else:
                flattened_search_criteria[k] = v

        for p_hash, entry_info in self.hex_dict.entries.items():
            metadata = entry_info.get('meta', {})
            
            match = True
            for criterion_key_flat, criterion_value in flattened_search_criteria.items():
                
                # Navigate nested metadata for comparison
                current_meta_val = metadata
                key_parts = criterion_key_flat.split('.')
                found_path = True
                for i, part in enumerate(key_parts):
                    if i == len(key_parts) - 1: # Last part of the key
                        if part.endswith('_min'):
                            actual_key = part[:-4]
                            if actual_key not in current_meta_val or not isinstance(current_meta_val[actual_key], (int, float)):
                                match = False
                                break
                            if current_meta_val[actual_key] < criterion_value:
                                match = False
                                break
                        elif part.endswith('_max'):
                            actual_key = part[:-4]
                            if actual_key not in current_meta_val or not isinstance(current_meta_val[actual_key], (int, float)):
                                match = False
                                break
                            if current_meta_val[actual_key] > criterion_value:
                                match = False
                                break
                        else: # Exact match for the final key part
                            if current_meta_val.get(part) != criterion_value:
                                match = False
                                break
                    else: # Not the last part, so traverse deeper
                        if part not in current_meta_val or not isinstance(current_meta_val[part], dict):
                            found_path = False
                            break
                        current_meta_val = current_meta_val[part]
                
                if not found_path or not match:
                    match = False
                    break
            
            if match:
                preview_file = os.path.join(self.output_dir, f"pattern_{p_hash[:8]}.png")
                matching_patterns.append({
                    "hash": p_hash,
                    "metadata": metadata,
                    "pattern_preview_file": preview_file # Placeholder path
                })
                if len(matching_patterns) >= limit:
                    break
        
        print(f"Found {len(matching_patterns)} matching patterns.")
        return matching_patterns

    def get_pattern_by_hash(self, pattern_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a pattern array and its metadata by hash.
        
        Args:
            pattern_hash: The HexDictionary hash of the pattern.
            
        Returns:
            A dictionary containing {'pattern_array': np.ndarray, 'metadata': Dict} or None.
        """
        pattern_array = self.hex_dict.retrieve(pattern_hash)
        metadata = self.hex_dict.get_metadata(pattern_hash)
        
        if pattern_array is not None and metadata is not None:
            return {"pattern_array": pattern_array, "metadata": metadata}
        return None
"""
Universal Binary Principle (UBP) Framework v3.2+ - Generate CRV Patterns and store
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any

from ubp_config import get_config, UBPConfig, RealmConfig
from hex_dictionary import HexDictionary
from ubp_pattern_integrator import UBPPatternIntegrator
from ubp_256_study_evolution import UBP256Evolution # Directly import UBP256Evolution


def generate_and_store_crv_patterns():
    """
    Generates cymatic patterns for each realm's main CRV and stores them
    in the HexDictionary using the UBPPatternIntegrator.
    This version uses the '256_study' method for comprehensive generation.
    It now checks for existing high-quality patterns to avoid redundant generation.
    """
    print("\n--- Starting CRV Pattern Generation and Storage (Comprehensive 256 Study) ---")

    config = get_config()
    integrator = UBPPatternIntegrator(config=config)
    study_evolution = UBP256Evolution(resolution=256, config=config)

    crv_keys = list(study_evolution.crv_constants.keys())
    removal_types = ['adaptive', 'fundamental', 'golden'] # Hardcoded in UBP256Evolution

    generated_count = 0
    skipped_count = 0
    stored_patterns_info = {} # To collect results for summary

    # Use config's coherence threshold as a benchmark for "high-quality" existing patterns
    COHERENCE_THRESHOLD_FOR_SKIP = config.performance.COHERENCE_THRESHOLD

    for crv_key in crv_keys:
        for removal_type in removal_types:
            # 1. Check if a high-quality pattern for this specific CRV/removal_type already exists in HexDictionary
            search_criteria = {
                "data_type": "ubp_pattern_256study",
                "pattern_details": {
                    "crv_key": crv_key,
                    "removal_type": removal_type
                },
                "analysis_results": {
                    "coherence_score_min": COHERENCE_THRESHOLD_FOR_SKIP # Only skip if existing pattern is high quality
                }
            }
            existing_patterns = integrator.search_patterns_by_metadata(search_criteria, limit=1)

            if existing_patterns:
                skipped_count += 1
                existing_hash = existing_patterns[0]['hash']
                existing_meta = existing_patterns[0]['metadata']
                # Safely get coherence and classification for printing
                existing_analysis_results = existing_meta.get('additional_metadata', {}).get('analysis_results', {})
                existing_coherence = existing_analysis_results.get('coherence_score', 'N/A')
                existing_classification = existing_analysis_results.get('pattern_classification', 'N/A')
                
                # Format coherence score for display
                formatted_existing_coherence = f"{float(existing_coherence):.3f}" if isinstance(existing_coherence, (float, int)) else str(existing_coherence)

                print(f"  Skipping generation for {crv_key}_{removal_type}: High-quality pattern already exists "
                      f"(Coherence: {formatted_existing_coherence}, Classification: {existing_classification}, Hash: {existing_hash[:8]}...).")
                stored_patterns_info[f"{crv_key}_{removal_type}"] = {
                    "hash": existing_hash, 
                    "analysis_summary": existing_classification # Store classification for summary
                }
                continue # Skip to next combination

            # 2. If no high-quality existing pattern found, generate a new one
            print(f"  Generating pattern for CRV: {crv_key}, Removal: {removal_type}...")
            base_pattern = study_evolution.generate_crv_pattern(crv_key)
            filtered_pattern = study_evolution.apply_subharmonic_removal(base_pattern, removal_type)
            analysis = study_evolution.analyze_coherence_geometry(filtered_pattern)

            # 3. Store the newly generated pattern
            unique_id = f"pattern_256study_{crv_key}_{removal_type}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            realm_context = 'universal' # Default for these patterns, or derive from crv_key if needed

            pattern_details = {
                "crv_key": crv_key,
                "removal_type": removal_type,
                "resolution": 256,
                "pattern_type": "crv_harmonic_filtered"
            }

            pattern_hash = integrator.store_pattern_data(
                pattern_array=filtered_pattern, # filtered_pattern is np.ndarray here, correctly handled by integrator
                analysis_results=analysis,
                pattern_metadata={
                    "data_type": "ubp_pattern_256study",
                    "unique_id": unique_id,
                    "realm_context": realm_context,
                    "description": f"UBP 256 study pattern for CRV {crv_key} with {removal_type} removal.",
                    "source_module": "ubp_256_study_evolution.py",
                    "pattern_details": pattern_details
                }
            )
            generated_count += 1
            stored_patterns_info[f"{crv_key}_{removal_type}"] = {"hash": pattern_hash, "analysis_summary": analysis['pattern_classification']}
            
            # Print a summary of the newly generated pattern
            coherence_score_val = analysis.get("coherence_score", "N/A")
            if isinstance(coherence_score_val, (float, int, np.floating, np.integer)):
                formatted_coherence = f"{float(coherence_score_val):.3f}"
            else:
                formatted_coherence = str(coherence_score_val)
            print(
                f"  -> Generated & Stored: {crv_key}_{removal_type}, "
                f"Coherence: {formatted_coherence}, "
                f"Classification: {analysis['pattern_classification']} (Hash: {pattern_hash[:8]}...)"
            )

    if generated_count > 0 or skipped_count > 0:
        print("\n--- CRV Pattern Generation and Storage Complete ---")
        print(f"Summary: Generated {generated_count} new patterns, Skipped {skipped_count} existing high-quality patterns.")
        print("Summary of all relevant CRV patterns:")
        for key, info in stored_patterns_info.items():
            pattern_hash = info["hash"]
            analysis_summary_str = info["analysis_summary"] # Classification string

            # Retrieve the full metadata to get the numeric coherence score
            retrieved_metadata = integrator.hex_dict.get_metadata(pattern_hash)
            # The structure for analysis_results is now nested under 'additional_metadata'
            analysis_results_from_meta = retrieved_metadata.get("additional_metadata", {}).get(
                "analysis_results", {}
            )
            coherence_score_val = analysis_results_from_meta.get("coherence_score", "N/A")

            # Format coherence safely
            if isinstance(coherence_score_val, (float, int, np.floating, np.integer)):
                formatted_coherence = f"{float(coherence_score_val):.3f}"
            else:
                formatted_coherence = str(coherence_score_val)

            print(
                f"  - Pattern Key: {key}, "
                f"Classification: {analysis_summary_str}, "
                f"Coherence: {formatted_coherence} (Hash: {pattern_hash[:8]}...)"
            )
    else:
        print("❌ No patterns were generated or found after checking existing entries.")
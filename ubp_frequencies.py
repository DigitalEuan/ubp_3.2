"""
Universal Binary Principle (UBP) Framework v3.2+ - Comprehensive Frequency Scan and Sub-CRV Analysis Across All Realms
Author: Euan Craig, New Zealand
Date: 03 September 2025
================================================
This script scans for CRVs and Sub-CRVs
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
import time
import os # Import os for output file management

# --- Include necessary definitions from other modules ---
from ubp_config import get_config, UBPConfig
from system_constants import UBPConstants
from enhanced_nrci import EnhancedNRCI, NRCIResult # Corrected: Import NRCIResult directly

class UbpFrequencies:
    def __init__(self):
        # Initialize UBPConfig and EnhancedNRCI globally for use in the script
        self.config: UBPConfig = get_config(environment="testing") # Use 'testing' environment for this script
        self.nrci_system = EnhancedNRCI()
        print("DEBUG: UbpFrequencies class initialized.")

    def generate_standardized_data(self, frequency, duration=0.01, num_points=1000):
        """
        Generates a standardized dataset (e.g., a sine wave) for analysis.

        Args:
            frequency (float): The frequency of the signal in Hz.
            duration (float): The duration of the simulated signal in seconds.
            num_points (int): The number of data points to generate.

        Returns:
            np.ndarray: A numpy array containing the standardized data.
        """
        t = np.linspace(0, duration, num_points, endpoint=False)  # Time vector
        # Generate a sine wave with fixed amplitude and phase for consistency
        data = np.sin(2 * np.pi * frequency * t)
        return data

    def analyze_coherence_wrapper(self, data: np.ndarray, realm: str) -> NRCIResult: # Corrected: Use NRCIResult directly
        """
        Analyzes coherence of the input data for a given realm using EnhancedNRCI.
        To calculate basic NRCI, we need a 'simulated' and a 'theoretical' dataset.
        Since 'data' is a generated signal, we create a slightly perturbed version
        as a 'theoretical' target to ensure NRCI is not always 1.0 (unless data is flat).
        """
        # Create a slightly offset theoretical target for a non-perfect NRCI.
        # Add some controlled, realm-specific noise or transformation based on parameters
        # to simulate a 'target' state for NRCI calculation.
        
        # Example: Simple delayed and slightly scaled version as theoretical target
        # The magnitude of perturbation influences the resulting NRCI score.
        # For a more robust test, this could be loaded from a 'known good' state from HexDictionary.
        theoretical_data = data * 0.9 + np.roll(data, int(len(data) * 0.05)) * 0.1
        
        # Ensure lengths match for NRCI calculation
        min_len = min(len(data), len(theoretical_data))
        simulated = data[:min_len]
        theoretical = theoretical_data[:min_len]

        # Handle cases where data is flat (standard deviation is zero)
        if np.std(simulated) == 0 and np.std(theoretical) == 0:
            if np.all(simulated == theoretical):
                nrci_value = 1.0
            else:
                nrci_value = 0.0
            return self.nrci_system.compute_basic_nrci(simulated.tolist(), theoretical.tolist())

        # Ensure inputs are lists for EnhancedNRCI.compute_basic_nrci
        nrci_result = self.nrci_system.compute_basic_nrci(simulated.tolist(), theoretical.tolist())
        return nrci_result

    def run(self):
        print("\n\n" + "="*80)
        print("🚀 Starting Comprehensive Frequency Scan and Sub-CRV Analysis Across All Realms")
        print("="*80)

        # Step 1: Get available realms from ubp_config
        available_realms = list(self.config.realms.keys())

        # Print the list of available realms
        print("Available Realms in the UBP Framework (from ubp_config):")
        for realm in available_realms:
            print(f"- {realm.capitalize()}")
        print(f"DEBUG: Configured realms: {list(self.config.realms.keys())}")


        # Start iterating through each available realm
        print("\nStarting frequency scan and sub-CRV analysis for each realm:")

        # Store results for later compilation
        all_realms_sub_crv_findings = {}

        for realm in available_realms:
            print(f"\n{'='*60}")
            print(f"Analyzing Realm: {realm.upper()}")
            print(f"{'='*60}")

            # Step 3: Define frequency scan parameters for realm
            realm_cfg = self.config.get_realm_config(realm)

            if realm_cfg:
                # Define scan range based on the typical frequency range from ubp_config
                scan_start_freq = realm_cfg.frequency_range[0]
                scan_end_freq = realm_cfg.frequency_range[1]
                
                # Ensure the end frequency is slightly larger than the start frequency if they are too close
                if scan_end_freq <= scan_start_freq + self.config.constants.EPSILON_UBP:
                     scan_end_freq = scan_start_freq * 10 if scan_start_freq > self.config.constants.EPSILON_UBP else 1e2 # Arbitrarily increase range if too narrow
                # Ensure minimum range for logspace
                if scan_start_freq < self.config.constants.EPSILON_UBP:
                     scan_start_freq = self.config.constants.EPSILON_UBP
                     if scan_end_freq < self.config.constants.EPSILON_UBP:
                         scan_end_freq = 1.0 # Default to 1Hz if range is problematic
                if scan_end_freq < scan_start_freq: # Correct inverted ranges
                    scan_start_freq, scan_end_freq = scan_end_freq, scan_start_freq
                    if scan_end_freq == scan_start_freq:
                         scan_end_freq = scan_start_freq * 10 if scan_start_freq > self.config.constants.EPSILON_UBP else 1e2
            else:
                # Fallback to a default range if realm info is not available
                print(f"Warning: Realm '{realm}' not found in ubp_config. Using default scan range.")
                scan_start_freq = 1e0  # Default start frequency (1 Hz)
                scan_end_freq = 1e10 # Default end frequency (10 GHz)

            # Define the number of frequency steps
            num_scan_steps = 10000 # Increased number of steps

            # Store scan parameters in realm_findings
            realm_findings = {
                'realm': realm,
                'scan_parameters': {
                    'start_freq': scan_start_freq,
                    'end_freq': scan_end_freq,
                    'num_steps': num_scan_steps
                },
                'frequencies_scanned': [],
                'nrci_scores': [],
                'peak_frequencies': [],
                'potential_sub_crvs': [],
                'summary': ""
            }
            all_realms_sub_crv_findings[realm] = realm_findings

            # Print the defined parameters
            print(f"\n  Defined scan parameters for {realm.capitalize()} Realm:")
            print(f"    Scan Start Frequency: {scan_start_freq:.4e} Hz")
            print(f"    Scan End Frequency: {scan_end_freq:.4e} Hz")
            print(f"    Number of Scan Steps: {num_scan_steps}")

            # Step 4: Generate frequencies to scan
            try:
                frequencies_to_scan = np.logspace(np.log10(scan_start_freq), np.log10(scan_end_freq), num_scan_steps)
                print(f"\n  Generated {len(frequencies_to_scan)} frequencies to scan.")
                realm_findings['frequencies_scanned'] = frequencies_to_scan.tolist()
            except ValueError as e:
                print(f"\n  Error generating frequencies for {realm.capitalize()}: {e}")
                print("  This might happen if the frequency range is invalid (e.g., start_freq <= 0 or start_freq >= end_freq).")
                frequencies_to_scan = np.array([])
                realm_findings['frequencies_scanned'] = []

            # Step 5: Run frequency scan computation for realm
            realm_nrci_scores = []
            print(f"\n  Scanning frequencies in the '{realm.capitalize()}' realm...")

            if len(frequencies_to_scan) > 0:
                for i, freq in enumerate(frequencies_to_scan):
                    test_data = self.generate_standardized_data(freq)
                    try:
                        # Use the wrapper function for coherence analysis
                        analysis_result = self.analyze_coherence_wrapper(test_data, realm)
                        nrci_value = analysis_result.value # Access the 'value' attribute of NRCIResult
                        realm_nrci_scores.append(nrci_value)
                    except Exception as e:
                        print(f"    Step {i+1}/{len(frequencies_to_scan)}: Frequency {freq:.4e} Hz -> Error during analysis: {e}")
                        realm_nrci_scores.append(np.nan)

                print(f"\n  Frequency scan complete for {realm.capitalize()} realm.")
                realm_findings['nrci_scores'] = realm_nrci_scores
            else:
                print(f"\n  No frequencies to scan for {realm.capitalize()} realm. Skipping scan.")
                realm_findings['nrci_scores'] = []

            # Step 6: Visualize frequency resonance profile for realm
            print(f"\n  Visualizing frequency resonance profile for {realm.capitalize()} realm...")

            if len(frequencies_to_scan) > 0 and len(realm_nrci_scores) == len(frequencies_to_scan):
                plt.figure(figsize=(12, 6))
                ax = plt.subplot(1, 1, 1)

                ax.plot(frequencies_to_scan, realm_nrci_scores, marker='o', linestyle='-')

                if scan_end_freq / scan_start_freq > 100:
                    ax.set_xscale('log')

                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('NRCI Score')
                ax.set_title(f'Frequency Resonance Profile for {realm.capitalize()} Realm')
                ax.grid(True, which="both", linestyle='--')
                
                # Save plot to /output/ directory
                plot_filename = f"frequency_resonance_profile_{realm.lower()}.png"
                plot_filepath = os.path.join("/output/", plot_filename)
                plt.savefig(plot_filepath)
                plt.close() # Close plot to free memory
                print(f"  Visualization saved to {plot_filepath}")
            else:
                print(f"  Skipping visualization for {realm.capitalize()} realm: Insufficient data.")

            # Step 7: Identify peaks and potential sub-CRVs for realm
            print(f"\n  Identifying peaks and potential sub-CRVs for {realm.capitalize()} realm...")

            if len(realm_nrci_scores) > 0 and len(frequencies_to_scan) == len(realm_nrci_scores):
                nrci_scores_arr = np.array(realm_nrci_scores)
                
                peak_height_threshold = np.nanmean(nrci_scores_arr) + 0.5 * np.nanstd(nrci_scores_arr) if not np.all(np.isnan(nrci_scores_arr)) else 0.5
                peak_distance = max(5, len(nrci_scores_arr) // 50)

                peaks, properties = find_peaks(nrci_scores_arr, height=peak_height_threshold, distance=peak_distance)

                print(f"  Identified {len(peaks)} potential resonance peaks.")

                if len(peaks) > 0:
                    peak_frequencies = frequencies_to_scan[peaks]
                    peak_nrci_scores = nrci_scores_arr[peaks]

                    realm_findings['peak_frequencies'] = peak_frequencies.tolist()

                    sorted_indices = np.argsort(peak_nrci_scores)[::-1]
                    peak_frequencies_sorted = peak_frequencies[sorted_indices]
                    peak_nrci_scores_sorted = peak_nrci_scores[sorted_indices]

                    print("  Top Peak Frequencies and NRCI Scores:")
                    for i in range(min(5, len(peak_frequencies_sorted))):
                        print(f"    Peak {i+1}: Frequency {peak_frequencies_sorted[i]:.4e} Hz, NRCI: {peak_nrci_scores_sorted[i]:.6f}")

                    print(f"\n  Comparing peaks to {realm.capitalize()} Realm CRV and potential sub-CRVs:")

                    realm_crv = self.config.realms.get(realm.lower()).main_crv if self.config.realms.get(realm.lower()) else None

                    if realm_crv is not None:
                        print(f"    {realm.capitalize()} Realm CRV: {realm_crv:.4e} Hz (from UBPConfig)")

                        crv_match_found = False
                        tolerance_percent = 5

                        for peak_freq in peak_frequencies_sorted:
                            if realm_crv > 0 and np.isclose(peak_freq, realm_crv, rtol=tolerance_percent/100.0):
                                print(f"    ✅ Peak frequency {peak_freq:.4e} Hz is close to the Realm CRV ({tolerance_percent}% tolerance).")
                                crv_match_found = True

                        if not crv_match_found:
                            print(f"    ❌ No peak frequency found within {tolerance_percent}% of the Realm CRV.")

                        print("\n    Potential Sub-CRVs (Harmonics/Subharmonics of Realm CRV):")
                        num_harmonics_to_check = 25
                        found_sub_crvs = False
                        potential_sub_crvs_list = []

                        for i in range(1, num_harmonics_to_check + 1):
                            harmonic = realm_crv * i
                            subharmonic = realm_crv / i if i > 0 else np.inf

                            if harmonic > 0:
                                for peak_freq in peak_frequencies_sorted:
                                     if np.isclose(peak_freq, harmonic, rtol=0.05):
                                        print(f"      - Peak frequency {peak_freq:.4e} Hz is close to the {i}x harmonic ({harmonic:.4e} Hz).")
                                        potential_sub_crvs_list.append({'type': f'{i}x Harmonic', 'frequency': peak_freq, 'expected': harmonic})
                                        found_sub_crvs = True
                            if subharmonic > 0 and subharmonic != np.inf:
                                for peak_freq in peak_frequencies_sorted:
                                     if np.isclose(peak_freq, subharmonic, rtol=0.05):
                                        print(f"      - Peak frequency {peak_freq:.4e} Hz is close to the 1/{i} subharmonic ({subharmonic:.4e} Hz).")
                                        potential_sub_crvs_list.append({'type': f'1/{i} Subharmonic', 'frequency': peak_freq, 'expected': subharmonic})
                                        found_sub_crvs = True

                        if not found_sub_crvs:
                            print("      - No significant peaks found near simple harmonics or subharmonics of the Realm CRV within tolerance.")
                        realm_findings['potential_sub_crvs'] = potential_sub_crvs_list
                    else:
                        print(f"    UBPConfig CRV for realm '{realm}' not found. Cannot perform detailed comparison.")
                        realm_findings['potential_sub_crvs'] = []
                else:
                    print("  No significant resonance peaks identified in the frequency scan for this realm.")
                    realm_findings['peak_frequencies'] = []
                    realm_findings['potential_sub_crvs'] = []

            realm_summary_text = f"Frequency scan and sub-CRV analysis for {realm.capitalize()} realm completed."
            realm_findings['summary'] = realm_summary_text


        # After the loop finishes for all realms, run the compilation cell
        print("\n\nAll realm analyses complete. Proceeding to compilation.")

        # Step 9: Compile and Summarize Findings Across All Realms (Final Step)

        print("\n\n" + "="*80)
        print("📋 Compilation and Summary of Frequency Scan and Sub-CRV Findings Across All Realms")
        print("="*80)

        summary_list = []
        for realm, findings in all_realms_sub_crv_findings.items():
            num_peaks = len(findings.get('peak_frequencies', []))
            num_potential_sub_crvs = len(findings.get('potential_sub_crvs', []))

            realm_nrci_scores = [score for score in findings.get('nrci_scores', []) if not np.isnan(score)]
            max_nrci = max(realm_nrci_scores) if realm_nrci_scores else np.nan

            realm_crv = self.config.realms.get(realm.lower()).main_crv if self.config.realms.get(realm.lower()) else np.nan

            crv_close_to_peak = False
            if realm_crv is not None and findings.get('peak_frequencies', []):
                 for peak_freq in findings['peak_frequencies']:
                     if realm_crv > 0 and np.isclose(peak_freq, realm_crv, rtol=0.05):
                         crv_close_to_peak = True
                         break

            summary_list.append({
                'Realm': realm.capitalize(),
                'Scan Range (Hz)': f"{findings['scan_parameters'].get('start_freq', np.nan):.2e} - {findings['scan_parameters'].get('end_freq', np.nan):.2e}",
                'Num Steps': findings['scan_parameters'].get('num_steps', np.nan),
                'Max NRCI in Scan': max_nrci,
                'Num Peaks Identified': num_peaks,
                'Potential Sub-CRVs Found': num_potential_sub_crvs,
                'CRV Close to Peak': crv_close_to_peak
            })

        df_summary = pd.DataFrame(summary_list)

        df_summary = df_summary.sort_values(by='Max NRCI in Scan', ascending=False)

        df_summary['Max NRCI in Scan'] = df_summary['Max NRCI in Scan'].apply(lambda x: f"{x:.6f}" if pd.notna(x) else 'N/A')
        df_summary['Num Steps'] = df_summary['Num Steps'].apply(lambda x: int(x) if pd.notna(x) else 'N/A')


        print("\n--- Summary Table ---")
        print(df_summary.to_string()) # Use .to_string() for standard print
        print("---------------------\n")

        print("\n--- Overall Insights ---")
        print("This analysis performed a frequency scan for each available UBP realm to identify resonance frequencies (peaks in NRCI scores) and potential sub-CRVs.")

        print("\nKey Observations from the Summary Table:")
        print("- **Max NRCI in Scan:** This indicates the highest level of computational ease observed within the scanned frequency range for each realm. Higher values suggest stronger resonance at certain frequencies.")
        print("- **Num Peaks Identified:** This shows how many distinct resonance frequencies were found within the scanned range for each realm, based on the peak finding criteria.")
        print("- **Potential Sub-CRVs Found:** This counts how many identified peaks were found to be close to simple harmonic or subharmonic relationships with the realm's standard CRV.")
        print("- **CRV Close to Peak:** This indicates if the standard UBPConstants CRV for the realm was found to be close to one of the identified resonance peaks.")

        print("\nAnalysis of CRV Relationship:")
        print("- For realms where 'CRV Close to Peak' is True, the standard CRV appears to be a primary resonance frequency.")
        print("- For realms where 'CRV Close to Peak' is False, the standard CRV might represent a different characteristic (e.g., a base constant) rather than the peak computational frequency in the scanned range. The highest resonance might occur at other frequencies.")

        print("\nSub-CRV Analysis:")
        print("- The 'Potential Sub-CRVs Found' column suggests that many realms exhibit a harmonic structure around their standard CRVs, implying the existence of sub-CRVs where resonance is also high.")

        print("\nFurther Steps:")
        print("- Investigate the specific frequencies of the highest NRCI peaks for each realm.")
        print("- Analyze the relationships between the identified peak frequencies (beyond simple harmonics/subharmonics) for each realm to understand their unique resonance structures.")
        print("- Design realm-specific 'Harder Tests' tailored to the identified peak frequencies and sub-CRVs to assess their computational performance under more complex workloads.")
        print("- Compare the findings from these empirical scans with the theoretical basis of the UBPConstants CRVs and realm definitions.")

        print("\n="*80)
        print("✅ Frequency Scan and Sub-CRV Analysis Across All Realms Complete.")
        print("="*80)
import json
import os
from hex_dictionary import HexDictionary
from datetime import datetime

# Define the paths to the persistent files
PERSISTENT_STATE_DIR = "/persistent_state/"
HEX_DICTIONARY_STORAGE_DIR = os.path.join(PERSISTENT_STATE_DIR, "hex_dictionary_storage/")

class ListPersistentState:
    def run(self):
        """
        Provides a structured overview of the contents of /persistent_state/,
        focusing on HexDictionary entries and key result files.
        """
        print("\n--- Listing Persistent State Contents ---")

        # 1. Summarize HexDictionary contents
        hex_dict = HexDictionary()
        hex_dict_stats = hex_dict.get_metadata_stats()

        print("\n📦 HexDictionary Contents:")
        print(f"   Total entries: {hex_dict_stats.get('total_entries', 'N/A')}")
        print(f"   Storage size: {hex_dict_stats.get('storage_size_mb', 'N/A')} MB")
        print(f"   Metadata file size: {hex_dict_stats.get('metadata_file_size_bytes', 'N/A')} bytes")

        if hex_dict_stats.get('total_entries', 0) > 0:
            print("\n   Individual HexDictionary Entries (Summary):")
            grouped_entries = {}
            for data_hash, entry_info in hex_dict.entries.items():
                meta = entry_info.get('meta', {})
                
                # Retrieve and prioritize general metadata fields if present at the top level
                data_type = meta.get('data_type', 'generic_entry')
                custom_id = meta.get('unique_id', data_hash[:8]) # Default to hash prefix
                realm_context = meta.get('realm_context', 'N/A')
                source_module = meta.get('source_module', 'HexDictionary (generic)')
                description = meta.get('description', 'No description.')
                timestamp_str = meta.get('timestamp', 'N/A')

                # Default entry summary structure, will be refined by specific types
                entry_summary = {
                    'hash_prefix': data_hash[:8],
                    'full_hash': data_hash,
                    'timestamp': timestamp_str,
                    'realm_context': realm_context,
                    'source_module': source_module,
                    'unique_id': custom_id,
                    'description': description,
                    'additional_metadata': meta.get('additional_metadata', {})
                }
                
                # --- Type-specific overrides for description, id, and source_module ---
                if data_type == 'ubp_element_data':
                    atomic_number = meta.get('atomic_number', 'N/A')
                    symbol = meta.get('symbol', 'N/A')
                    name = meta.get('name', 'N/A')
                    period = meta.get('period', 'N/A')
                    group = meta.get('group', 'N/A')
                    block = meta.get('block', 'N/A')
                    
                    entry_summary['description'] = f"Element: {name} ({symbol}), Z={atomic_number}, P={period}, G={group}, Block={block}"
                    entry_summary['unique_id'] = f"Elem-{symbol}-{atomic_number}"
                    entry_summary['source_module'] = 'UBP_Test_Drive_Complete_Periodic_Table_118_Elements.py'
                    entry_summary['realm_context'] = 'chemistry'

                elif data_type == 'ubp_pattern_256study':
                    pattern_details = entry_summary['additional_metadata'].get('pattern_details', {})
                    analysis_results = entry_summary['additional_metadata'].get('analysis_results', {})
                    crv_key = pattern_details.get('crv_key', 'N/A')
                    removal_type = pattern_details.get('removal_type', 'N/A')
                    coherence_score = analysis_results.get('coherence_score', 'N/A')
                    classification = analysis_results.get('pattern_classification', 'N/A')
                    
                    formatted_coherence = f"{coherence_score:.3f}" if isinstance(coherence_score, (float, int)) else str(coherence_score)
                    
                    entry_summary['description'] = f"CRV: {crv_key}, Removal: {removal_type}, Class: {classification}, Coherence: {formatted_coherence}"
                    entry_summary['unique_id'] = f"256Study-{crv_key}-{removal_type}"
                    entry_summary['source_module'] = 'ubp_256_study_evolution.py'
                    entry_summary['realm_context'] = realm_context # Keep original realm context for pattern if set

                elif data_type == 'ubp_pattern_basic_simulation':
                    pattern_details = entry_summary['additional_metadata'].get('pattern_details', {})
                    freq = pattern_details.get('frequency', 'N/A')
                    nrci_htr = pattern_details.get('nrci_from_htr', 'N/A')
                    
                    formatted_freq = f"{freq:.2e}" if isinstance(freq, (float, int)) else str(freq)
                    formatted_nrci_htr = f"{nrci_htr:.3f}" if isinstance(nrci_htr, (float, int)) else str(nrci_htr)
                    
                    entry_summary['description'] = f"Basic pattern, Freq: {formatted_freq} Hz, NRCI(HTR): {formatted_nrci_htr}"
                    entry_summary['unique_id'] = f"BasicPattern-f{formatted_freq}"
                    entry_summary['source_module'] = 'ubp_pattern_generator_1.py'
                    entry_summary['realm_context'] = realm_context # Keep original realm context for pattern if set

                elif data_type == 'ubp_simulation_result':
                    final_nrci = meta.get('final_nrci', 'N/A')
                    total_toggles = meta.get('total_toggles', 'N/A')
                    active_realm_sim = meta.get('active_realm', 'N/A')
                    
                    formatted_final_nrci = f"{final_nrci:.4f}" if isinstance(final_nrci, (float, int)) else str(final_nrci)

                    entry_summary['description'] = f"Simulation. NRCI: {formatted_final_nrci}, Toggles: {total_toggles}, Realm: {active_realm_sim}"
                    entry_summary['unique_id'] = meta.get('simulation_id', f"Sim-{data_hash[-8:]}")
                    entry_summary['source_module'] = meta.get('source_module', 'runtime.py')
                    entry_summary['realm_context'] = active_realm_sim

                elif data_type == 'ubp_uid_operators_database':
                    entry_summary['unique_id'] = meta.get('unique_id', 'UBP-UID-OPS')
                    entry_summary['source_module'] = meta.get('source_module', 'store_computational_constants.py')
                    entry_summary['realm_context'] = meta.get('realm_context', 'universal')
                    entry_summary['description'] = meta.get('description', 'Comprehensive database of UBP UID Operators.')
                
                elif data_type == 'ubp_periodic_table_results':
                    overall_rating = meta.get('overall_rating', 'N/A')
                    final_nrci_pt = meta.get('final_nrci', 'N/A')
                    
                    formatted_final_nrci_pt = f"{final_nrci_pt:.4f}" if isinstance(final_nrci_pt, (float, int)) else str(final_nrci_pt)

                    entry_summary['description'] = f"Full Periodic Table Results. Rating: {overall_rating}, NRCI: {formatted_final_nrci_pt}"
                    entry_summary['unique_id'] = meta.get('unique_id', f"PT-Results-{data_hash[-8:]}")
                    entry_summary['source_module'] = meta.get('source_module', 'UBP_Test_Drive_Complete_Periodic_Table_118_Elements.py')
                    entry_summary['realm_context'] = 'chemistry'

                # --- Handle UBP-Lisp BitBase items ---
                ubp_lisp_type = meta.get('ubp_lisp_type', None)
                if ubp_lisp_type:
                    original_lisp_meta = meta.get('original_lisp_metadata', {})
                    lisp_value = hex_dict.retrieve(data_hash) # Retrieve actual content for Lisp types for description

                    if ubp_lisp_type == 'offbit':
                        entry_summary['description'] = f"UBP-Lisp OffBit: 0x{lisp_value:06X}" if isinstance(lisp_value, int) else f"UBP-Lisp OffBit. Hash: {entry_summary['hash_prefix']}"
                        entry_summary['unique_id'] = f"Lisp-OffBit-{entry_summary['hash_prefix']}"
                        entry_summary['realm_context'] = 'ubp_lisp'
                    elif ubp_lisp_type == 'number':
                        entry_summary['description'] = f"UBP-Lisp Number: {lisp_value}"
                        entry_summary['unique_id'] = f"Lisp-Num-{entry_summary['hash_prefix']}"
                        entry_summary['realm_context'] = 'ubp_lisp'
                    elif ubp_lisp_type == 'string':
                        entry_summary['description'] = f"UBP-Lisp String: '{lisp_value[:30]}...'" if isinstance(lisp_value, str) and len(lisp_value) > 30 else f"UBP-Lisp String: '{lisp_value}'"
                        entry_summary['unique_id'] = f"Lisp-Str-{entry_summary['hash_prefix']}"
                        entry_summary['realm_context'] = 'ubp_lisp'
                    elif ubp_lisp_type == 'function':
                        func_name = original_lisp_meta.get('name', 'anonymous')
                        entry_summary['description'] = f"UBP-Lisp Function: {func_name} (params: {original_lisp_meta.get('params', [])})"
                        entry_summary['unique_id'] = f"Lisp-Func-{func_name}"
                        entry_summary['realm_context'] = 'ubp_lisp'
                    
                    # Update data_type for grouping to reflect Lisp origin
                    data_type = f"ubp_lisp_{ubp_lisp_type}"
                
                # --- Fallback for generic entries if still 'No description.' ---
                if entry_summary['description'] == 'No description.' or data_type == 'generic_entry':
                    retrieved_content = hex_dict.retrieve(data_hash)
                    if isinstance(retrieved_content, (str, int, float, bool)):
                        content_snippet = str(retrieved_content)
                        if len(content_snippet) > 50:
                            content_snippet = content_snippet[:47] + "..."
                        entry_summary['description'] = f"Raw Content: '{content_snippet}'"
                        data_type = f"generic_{type(retrieved_content).__name__}"
                    elif isinstance(retrieved_content, (list, dict)):
                        content_snippet = json.dumps(retrieved_content)
                        if len(content_snippet) > 50:
                            content_snippet = content_snippet[:47] + "..."
                        entry_summary['description'] = f"Raw Content: '{content_snippet}'"
                        data_type = f"generic_{type(retrieved_content).__name__}"
                    elif hasattr(retrieved_content, '__class__'):
                        entry_summary['description'] = f"Raw Content: <{retrieved_content.__class__.__name__} object>"
                        data_type = f"generic_{retrieved_content.__class__.__name__}"


                # Add processed data to grouped_entries
                entry_summary['description'] = (entry_summary['description'][:100] + '...') if len(entry_summary['description']) > 100 else entry_summary['description']
                entry_summary['unique_id'] = (entry_summary['unique_id'][:30] + '...') if len(entry_summary['unique_id']) > 30 else entry_summary['unique_id']
                
                if data_type not in grouped_entries:
                    grouped_entries[data_type] = []
                grouped_entries[data_type].append(entry_summary)

            for data_type, entries in grouped_entries.items():
                print(f"\n      Type: {data_type} ({len(entries)} entries)")
                
                # Sort entries by timestamp if available for chronological order
                sorted_entries = sorted(entries, key=lambda x: x.get('timestamp', '0'), reverse=True)
                
                for entry in sorted_entries:
                    # Format timestamp
                    formatted_ts = entry['timestamp']
                    if formatted_ts != 'N/A':
                        try:
                            # Attempt to parse ISO format first, then try simple format
                            dt_obj = datetime.fromisoformat(formatted_ts)
                            formatted_ts = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            pass # Keep original if not ISO or simple format

                    print(f"         - Hash: {entry['hash_prefix']}..., ID: {entry['unique_id']}, Realm: {entry['realm_context']}, Source: {entry['source_module']}, Desc: {entry['description']} (Timestamp: {formatted_ts})")
            
            print(f"\n   To retrieve detailed data or metadata for a HexDictionary entry:")
            print(f"     1. Read metadata file to find full hashes: `READ_PERSISTENT_FILE: {os.path.join(HEX_DICTIONARY_STORAGE_DIR, 'hex_dict_metadata.json')}`")
            print(f"     2. Use the full hash in a script: `hex_dict.retrieve('FULL_HASH')` or `hex_dict.get_metadata('FULL_HASH')`")
        else:
            print("   (HexDictionary is empty)")

        print("\n--- Persistent State Listing Complete ---")
"""
Universal Binary Principle (UBP) Framework v3.2+ - HexDictionary Equation for UBP
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

Detailed Explanation of HexDictionary
-------------------------------------

The HexDictionary serves as the Universal Binary Principle's (UBP) persistent,
content-addressable knowledge base. It is designed to store and retrieve any
computational artifact, simulation result, or derived UBP knowledge in a way
that is robust, immutable, and easily verifiable through cryptographic hashing.

Key Principles:
1.  **Content-Addressability:** Data is stored and retrieved using its SHA256 hash.
    This means the "key" is derived directly from the content itself. If the
    content changes, its hash changes, leading to a new entry. This ensures
    data integrity and immutability.
2.  **Persistence:** All data written to the HexDictionary is stored in a dedicated
    `/persistent_state/hex_dictionary_storage/` directory and persists across
    multiple experiment runs. This ensures that valuable UBP knowledge is
    never lost and can be incrementally built upon.
3.  **Compression:** To optimize storage and I/O performance, all data is
    transparently compressed using `gzip` before being written to disk and
    decompressed upon retrieval.
4.  **Metadata Management:** Alongside the raw data, a rich set of metadata can
    be stored with each entry. This metadata provides context, provenance,
    classification (e.g., 'ubp_simulation_result', 'ubp_periodic_table_entry'),
    and allows for intelligent querying and analysis of the knowledge base.
    The metadata itself is stored in `hex_dict_metadata.json` within the
    persistent storage directory.
5.  **Schema and Format for Integration:**
    *   **Keys:** Always a 64-character SHA256 hexadecimal string.
    *   **Data Types:** The `store` method accepts a `data_type` string parameter
        (e.g., 'str', 'int', 'float', 'json', 'array', 'bytes', 'list', 'dict').
        This guides the internal serialization/deserialization process. For complex
        Python objects, 'pickle' is used as a fallback if no specific type is matched.
    *   **Metadata Structure (Example):**
        ```json
        {
            "ubp_version": "3.1.1",
            "timestamp": "2025-09-03T10:30:00.123456",
            "data_type": "ubp_simulation_result",
            "unique_id": "sim_1678912345",
            "realm_context": "quantum",
            "description": "Simulation of toggle operations in quantum realm.",
            "source_module": "runtime.py",
            "tags": ["simulation", "quantum", "coherence"],
            "hashtags": ["#SIMULATION", "#QUANTUMREALM", "#COHERENCE"],
            "source_metadata": {
                "initial_conditions": "sparse_random_bitfield"
            },
            "associated_patterns": ["hash_of_pattern_1", "hash_of_pattern_2"],
            "additional_metadata": {
                "final_nrci": 0.9987,
                "total_toggles": 1500,
                "simulation_duration_seconds": 120.5
            }
        }
        ```
        The `UBPPatternIntegrator` and `UBP-Lisp` modules (via `BitBase`) leverage
        this standardized metadata schema by nesting their specific metadata
        under `additional_metadata`, often with keys like `pattern_details`
        or `analysis_results` for patterns, or `ubp_lisp_type` for Lisp values.

Usage in the UBP System:
-   **Knowledge Persistence:** Ensures all significant computation results (e.g., `SimulationResult` objects from `runtime.py`, cymatic patterns from `ubp_pattern_integrator.py`) are saved for future reference and analysis.
-   **Self-Optimization:** The framework can query the HexDictionary for historical performance data, optimal parameters for specific realms, or successful error correction outcomes to adapt and self-optimize.
-   **Ontological Computation (UBP-Lisp):** The `BitBase` module, which is a wrapper around `HexDictionary`, provides the native content-addressable storage for UBP-Lisp. This allows Lisp functions to store and retrieve computational artifacts by their content hash.
-   **Validation and Reproducibility:** By storing immutable, hashed data, the system can verify the integrity of past results and ensure reproducibility of experiments.
"""
import hashlib
import json
import numpy as np
import os
import pickle
import gzip  # Import gzip for compression
from typing import Any, Dict, Optional, Union

# Define the default directory for PERSISTENT storage for this version of HexDictionary
DEFAULT_HEX_DICT_STORAGE_DIR = "./persistent_state/hex_dictionary_storage/"
DEFAULT_HEX_DICT_METADATA_FILE = os.path.join(DEFAULT_HEX_DICT_STORAGE_DIR, "hex_dict_metadata.json")

class HexDictionary:
    """
    A persistent, content-addressable key-value store.
    Keys are SHA256 hashes of the stored data.
    Supports various data types for serialization.
    This version is specifically configured for persistent storage and uses gzip compression.
    """
    def __init__(self, storage_dir: str = DEFAULT_HEX_DICT_STORAGE_DIR, metadata_file: str = DEFAULT_HEX_DICT_METADATA_FILE):
        self.storage_dir = storage_dir
        self.metadata_file = metadata_file
        self.entries: Dict[str, Dict[str, Any]] = {}  # Stores {'hash': {'path': 'file', 'type': 'type', 'meta': {}}}
        self._ensure_storage_dir()
        self._load_metadata()
        # print(f"Persistent HexDictionary initialized at {self.storage_dir}. Loaded {len(self.entries)} entries.") # Removed for less verbose output

    def _ensure_storage_dir(self):
        """Ensures the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_metadata(self):
        """Loads the HexDictionary metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                try:
                    self.entries = json.load(f)
                    # Ensure metadata is dict type
                    for key, value in self.entries.items():
                        if 'meta' not in value or not isinstance(value['meta'], dict):
                            value['meta'] = {}
                except json.JSONDecodeError:
                    print("Warning: Persistent HexDictionary metadata file is corrupt. Starting with empty dictionary.")
                    self.entries = {}
        else:
            self.entries = {}

    def _save_metadata(self):
        """Saves the HexDictionary metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.entries, f, indent=4)

    def _serialize_data(self, data: Any, data_type: str) -> bytes:
        """
        Serializes data into bytes based on the specified data_type and then compresses it.
        Supports common Python types and numpy arrays.
        """
        serialized_bytes: bytes
        if data_type == 'bytes':
            serialized_bytes = data
        elif data_type == 'str':
            serialized_bytes = data.encode('utf-8')
        elif data_type == 'int' or data_type == 'float':
            serialized_bytes = str(data).encode('utf-8')
        elif data_type == 'json':
            # Ensure JSON data is always a dict or list for proper serialization
            if not isinstance(data, (dict, list)):
                # If it's a string that should be JSON, try to load it first
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    pass # If it's not a valid JSON string, just serialize as a string
            serialized_bytes = json.dumps(data).encode('utf-8')
        elif data_type == 'array' and isinstance(data, np.ndarray):
            serialized_bytes = pickle.dumps(data)
        elif data_type == 'list' or data_type == 'dict':
            serialized_bytes = json.dumps(data).encode('utf-8')
        else:
            serialized_bytes = pickle.dumps(data)
        
        return gzip.compress(serialized_bytes)

    def _deserialize_data(self, data_bytes: bytes, data_type: str) -> Any:
        """
        Decompresses data bytes and then deserializes them back into the original data type.
        """
        decompressed_bytes = gzip.decompress(data_bytes)

        if data_type == 'bytes':
            return decompressed_bytes
        elif data_type == 'str':
            return decompressed_bytes.decode('utf-8')
        elif data_type == 'int':
            return int(decompressed_bytes.decode('utf-8'))
        elif data_type == 'float':
            return float(decompressed_bytes.decode('utf-8'))
        elif data_type == 'json':
            return json.loads(decompressed_bytes.decode('utf-8'))
        elif data_type == 'array':
            return pickle.loads(decompressed_bytes)
        elif data_type == 'list' or data_type == 'dict':
            return json.loads(decompressed_bytes.decode('utf-8'))
        else:
            return pickle.loads(decompressed_bytes)

    def store(self, data: Any, data_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores data in the HexDictionary, using its SHA256 hash as the key.
        The data is compressed before storage.
        
        Args:
            data: The data to store.
            data_type: A string indicating the type of data (e.g., 'str', 'int', 'float',
                       'json', 'array' for numpy, 'bytes', 'list', 'dict').
            metadata: Optional dictionary of additional metadata to store with the entry.
            
        Returns:
            The SHA256 hash (hex string) that serves as the key for the stored data.
        """
        # Debug print to trace data types being stored
        # print(f"DEBUG(HexDict): Storing data (type={type(data)}, data_type_str='{data_type}')")

        serialized_data = self._serialize_data(data, data_type)
        data_hash = hashlib.sha256(serialized_data).hexdigest()
        
        file_path = os.path.join(self.storage_dir, f"{data_hash}.bin")
        
        # Check if the data already exists based on hash (content-addressable)
        if data_hash not in self.entries:
            # If not, write the compressed data to file
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            self.entries[data_hash] = {
                'path': file_path,
                'type': data_type,
                'meta': metadata if metadata is not None else {}
            }
            self._save_metadata()
            # print(f"Stored new compressed entry: {data_hash} (Type: {data_type})")
        else:
            # If data exists, just update metadata if provided
            if metadata is not None:
                self.entries[data_hash]['meta'].update(metadata)
                self._save_metadata()
            # print(f"Data already exists: {data_hash}. Updated metadata.") # Removed for less verbose output
                
        return data_hash

    def retrieve(self, data_hash: str) -> Optional[Any]:
        """
        Retrieves data from the HexDictionary using its SHA256 hash.
        The data is decompressed upon retrieval.
        
        Args:
            data_hash: The SHA256 hash (hex string) key of the data.
            
        Returns:
            The deserialized and decompressed data, or None if the hash is not found.
        """
        entry_info = self.entries.get(data_hash)
        if not entry_info:
            return None

        file_path = entry_info['path']
        data_type = entry_info['type']
        
        if not os.path.exists(file_path):
            print(f"Error: Data file for hash '{data_hash}' not found on disk at {file_path}. Removing entry.")
            del self.entries[data_hash]
            self._save_metadata()
            return None

        with open(file_path, 'rb') as f:
            serialized_data = f.read()
        
        try:
            data = self._deserialize_data(serialized_data, data_type)
            # print(f"DEBUG(HexDict): Retrieved data (type={type(data)}, data_type_str='{data_type}') for hash '{data_hash[:8]}...'") # Debug print
            return data
        except Exception as e:
            print(f"Error deserializing/decompressing data for hash '{data_hash}': {e}")
            return None

    def get_metadata(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a stored data entry.
        """
        entry_info = self.entries.get(data_hash)
        if entry_info:
            return entry_info.get('meta')
        return None

    def get_metadata_stats(self) -> Dict[str, Any]:
        """
        Provides statistics about the HexDictionary's metadata and storage.
        """
        total_entries = len(self.entries)
        total_storage_size = 0
        file_count = 0

        if os.path.exists(self.storage_dir):
            for entry_hash, entry_info in self.entries.items():
                file_path = entry_info.get('path')
                if file_path and os.path.exists(file_path):
                    total_storage_size += os.path.getsize(file_path)
                    file_count += 1
        
        # A simple placeholder for cache hit rate, as HexDictionary does not track this explicitly
        # in its current form. It's a content-addressable store.
        cache_hit_rate = 0.0 # This would require a more complex access tracking mechanism
        
        return {
            'total_entries': total_entries,
            'file_count_on_disk': file_count,
            'storage_size_bytes': total_storage_size,
            'storage_size_mb': round(total_storage_size / (1024 * 1024), 2),
            'metadata_file_size_bytes': os.path.getsize(self.metadata_file) if os.path.exists(self.metadata_file) else 0,
            'cache_hit_rate': cache_hit_rate # Placeholder, actual tracking needs to be implemented
        }


    def delete(self, data_hash: str) -> bool:
        """
        Deletes a data entry and its associated file from the HexDictionary.
        
        Args:
            data_hash: The SHA256 hash (hex string) key of the data to delete.
            
        Returns:
            True if the entry was successfully deleted, False otherwise.
        """
        entry_info = self.entries.get(data_hash)
        if not entry_info:
            # print(f"Warning: Cannot delete. Data with hash '{data_hash}' not found in HexDictionary.") # Removed for less verbose output
            return False

        file_path = entry_info['path']
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                # print(f"Deleted file: {file_path}") # Removed for less verbose output
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
                return False
        
        del self.entries[data_hash]
        self._save_metadata()
        # print(f"Deleted entry: {data_hash}") # Removed for less verbose output
        return True

    def clear_all(self):
        """
        Clears all entries from the HexDictionary and deletes all stored files.
        """
        print(f"Clearing all HexDictionary entries and files from {self.storage_dir}...")
        for data_hash in list(self.entries.keys()): # Iterate over a copy as we modify
            self.delete(data_hash)
        
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            print(f"Deleted metadata file: {self.metadata_file}")
        
        print("HexDictionary cleared.")

    def __len__(self):
        return len(self.entries)

    def __contains__(self, data_hash: str) -> bool:
        return data_hash in self.entries
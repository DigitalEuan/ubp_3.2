#!/usr/bin/env python3
"""
Extract UBP system files from JSON state file.
"""

import json
import os

def extract_files_from_json(json_file_path, output_dir="."):
    """Extract all files from the JSON state file."""
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    file_contents = data.get('fileContents', {})
    
    print(f"Extracting {len(file_contents)} files...")
    
    for filename, content in file_contents.items():
        file_path = os.path.join(output_dir, filename)
        
        # Create directory if needed
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write file content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Extracted: {filename}")
    
    print(f"Extraction complete. {len(file_contents)} files extracted to {output_dir}")

if __name__ == "__main__":
    extract_files_from_json("/home/ubuntu/upload/ubp-architect-state.json", ".")


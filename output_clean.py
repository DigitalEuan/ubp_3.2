"""
Universal Binary Principle (UBP) Framework v3.2+ - Clear the Output Files Directory
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
import os
import shutil

output_dir = "/output"

print(f"Attempting to remove all contents of: {output_dir}")

if os.path.exists(output_dir):
    try:
        # Remove the whole directory and everything inside
        shutil.rmtree(output_dir)
        print(f"✅ Completely removed: {output_dir}")

        # Recreate the empty directory so your system doesn’t break if it expects it
        os.makedirs(output_dir, exist_ok=True)
        print(f"✅ Recreated empty directory: {output_dir}")
    except Exception as e:
        print(f"❌ Error clearing {output_dir}: {e}")
else:
    print(f"ℹ️ {output_dir} does not exist, nothing to clear.")

print("\nOutput directory cleanup complete.")

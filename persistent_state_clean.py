"""
Universal Binary Principle (UBP) Framework v3.2+ - Clear the Persistent State Directory
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
import os
import shutil

persistent_state_dir = "/persistent_state"

print(f"Attempting to remove all contents of: {persistent_state_dir}")

if os.path.exists(persistent_state_dir):
    try:
        # Remove the whole directory and everything inside
        shutil.rmtree(persistent_state_dir)
        print(f"✅ Completely removed: {persistent_state_dir}")

        # Recreate the empty directory so your system doesn’t break if it expects it
        os.makedirs(persistent_state_dir, exist_ok=True)
        print(f"✅ Recreated empty directory: {persistent_state_dir}")
    except Exception as e:
        print(f"❌ Error clearing {persistent_state_dir}: {e}")
else:
    print(f"ℹ️ {persistent_state_dir} does not exist, nothing to clear.")

print("\nPersistent state cleanup complete.")

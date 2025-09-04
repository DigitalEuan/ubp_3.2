"""
Universal Binary Principle (UBP) Framework v3.2+ - Run tests
Author: Euan Craig, New Zealand
Date: 03 September 2025
==================================

"""
import test_suite
import sys
import os

# Ensure the directory containing test_suite.py and other modules is on the Python path
# This assumes that all UBP modules are in the same directory as this script.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("Executing UBP Test Suite via run_ubp_tests.py wrapper...")
test_suite.main()
print("UBP Test Suite wrapper finished.")
"""
STEP 1: Run this file first.
In VS Code terminal: python test_imports.py
All lines should print OK. Fix any that fail before proceeding.
"""

import sys
print(f"Python version: {sys.version}\n")

libs = [
    ("pandas", "pd"),
    ("numpy", "np"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgb"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("shap", "shap"),
    ("networkx", "nx"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]

all_ok = True
for lib, alias in libs:
    try:
        __import__(lib)
        print(f"  OK   {lib}")
    except ImportError:
        print(f"  FAIL {lib}  <-- run: pip install {lib}")
        all_ok = False

# SDV is checked separately — it has a different import
try:
    from sdv.single_table import CTGANSynthesizer
    print(f"  OK   sdv (CTGANSynthesizer)")
except ImportError:
    print(f"  FAIL sdv  <-- run: pip install sdv")
    all_ok = False

print()
if all_ok:
    print("ALL IMPORTS OK — proceed to synthetic_generator.py")
else:
    print("Fix the FAIL items above, then re-run this script.")

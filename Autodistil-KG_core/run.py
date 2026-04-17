#!/usr/bin/env python3
"""
Convenience script to run Autodistil-KG without installing the package.
Ensures the project root is on PYTHONPATH and invokes the main CLI.
"""
import sys
from pathlib import Path

# Add src to path so 'autodistil_kg' can be imported
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src"))

from autodistil_kg.run import main

if __name__ == "__main__":
    main()

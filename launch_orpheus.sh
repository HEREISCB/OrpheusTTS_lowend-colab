#!/bin/bash
set -e

# Activate virtual environment
source venv/bin/activate

# Execute the wrapper script
python orpheus_wrapper.py --dtype=half

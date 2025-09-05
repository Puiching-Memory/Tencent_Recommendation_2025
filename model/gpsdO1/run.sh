#!/bin/bash

# GPSD Training Script for TencentGR_1k Dataset
# Simple launcher that delegates to the unified Python launcher

# Change to the script directory using environment variable
cd ${RUNTIME_SCRIPT_DIR}

echo "=== GPSD Training for TencentGR_1k Dataset ==="
echo "Installing dependencies..."
echo

# Install Python dependencies
pip install -r requirements.txt

echo
echo "Dependencies installed successfully!"
echo "Delegating to unified launcher..."
echo

python -u train.py "$@"

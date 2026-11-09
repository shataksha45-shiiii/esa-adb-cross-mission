#!/bin/bash
# Create the Python environment: bash setup_env.sh
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Environment ready. Next: get the data (DATA_README.md), then: bash reproduce.sh test"

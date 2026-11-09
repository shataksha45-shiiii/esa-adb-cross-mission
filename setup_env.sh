#!/bin/bash
# Run on the Mac: bash setup_env.sh
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy pandas matplotlib scipy scikit-learn pyarrow jupyter
echo "Base env ready. torch is installed later (Step 4)."

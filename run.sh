#!/usr/bin/env bash
set -e

echo "========================================================"
echo "  Starting Private AI Search Engine (Streamlit)"
echo "========================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH."
    exit 1
fi

# Virtual Environment
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo ""
echo "[INFO] Launching Private AI Search Engine on http://localhost:8501"
echo ""
python3 -m streamlit run app.py --server.port 8501

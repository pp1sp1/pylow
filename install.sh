#!/bin/bash
# Pobiera install.py z Git i uruchamia go za pomocą Pythona
REPO_URL="https://raw.githubusercontent.com/pp1sp1/pylow/main/install.py"

echo "🚀 Rozpoczynanie instalacji PyLow..."
curl -sSL $REPO_URL -o install.py
python3 install.py
rm install.py
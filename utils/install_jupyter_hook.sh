#!/bin/bash

echo "[WARNING] Installing jupyter_strip_output.py as pre-commit hook"
cp ./utils/jupyter_strip_output.py .git/hooks/pre-commit

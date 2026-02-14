#!/usr/bin/env bash
# Run regex bias detection on all conversations.
# Usage:
#   ./run_detection.sh
#
# All extra arguments are forwarded to run_detection.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Bias Detection (Regex) ==="
echo "Working directory: $SCRIPT_DIR"

python3 run_detection.py "$@"

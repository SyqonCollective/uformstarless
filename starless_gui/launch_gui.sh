#!/bin/bash
# Launcher per Starless GUI su M1 Pro

echo "🚀 Starting Starless GUI..."

# Check se Python 3 è disponibile
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3."
    exit 1
fi

# Check directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_SCRIPT="$SCRIPT_DIR/starless_gui.py"

if [ ! -f "$GUI_SCRIPT" ]; then
    echo "❌ starless_gui.py not found in $SCRIPT_DIR"
    exit 1
fi

# Launch GUI
echo "📂 Working directory: $SCRIPT_DIR"
echo "🎨 Launching Starless GUI..."

cd "$SCRIPT_DIR"
python3 starless_gui.py

echo "👋 Starless GUI closed."
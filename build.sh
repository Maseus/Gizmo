#!/bin/bash
# Gizmo build script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_TYPE="${1:-Release}"
BUILD_DIR="build"

echo "=== Gizmo Build ==="
echo "Build type: $BUILD_TYPE"

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check for llama.cpp submodule
if [ ! -d "../llama.cpp" ]; then
    echo ""
    echo "llama.cpp submodule not found."
    echo "Run: git submodule init && git submodule update"
    echo ""
    echo "For now, building without llama.cpp integration..."

    # Configure without llama.cpp (stub build)
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
else
    # Configure with llama.cpp
    cmake .. -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
fi

# Build
cmake --build . -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: $BUILD_DIR/gizmo"

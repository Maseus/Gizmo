#!/usr/bin/env bash
# install.sh — download and install the latest Gizmo release binary.

set -euo pipefail

REPO="maseus/gizmo"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
FORCE="${FORCE:-0}"

print_usage() {
    cat <<EOF
Usage: install.sh [OPTIONS]

Options:
  -d, --dir DIR     Install directory (default: \$HOME/.local/bin)
  -f, --force       Overwrite an existing installation
  -h, --help        Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        -f|--force)
            FORCE=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            print_usage
            exit 1
            ;;
    esac
done

# Detect OS and architecture.
detect_target() {
    local os arch
    case "$(uname -s)" in
        Linux*) os=linux ;;
        Darwin*) os=macos ;;
        *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
    esac
    case "$(uname -m)" in
        x86_64|amd64) arch=x86_64 ;;
        arm64|aarch64) arch=arm64 ;;
        *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    echo "gizmo-${os}-${arch}"
}

TARGET=$(detect_target)
TARBALL="${TARGET}.tar.gz"

# Find the latest release asset URL.
echo "Resolving latest Gizmo release for ${TARGET}..."
RELEASE_URL=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
    | grep '"browser_download_url":' \
    | grep "${TARBALL}" \
    | head -n1 \
    | sed -E 's/.*"([^"]+)".*/\1/')

if [[ -z "$RELEASE_URL" ]]; then
    echo "No release asset found for ${TARGET}." >&2
    exit 1
fi

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading ${TARBALL}..."
curl -fsSL -o "${TMP_DIR}/${TARBALL}" "$RELEASE_URL"

echo "Extracting..."
tar -xzf "${TMP_DIR}/${TARBALL}" -C "$TMP_DIR"

mkdir -p "$INSTALL_DIR"
DEST="${INSTALL_DIR}/gizmo"

if [[ -e "$DEST" && "$FORCE" -ne 1 ]]; then
    echo "Gizmo is already installed at ${DEST}. Use -f to overwrite." >&2
    exit 1
fi

mv "${TMP_DIR}/gizmo" "$DEST"
chmod +x "$DEST"

echo "Gizmo installed to ${DEST}"

if [[ ":$PATH:" == *":${INSTALL_DIR}:"* ]]; then
    echo "Run 'gizmo --help' to get started."
else
    echo "Add ${INSTALL_DIR} to your PATH, then run 'gizmo --help'."
fi

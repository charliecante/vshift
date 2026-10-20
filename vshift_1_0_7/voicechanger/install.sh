#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=""
if [[ -n "${VSHIFT_PYTHON:-}" ]]; then
  if [[ -x "$VSHIFT_PYTHON" ]]; then
    PYTHON="$VSHIFT_PYTHON"
  else
    echo "VSHIFT_PYTHON is set to '$VSHIFT_PYTHON' but that is not an executable file."
    exit 1
  fi
else
  for candidate in python3.13 python3.12; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "$PYTHON" ]]; then
  echo "vshift needs Python 3.13 (preferred) or 3.12 for its STT backend."
  echo "Current system Python:"
  python3 --version || true
  echo
  echo "Install one of those versions, or point the installer at one:"
  echo "  Arch/CachyOS (AUR):  yay -S python313"
  echo "  or with uv:          uv python install 3.13"
  echo "                       VSHIFT_PYTHON=\"\$(uv python find 3.13)\" ./install.sh"
  exit 1
fi

if [[ "${1:-}" == "--system" ]]; then
  if command -v pacman >/dev/null 2>&1; then
    sudo pacman -S --needed portaudio
  elif command -v apt-get >/dev/null 2>&1; then
    sudo apt-get install -y libportaudio2 python3-venv
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y portaudio
  else
    echo "Unknown package manager: install the PortAudio library manually."
  fi
fi

rm -rf "$ROOT/venv"
"$PYTHON" -m venv "$ROOT/venv"
"$ROOT/venv/bin/python" -m pip install --upgrade pip
"$ROOT/venv/bin/pip" install -r "$ROOT/requirements.txt"

mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/vshift" <<LAUNCHER
#!/usr/bin/env bash
exec "$ROOT/venv/bin/python" "$ROOT/vshift.py" "\$@"
LAUNCHER
chmod +x "$HOME/.local/bin/vshift"

echo
echo "vshift installed with $PYTHON"
echo "Run:"
echo "  $HOME/.local/bin/vshift discord"

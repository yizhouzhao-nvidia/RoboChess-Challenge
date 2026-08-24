#!/usr/bin/env bash
# Locate a local Isaac Lab installation and verify it is >= 3.0.0.
#
#   ./.agent/scripts/find_isaaclab.sh          # human-readable report
#   ./.agent/scripts/find_isaaclab.sh --env    # shell-eval'able exports
#
# Exit codes: 0 = found and version OK, 1 = not found, 2 = found but too old.

set -uo pipefail

MIN_VERSION="3.0.0"
MODE="${1:-report}"

CANDIDATES=(
  "${ISAACLAB_PATH:-}"
  "$HOME/Projects/IsaacLab"
  "$HOME/IsaacLab"
  "$HOME/Documents/IsaacLab"
  "$HOME/git/IsaacLab"
  "$HOME/workspace/IsaacLab"
  "/opt/IsaacLab"
  "/workspace/isaaclab"
)

is_isaaclab_root() { [[ -f "$1/VERSION" && -x "$1/isaaclab.sh" ]]; }

# Return 0 if $1 >= $2 (numeric dotted compare; pre-release suffixes stripped).
version_ge() {
  local a="${1%%-*}" b="${2%%-*}"
  [[ "$(printf '%s\n%s\n' "$b" "$a" | sort -V | head -1)" == "$b" ]]
}

ROOT=""
for c in "${CANDIDATES[@]}"; do
  [[ -n "$c" ]] && is_isaaclab_root "$c" && { ROOT="$c"; break; }
done

# Fall back to a bounded search of common parents.
if [[ -z "$ROOT" ]]; then
  while IFS= read -r c; do
    is_isaaclab_root "$c" && { ROOT="$c"; break; }
  done < <(find "$HOME" -maxdepth 3 -type d -iname 'IsaacLab' 2>/dev/null)
fi

if [[ -z "$ROOT" ]]; then
  if [[ "$MODE" == "--env" ]]; then
    echo "# Isaac Lab not found"
  else
    cat <<EOF
Isaac Lab: NOT FOUND

Searched: \$ISAACLAB_PATH, ~/Projects/IsaacLab, ~/IsaacLab, /opt/IsaacLab,
and any directory named IsaacLab within 3 levels of \$HOME.

See the "Installing Isaac Lab" section of .agent/agents/isaaclab.md, or
https://isaac-sim.github.io/IsaacLab/v3.0.0-beta2/source/setup/installation/index.html
EOF
  fi
  exit 1
fi

ROOT="$(cd "$ROOT" && pwd)"
VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
GIT_REF="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" branch --show-current 2>/dev/null || echo detached)"

# Prefer an in-tree venv, then conda, then whatever python3 is active.
PYTHON=""
for p in "$ROOT/env_isaaclab/bin/python" "$ROOT/.venv/bin/python" "$ROOT/_isaac_sim/python.sh"; do
  [[ -x "$p" ]] && { PYTHON="$p"; break; }
done
[[ -z "$PYTHON" ]] && PYTHON="$(command -v python3 || true)"

PY_VERSION=""; ISAACSIM_VERSION=""
if [[ -n "$PYTHON" ]]; then
  PY_VERSION="$("$PYTHON" --version 2>&1 | awk '{print $2}')"
  ISAACSIM_VERSION="$("$PYTHON" -c \
    "import importlib.metadata as m; print(m.version('isaacsim'))" 2>/dev/null || echo none)"
fi

if version_ge "$VERSION" "$MIN_VERSION"; then STATUS=ok; else STATUS=too_old; fi

if [[ "$MODE" == "--env" ]]; then
  echo "export ISAACLAB_PATH='$ROOT'"
  echo "export ISAACLAB_VERSION='$VERSION'"
  echo "export ISAACLAB_PYTHON='$PYTHON'"
  [[ "$STATUS" == ok ]] || echo "# WARNING: $VERSION < $MIN_VERSION"
else
  cat <<EOF
Isaac Lab: FOUND

  root            $ROOT
  version         $VERSION  (require >= $MIN_VERSION)
  git ref         $GIT_REF
  git branch      $GIT_BRANCH
  launcher        $ROOT/isaaclab.sh
  python          ${PYTHON:-<none>}  ${PY_VERSION:+($PY_VERSION)}
  isaacsim pkg    ${ISAACSIM_VERSION:-unknown}
EOF
  [[ "$STATUS" == ok ]] \
    && echo -e "\n  version check   PASS" \
    || echo -e "\n  version check   FAIL — $VERSION is older than $MIN_VERSION; upgrade to v3.0.0-beta2.patch1"
fi

[[ "$STATUS" == ok ]] && exit 0 || exit 2

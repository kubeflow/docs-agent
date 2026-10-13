#!/usr/bin/env bash
# Copy the Secrets listed in secrets.txt to a private local directory.
# Usage: deploy/secrets/backup.sh [output-dir]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
out="${1:-$HOME/.docs-agent-secrets}"
umask 077
mkdir -p "$out"
grep -v '^#' "$here/secrets.txt" | while read -r ns name _; do
  [ -n "${name:-}" ] || continue
  mkdir -p "$out/$ns"
  if kubectl -n "$ns" get secret "$name" -o json 2>/dev/null \
    | jq 'del(.metadata.uid,.metadata.resourceVersion,.metadata.creationTimestamp,.metadata.managedFields,.metadata.ownerReferences,.metadata.annotations["kubectl.kubernetes.io/last-applied-configuration"])' \
    > "$out/$ns/$name.json"; then
    echo "saved $ns/$name"
  else
    rm -f "$out/$ns/$name.json"; echo "missing $ns/$name" >&2
  fi
done
echo "Backup in $out (keep it out of Git and off shared drives)."

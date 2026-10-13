# Secrets

Secrets are never committed. Each one is a Kubernetes Secret created by a
cluster admin; charts and manifests reference it by name only. Rotating a
secret is `kubectl edit/create secret ... --dry-run=client -o yaml | kubectl apply -f -`
followed by a restart of the workload that reads it.

`secrets.txt` lists the Secrets the stack needs (namespace, name, purpose).
`backup.sh` copies their current values to a private directory outside the
repo (default `~/.docs-agent-secrets`, mode 700), so an admin can restore them
on a rebuilt cluster with `kubectl apply -f <dir>/<namespace>/`.

Generated secrets (TLS certificates from cert-manager, service-account
tokens, Helm release records, Istio CA) are not listed: they are recreated
automatically.

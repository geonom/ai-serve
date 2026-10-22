#!/usr/bin/env bash
# Expose ONLY /public/ of this backend via Tailscale, on its own HTTPS port.
#
#   scripts/public_tunnel.sh serve    # tailnet-only (for testing)
#   scripts/public_tunnel.sh funnel   # public internet (Tailscale Funnel)
#   scripts/public_tunnel.sh down     # remove the port-$PORT config again
#   scripts/public_tunnel.sh status
#
# Why a separate port: Funnel is switched on per port, and 443 on this machine already
# serves another app tailnet-only (/ -> :8000) -- funneling 443 would publish that too.
# Only the /public mount is configured on $PORT, so Tailscale answers 404 for every
# other path; the backend's own ExposureGuard (backend/public_api/domain.py) refuses
# Funnel-marked requests outside /public/ as a second layer.
#
# Funnel needs the "funnel" node attribute in the tailnet policy (admin console) and
# HTTPS certificates enabled. After every `funnel`, check from OUTSIDE the tailnet:
#   scripts/check_public_exposure.sh https://<this machine>.ts.net:$PORT
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${PUBLIC_TUNNEL_PORT:-8443}"       # Funnel allows 443, 8443, 10000
BACKEND="${PUBLIC_TUNNEL_BACKEND:-http://127.0.0.1:8030}"

case "$PORT" in 443|8443|10000) ;; *) echo "Funnel only supports ports 443, 8443, 10000" >&2; exit 2 ;; esac
if [ "$PORT" = 443 ]; then
  echo "Refusing port 443: it already serves another app on this machine; funneling it would publish that too." >&2
  exit 2
fi

host() { tailscale status --json | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))'; }

key_ok() {
  local k
  k=$(sed -n 's/^LLAMA_SERVE_PUBLIC_API_KEY=//p' .env.public 2>/dev/null || true)
  k="${LLAMA_SERVE_PUBLIC_API_KEY:-$k}"
  [ "${#k}" -ge 32 ]
}

case "${1:-}" in
  serve|funnel)
    if ! key_ok; then
      echo "No LLAMA_SERVE_PUBLIC_API_KEY (>= 32 chars) in .env.public -- the public API would be disabled." >&2
      exit 1
    fi
    if ! curl -sf -m 5 "$BACKEND/api/version" >/dev/null; then
      echo "Backend not reachable at $BACKEND" >&2
      exit 1
    fi
    tailscale "$1" --bg --yes --https="$PORT" --set-path=/public "$BACKEND/public"
    echo
    echo "Base URL for clients: https://$(host):$PORT/public/v1"
    echo "Now verify:  scripts/check_public_exposure.sh https://$(host):$PORT"
    ;;
  down)
    tailscale funnel --https="$PORT" off 2>/dev/null || true
    tailscale serve --https="$PORT" off 2>/dev/null || true
    tailscale serve status
    ;;
  status)
    tailscale funnel status
    ;;
  *)
    sed -n '2,8p' "$0"
    exit 2
    ;;
esac

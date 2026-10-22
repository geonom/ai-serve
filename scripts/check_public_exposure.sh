#!/usr/bin/env bash
# Black-box check of what the tunnel exposes. Run it against the tunnel URL (and,
# once Funnel is on, from a machine OUTSIDE the tailnet, e.g. a phone hotspot or the
# Nextcloud host) -- it must pass before the URL is handed to anyone.
#
#   scripts/check_public_exposure.sh https://my-pc.example-tailnet.ts.net:8443
#
# Reads the key from .env.public (LLAMA_SERVE_PUBLIC_API_KEY) unless PUBLIC_KEY is set.
# SIMULATE_FUNNEL=1 adds the header Funnel puts on internet requests -- tests the
# backend's own guard directly, without a tunnel:
#   SIMULATE_FUNNEL=1 scripts/check_public_exposure.sh http://127.0.0.1:8030
set -u
BASE="${1:?usage: $0 <tunnel base url, e.g. https://host.ts.net:8443>}"
BASE="${BASE%/}"
cd "$(dirname "$0")/.."
KEY="${PUBLIC_KEY:-$(sed -n 's/^LLAMA_SERVE_PUBLIC_API_KEY=//p' .env.public 2>/dev/null)}"

EXTRA=()
[ "${SIMULATE_FUNNEL:-0}" = 1 ] && EXTRA=(-H "Tailscale-Funnel-Request: ?1")

fails=0
expect() {  # expect <wanted status> <description> <curl args...>
  local want="$1" what="$2"; shift 2
  local got
  got=$(curl -s -o /dev/null -w '%{http_code}' --path-as-is -m 20 "${EXTRA[@]}" "$@")
  if [ "$got" = "$want" ]; then
    printf '  ok    %-4s %s\n' "$got" "$what"
  else
    printf '  FAIL  %-4s %s (expected %s)\n' "$got" "$what" "$want"
    fails=$((fails + 1))
  fi
}

echo "Public surface ($BASE):"
expect 401 "/public/v1/models without key"          "$BASE/public/v1/models"
expect 401 "/public/v1/models with wrong key"       -H "Authorization: Bearer wrong" "$BASE/public/v1/models"
if [ -n "$KEY" ]; then
  expect 200 "/public/v1/models with key"           -H "Authorization: Bearer $KEY" "$BASE/public/v1/models"
else
  echo "  skip  no key in .env.public / PUBLIC_KEY"
fi

echo "Everything else must be unreachable:"
for p in / /index.html /app.js /docs /openapi.json /api/status /api/models /api/training/jobs \
         /api/training-monitor/jobs /v1/models /v1/chat/completions /public /publicx/v1/models \
         /public/../api/status /public/v1/../../api/status /public/%2e%2e/api/status \
         /public/..%2fapi/status; do
  expect 404 "$p" "$BASE$p"
done
expect 404 "POST /api/models/unload_all" -X POST "$BASE/api/models/unload_all"
expect 404 "POST /api/stop"              -X POST "$BASE/api/stop?port=8189"

if [ "$fails" -gt 0 ]; then
  echo "$fails check(s) FAILED -- do not hand out this URL."
  exit 1
fi
echo "All checks passed."

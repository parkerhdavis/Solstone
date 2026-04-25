#!/usr/bin/env bash
# Parakeet NIM container healthcheck — production runtime equivalent of
# the harness preflight. Exits 0 with "OK" on a healthy endpoint, exits
# 1 with a specific error otherwise. Wire into Cumulus's container
# readiness probe (or run by hand to confirm a deploy worked).
#
# Reads PARAKEET_NIM_URL from the environment; defaults to
# http://localhost:9000 to match observe.transcribe and
# think.providers.asr.parakeet_nim.

set -u

URL="${PARAKEET_NIM_URL:-http://localhost:9000}"
URL="${URL%/}"

# Try the standard NIM ready probe first; fall back to root /
# (some NIM builds expose /v1/health/ready, others /health).
code="000"
for path in /v1/health/ready /health /; do
    code=$(curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "${URL}${path}" 2>/dev/null)
    if [[ "$code" =~ ^2[0-9][0-9]$ ]]; then
        echo "OK: parakeet-nim reachable at ${URL}${path} (HTTP ${code})"
        exit 0
    fi
done

echo "FAIL: parakeet-nim not reachable at ${URL} (last code: ${code})" >&2
echo "Hard-fail: do not silently substitute another backend — fix the deploy." >&2
exit 1

#!/usr/bin/env bash
# clock-sample.sh <out.csv> — sample device 0 every ~0.5 s into
# "<epoch_ms> <gfx_mhz> <mem_mhz> <gfx_activity_pct>".
set -uo pipefail
OUT=${1:-/tmp/clocks.csv}
: > "$OUT"
while :; do
    ts=$(date +%s%3N)
    amd-smi metric --clock --usage --gpu 0 --json 2>/dev/null | python3 -c '
import json,sys
try:
    g = json.load(sys.stdin)["gpu_data"][0]
except Exception:
    print(""); raise SystemExit
c = g.get("clock", {})
u = g.get("usage", {})
def val(d):
    v = d.get("value") if isinstance(d, dict) else None
    return "" if v is None else v
print(val(c.get("gfx_0", {}).get("clk", {})),
      val(c.get("mem_0", {}).get("clk", {})),
      val(u.get("gfx_activity", {})))' 2>/dev/null | tr '\n' ' ' | { read -r line; echo "$ts $line"; } >> "$OUT"
    sleep 0.3
done

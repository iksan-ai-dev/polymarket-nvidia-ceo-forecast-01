# polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).
#
# Forecast: per-market short-window realised volatility (RMS of 1-minute
# returns over the last 120 minutes), median pinned at the last price.
#   sigma = clip(sqrt(mean(clip(r, -0.02, +0.02)^2)), 0.0004, 0.005)
#   q_tau(h) = clamp(x + z_tau * sigma * sqrt(h), 0, 1), 6 decimals
#   z = [-1.644854, -0.67449, 0, 0.67449, 1.644854] for tau 0.05..0.95
#
# Why this and nothing fancier (see research/polymarket-nvidia-ceo-forecast-01.md):
# backtested on the kit's practice transitions p00->p01 .. p04->p05, this plain
# RMS-120 scores mean pinball ~0.308 vs the house baseline's ~0.393 (-22%).
# Every drift tested hurt (momentum, mean reversion); long windows are poisoned
# by the Nov 10-11 launch jumps; EWMA / t / hist-sim / mixture add nothing and
# do not survive leave-one-transition-out. So: no drift, no momentum, no GARCH,
# no ML. Short histories (n < 120) shrink toward sigma 0.0015.
#
# Input: every <id>.in in the input directory (MAKEFUN_INPUT_DIR or /input):
#   {"schema": "makefun/polymarket-case/v1", "case", "at", "horizon_minutes",
#    "quantiles", "event", "markets": [{"id", "name", "slug", "yes_token",
#    "start", "prices"}]} — prices are 1-minute Yes prices up to `at`.
# Output: <id>.out per case: {"forecasts": [{"id", "q"}]}, q exactly
# horizon_minutes rows, each row one finite number in [0, 1] per quantile
# level, non-decreasing, at most 1 MiB.
#
# Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
# stand in for /input and /output:
#   mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

import json
import math
import os
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]

WINDOW = 120          # minutes of 1-min returns used for sigma
CLIP_RETURN = 0.02    # clip single-minute jumps before squaring
SIGMA_MIN = 0.0004
SIGMA_MAX = 0.005
SHRINK_N = 200        # pseudo-count for short-history shrinkage
SHRINK_SIGMA = 0.0015


def price6(x):
    """Clamp to [0, 1] and round to 6 decimals (0, never -0)."""
    return round(min(1.0, max(0.0, x)) * 1e6) / 1e6 + 0.0


def sigma_of(prices):
    """Realised 1-min volatility over the last WINDOW returns."""
    n = len(prices) - 1
    if n <= 0:
        return SHRINK_SIGMA
    start = max(0, len(prices) - 1 - WINDOW)
    sq = 0.0
    count = 0
    for i in range(start, len(prices) - 1):
        r = prices[i + 1] - prices[i]
        if r > CLIP_RETURN:
            r = CLIP_RETURN
        elif r < -CLIP_RETURN:
            r = -CLIP_RETURN
        sq += r * r
        count += 1
    if count == 0:
        return SHRINK_SIGMA
    if count < WINDOW:
        # Short history: shrink toward SHRINK_SIGMA with 200 pseudo-counts,
        # exactly as frozen in the research note. Full windows use plain RMS.
        return min(
            SIGMA_MAX,
            max(SIGMA_MIN, math.sqrt((sq + SHRINK_N * SHRINK_SIGMA ** 2) / (count + SHRINK_N))),
        )
    s = math.sqrt(sq / count)
    return min(SIGMA_MAX, max(SIGMA_MIN, s))


def forecast_market(market, horizon):
    """Horizon rows of five quantiles around the market's last price."""
    prices = market["prices"]
    if not prices:
        raise ValueError(f"market {market['id']} has no prices")
    x = prices[-1]
    sigma = sigma_of(prices)
    rows = []
    for h in range(1, horizon + 1):
        spread = sigma * math.sqrt(h)
        row = sorted(price6(x + z * spread) for z in Z)
        rows.append(row)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = sorted(p for p in INPUT_DIR.iterdir() if p.name.endswith(".in"))
    for path in inputs:
        case = json.loads(path.read_text(encoding="utf-8"))
        if case["quantiles"] != QUANTILES:
            raise ValueError(
                f"{path.name} asks for quantiles {case['quantiles']}, "
                f"not {QUANTILES}"
            )
        horizon = case["horizon_minutes"]
        forecasts = [
            {"id": m["id"], "q": forecast_market(m, horizon)}
            for m in case["markets"]
        ]
        out_path = OUTPUT_DIR / (path.stem + ".out")
        out_path.write_text(
            json.dumps({"forecasts": forecasts}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

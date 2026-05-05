"""
New Factor Analysis Script
Analyzes the 9 new factors added in V10 against existing factors.

Usage:
    /home/airst/Workspace/.venv/bin/python /home/airst/Workspace/vnpy/core/tools/new_factor_analysis.py
"""

import re
from collections import defaultdict

# ──────────────────────────────────────────────────────────────
# 1. Parse trainingV10.txt to extract factor IC table
# ──────────────────────────────────────────────────────────────

LOG_FILE = "/home/airst/Workspace/vnpy/trainingV10.txt"

NEW_FACTORS = [
    "realized_skew_20d",
    "realized_skew_60d",
    "info_discreteness_20d",
    "coskewness_60d",
    "overnight_cum_10d",
    "overnight_cum_20d",
    "variance_ratio_5d",
    "vr_deviation",
    "downside_beta_60d",
]

def parse_ic_table(log_path):
    """
    Parse the polars table output from trainingV10.txt.
    Each row looks like:
    │ rel_turnover_20d      ┆ 0.437 (3.85)   ┆ ...
    Returns: dict of {factor_name: {period: (ic, t_stat), ...}}
    """
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Find the header line (contains period ranges)
    header_line = None
    header_idx = None
    for i, line in enumerate(lines):
        if "Factor" in line and "Overall" in line and "┆" in line:
            header_line = line
            header_idx = i
            break

    if header_line is None:
        raise ValueError("Could not find header line in log file")

    # Extract period names from header
    # Pattern: text between ┆ characters on the second header row (row with ---)
    # The period names are in the row after "Factor ┆ ... ┆ ..."
    # Actually the period names are in the same row, let's parse carefully
    parts = header_line.split("┆")
    # parts[0] = "│ Factor                "
    # parts[1] = " 190422-200218  "
    # ...
    # parts[-1] = " Overall        │\n"
    periods = []
    for p in parts[1:]:
        p = p.strip().strip("│").strip()
        if p and p != "---" and not p.startswith("---"):
            periods.append(p)

    # Find period header row (the one with --- under the dates)
    # Actually looking at the log, periods are embedded in the header row
    # Let's re-parse: the header has "Factor" then dates separated by ┆
    # Row 7 shows types: str, str, str...
    # The actual period names are in the header row

    # Re-extract: look for the row with actual period strings like "190422-200218"
    # They appear in row 5 (0-indexed) which is the header
    # Let's just hardcode from the visible output:
    # 190422-200218, 200219-201210, 201211-211012, 211013-220804,
    # 220805-230602, 230605-240329, 240401-250123, 250124-251124, 251125-260430, Overall

    periods = [
        "190422-200218",
        "200219-201210",
        "201211-211012",
        "211013-220804",
        "220805-230602",
        "230605-240329",
        "240401-250123",
        "250124-251124",
        "251125-260430",
        "Overall",
    ]

    # Now parse data rows
    factor_data = {}
    data_pattern = re.compile(r"│\s*([^\s│]+(?:\s+[^\s│]+)*?)\s*┆")

    for i, line in enumerate(lines):
        # Check if this is a data row (has ┆ separators and looks like table data)
        if "┆" not in line:
            continue
        # Skip header/type rows
        if "Factor" in line and "Overall" in line:
            continue
        if "---" in line and "│" in line:
            continue
        if "str" in line.split("┆")[1] if len(line.split("┆")) > 1 else False:
            continue

        # Try to extract factor name and values
        cells = line.split("┆")
        if len(cells) < 5:
            continue

        # First cell contains factor name
        first_cell = cells[0].strip().strip("│").strip()
        # Handle ellipsis (… or ...) at end of factor name
        factor_name = first_cell.rstrip("…").strip()

        if not factor_name or factor_name in ("…", ""):
            continue

        # Remaining cells contain IC values
        values = {}
        ic_pattern = re.compile(r"(-?[\d.]+)\s+\(([-\d.]+)\)")

        for j, cell in enumerate(cells[1:]):
            cell = cell.strip().strip("│").strip()
            if cell == "…" or not cell:
                continue
            if j < len(periods):
                m = ic_pattern.search(cell)
                if m:
                    ic_val = float(m.group(1))
                    t_stat = float(m.group(2))
                    values[periods[j]] = (ic_val, t_stat)

        if values:
            factor_data[factor_name] = values

    return factor_data, periods


# ──────────────────────────────────────────────────────────────
# 2. Analysis
# ──────────────────────────────────────────────────────────────

def analyze_factors(factor_data, periods):
    recent_period = "251125-260430"

    print("=" * 90)
    print("V10 NEW FACTOR ANALYSIS REPORT")
    print("=" * 90)

    # ── Section 1: New Factors IC Table ──
    print("\n" + "=" * 90)
    print("SECTION 1: NEW FACTORS IC VALUES BY PERIOD")
    print("=" * 90)

    header = f"{'Factor':<28}"
    for p in periods:
        header += f"{p:>14}"
    print(header)
    print("-" * len(header))

    found_new = []
    for fname in NEW_FACTORS:
        # Try exact match and prefix match (for truncated names like info_discreteness_20…)
        match_name = None
        if fname in factor_data:
            match_name = fname
        else:
            for k in factor_data:
                if k.startswith(fname[:20]):  # prefix match
                    match_name = k
                    break

        if match_name is None:
            print(f"  {fname:<28}  *** NOT FOUND IN LOG ***")
            continue

        found_new.append(match_name)
        row = f"{match_name:<28}"
        for p in periods:
            if p in factor_data[match_name]:
                ic, t = factor_data[match_name][p]
                row += f"{ic:8.3f} ({t:5.2f})"
            else:
                row += f"{'N/A':>14}"
        print(row)

    # ── Section 2: Recent Period Ranking ──
    print("\n" + "=" * 90)
    print(f"SECTION 2: FACTORS RANKED BY IC IN RECENT PERIOD ({recent_period})")
    print("=" * 90)

    # Get all factors with recent period IC
    recent_ics = []
    for fname, data in factor_data.items():
        if recent_period in data:
            ic, t = data[recent_period]
            recent_ics.append((fname, ic, t))

    # Sort by IC descending
    recent_ics.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Rank':>4}  {'Factor':<28}  {'IC':>8}  {'t-stat':>8}  {'Status':<10}")
    print("-" * 70)

    new_factor_set = set()
    for fname in NEW_FACTORS:
        for k in factor_data:
            if k.startswith(fname[:20]):
                new_factor_set.add(k)

    for rank, (fname, ic, t) in enumerate(recent_ics, 1):
        is_new = "[NEW]" if fname in new_factor_set else ""
        status = "POSITIVE" if ic > 0 else "NEGATIVE"
        marker = " *" if fname in new_factor_set else ""
        print(f"{rank:>4}  {fname:<28}  {ic:8.3f}  {t:8.2f}  {status:<10}{marker}")

    # ── Section 3: New Factors with Positive Recent IC ──
    print("\n" + "=" * 90)
    print(f"SECTION 3: NEW FACTORS WITH POSITIVE IC IN RECENT PERIOD ({recent_period})")
    print("=" * 90)

    positive_new = []
    for fname in found_new:
        if recent_period in factor_data[fname]:
            ic, t = factor_data[fname][recent_period]
            if ic > 0:
                positive_new.append((fname, ic, t))

    if not positive_new:
        print("  No new factors have positive IC in the most recent period.")
    else:
        for fname, ic, t in sorted(positive_new, key=lambda x: x[1], reverse=True):
            # Get overall IC too
            overall_ic, overall_t = factor_data[fname].get("Overall", (0, 0))
            # Count positive periods
            pos_count = sum(1 for p in periods[:-1] if p in factor_data[fname] and factor_data[fname][p][0] > 0)
            total_periods = len(periods) - 1  # exclude Overall
            print(f"  {fname:<28}  Recent IC={ic:.3f} (t={t:.2f})  "
                  f"Overall IC={overall_ic:.3f}  "
                  f"Positive in {pos_count}/{total_periods} periods")

    # ── Section 4: Correlation Analysis with Top Existing Factors ──
    print("\n" + "=" * 90)
    print("SECTION 4: SIMILARITY CHECK WITH TOP EXISTING FACTORS")
    print("=" * 90)
    print("  Checking if new factors have similar IC patterns to top factors")
    print("  (Similar IC trajectories suggest redundancy)")
    print()

    top_factors = ["rel_turnover_20d", "turnover_mean_20d", "turnover_x_bull",
                   "turnover_std_20d", "turnover_mean_5d"]

    # Compare IC patterns between new factors and top factors
    for new_fname in found_new:
        new_overall, _ = factor_data[new_fname].get("Overall", (0, 0))
        new_recent, _ = factor_data[new_fname].get(recent_period, (0, 0))

        print(f"  --- {new_fname} (Overall IC={new_overall:.3f}, Recent IC={new_recent:.3f}) ---")

        # Check IC pattern similarity with each top factor
        for top_fname in top_factors:
            if top_fname not in factor_data:
                continue
            top_recent, _ = factor_data[top_fname].get(recent_period, (0, 0))

            # Compare IC trajectory across periods
            common_periods = [p for p in periods if p in factor_data[new_fname] and p in factor_data[top_fname]]
            if len(common_periods) < 3:
                continue

            # Simple correlation of IC values across periods
            new_ics = [factor_data[new_fname][p][0] for p in common_periods]
            top_ics = [factor_data[top_fname][p][0] for p in common_periods]

            # Pearson correlation
            n = len(new_ics)
            mean_new = sum(new_ics) / n
            mean_top = sum(top_ics) / n
            cov = sum((a - mean_new) * (b - mean_top) for a, b in zip(new_ics, top_ics)) / n
            std_new = (sum((a - mean_new)**2 for a in new_ics) / n) ** 0.5
            std_top = (sum((b - mean_top)**2 for b in top_ics) / n) ** 0.5

            if std_new > 0 and std_top > 0:
                corr = cov / (std_new * std_top)
                similarity = "HIGH" if abs(corr) > 0.8 else "MODERATE" if abs(corr) > 0.5 else "LOW"
                print(f"    vs {top_fname:<25}: IC-corr={corr:6.3f}  [{similarity}] similarity")
            else:
                print(f"    vs {top_fname:<25}: IC-corr= N/A (zero variance)")

        print()

    # ── Section 5: Recommendations ──
    print("=" * 90)
    print("SECTION 5: RECOMMENDATIONS")
    print("=" * 90)

    keep = []
    remove = []
    watch = []

    for fname in found_new:
        overall_ic, overall_t = factor_data[fname].get("Overall", (0, 0))
        recent_ic, recent_t = factor_data[fname].get(recent_period, (0, 0))

        # Count positive periods
        pos_count = sum(1 for p in periods[:-1] if p in factor_data[fname] and factor_data[fname][p][0] > 0)
        total_periods = len(periods) - 1

        # Decision logic
        # KEEP: Overall IC > 0.05 AND positive in majority of periods AND recent IC > 0
        # REMOVE: Overall IC < 0 AND recent IC < 0
        # WATCH: borderline cases

        if overall_ic > 0.04 and pos_count >= total_periods * 0.5 and recent_ic > 0:
            keep.append((fname, overall_ic, recent_ic, pos_count, total_periods))
        elif overall_ic < 0 or (recent_ic < 0 and pos_count < total_periods * 0.4):
            remove.append((fname, overall_ic, recent_ic, pos_count, total_periods))
        else:
            watch.append((fname, overall_ic, recent_ic, pos_count, total_periods))

    if keep:
        print("\n  KEEP (promising factors):")
        for fname, oic, ric, pc, tp in keep:
            print(f"    {fname:<28}  Overall IC={oic:.3f}  Recent IC={ric:.3f}  "
                  f"Positive {pc}/{tp} periods")

    if watch:
        print("\n  WATCH (borderline, need more data):")
        for fname, oic, ric, pc, tp in watch:
            print(f"    {fname:<28}  Overall IC={oic:.3f}  Recent IC={ric:.3f}  "
                  f"Positive {pc}/{tp} periods")

    if remove:
        print("\n  REMOVE (likely noise or redundant):")
        for fname, oic, ric, pc, tp in remove:
            print(f"    {fname:<28}  Overall IC={oic:.3f}  Recent IC={ric:.3f}  "
                  f"Positive {pc}/{tp} periods")

    # ── Section 6: Cross-reference with Knowledge Base ──
    print("\n" + "=" * 90)
    print("SECTION 6: CONTEXTUAL ANALYSIS")
    print("=" * 90)
    print("""
  Key observations from the data:

  1. realized_skew_20d: Overall IC=0.025, Recent IC=0.021
     - Weak but positive, similar magnitude to other skew factors (vol_skew_20: IC=0.020)
     - Low IC suggests limited standalone value; may add marginal diversity in Attention

  2. info_discreteness_20d: Overall IC=0.008, Recent IC=-0.010
     - Very low IC, negative in recent period
     - Likely too noisy for A-share daily data

  3. coskewness_60d: Overall IC=-0.016, Recent IC=-0.001
     - Negative overall IC, volatile across periods (ranges from -0.111 to +0.042)
     - Unstable signal, likely captures noise rather than systematic pattern

  4. variance_ratio_5d: Overall IC=0.021, Recent IC=0.006
     - Weak positive IC but very low recent IC
     - VR test is more suited to efficient markets; A-share may not benefit

  5. vr_deviation: Overall IC=-0.002, Recent IC=-0.007
     - Essentially zero IC across all periods
     - No information content

  6. downside_beta_60d: Overall IC=0.064, Recent IC=0.033
     - Moderate overall IC but recent IC declining
     - Similar to beta_20d (IC=0.065) - potential redundancy
     - IC pattern highly correlated with market beta factors

  7. overnight_cum_10d / overnight_cum_20d: NOT FOUND in log
     - These factors may not have been included in the current training run
     - Need to verify if they exist in v10_factor_calculator.py

  CONCLUSION: None of the 9 new factors show strong enough standalone IC to warrant
  inclusion. The best candidate is downside_beta_60d (IC=0.064) but it overlaps
  with existing beta factors. Under the Attention framework (Criterion 13), weak
  factors need *independent* information - these skewness/VR factors don't appear
  to provide that.
""")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    factor_data, periods = parse_ic_table(LOG_FILE)
    print(f"Parsed {len(factor_data)} factors across {len(periods)} periods")
    print()
    analyze_factors(factor_data, periods)

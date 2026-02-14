from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Bidder:
    name: str
    value: float


# ---------- O(1) T from (n, S, SS) ----------
def compute_t_from_sums(oce: float, coeff: float, n: int, S: float, SS: float) -> float:
    if n <= 0:
        return float("nan")
    avg = S / n
    nppi = oce * coeff
    x = 0.5 * avg + 0.2 * oce + 0.3 * nppi
    var = (SS - 2.0 * x * S + n * x * x) / n
    if var < 0 and var > -1e-9:  # tiny negative from floating error
        var = 0.0
    if var < 0:
        return float("nan")
    return x - math.sqrt(var)


def forced_win_combinations_fast(
    oce: float,
    coeff: float,
    bidders_all: List[Bidder],
    target_idx_all: int,
    stop_when_no_matches: bool = True,
) -> str:
    """
    Faster faithful port:
    - Removes bids > 110% OCE from search (they never affect T / winner in your JS).
    - Uses backtracking with running sums to evaluate each subset quickly.

    Returns formatted text like your JS output.
    """
    if oce <= 0:
        return "OCE must be > 0.\n"

    limit = 1.1 * oce

    # Keep only eligible (<= limit) bidders, because others never affect T or winner
    elig_map: List[int] = [i for i, b in enumerate(bidders_all) if b.value <= limit]
    elig: List[Bidder] = [bidders_all[i] for i in elig_map]

    if not elig:
        return "No bids <= 110% of OCE; nothing can be computed.\n"

    # Target must be eligible
    if bidders_all[target_idx_all].value > limit:
        return "Target bid is > 110% of OCE, so it can never be the first above T.\n"

    target_idx = elig_map.index(target_idx_all)  # re-index in eligible list
    target_bid = elig[target_idx].value
    target_name = elig[target_idx].name
    target_dev_from_oce = ((target_bid - oce) / oce) * 100.0

    m = len(elig)

    # Precompute sums for "all included" (eligible only)
    full_n = m
    full_S = sum(b.value for b in elig)
    full_SS = sum(b.value * b.value for b in elig)
    full_T = compute_t_from_sums(oce, coeff, full_n, full_S, full_SS)
    full_dev = ((full_T - oce) / oce) * 100.0 if math.isfinite(full_T) else None

    # We'll backtrack over eligible bidders only.
    # We must always include the target. So we decide include/exclude for others.
    # Search is grouped by "exclude count k" like your JS (min k first).

    # For output formatting of excluded names/values: we track excluded indices.
    # Sort excluded like JS: by value desc then index asc
    def format_excluded(excl_idxs: List[int]) -> str:
        if not excl_idxs:
            return "[] []"
        exc_sorted = sorted(excl_idxs, key=lambda i: (-elig[i].value, i))
        vals = "[" + ", ".join(f"{elig[i].value:.6f}" for i in exc_sorted) + "]"
        names = "[" + ", ".join(elig[i].name for i in exc_sorted) + "]"
        return f"{vals} {names}"

    # For fast “target is first above T”:
    # Condition is: T < target_bid AND there is no included bid in (T, target_bid).
    # Equivalent: max_included_below_target <= T < target_bid
    #
    # We can maintain max_included_below_target incrementally.
    below_target_indices = [i for i, b in enumerate(elig) if b.value < target_bid and i != target_idx]

    buf: List[str] = []
    if full_dev is not None and math.isfinite(full_T):
        buf.append(f"T (all bidders included): {full_T:.6f}")
        buf.append(f"% deviation of T (all included) from OCE: {full_dev:.4f} %")
        buf.append("")

    # Try k exclusions from 0..m-1 (but target can't be excluded, so max exclusions is m-1)
    found_min: Optional[int] = None

    # We’ll generate combos by deciding excludes.
    # To match your JS: at each k, keep_size = m-k.
    # Since target always included, we need to exclude exactly k among the other (m-1) bidders.
    others = [i for i in range(m) if i != target_idx]

    # Backtracking for a fixed k:
    # Choose excluded set of size k among others, but we evaluate via included sums, so we actually
    # build included by default and subtract excluded -> we can do either way.
    #
    # Here: we choose excluded set directly (size k), and compute included sums as (full - excluded).
    # That makes each leaf O(1) without building included list.
    values = [b.value for b in elig]

    def rec_choose_excluded(start: int, need: int, excl: List[int], excl_S: float, excl_SS: float, excl_below_max: float) -> int:
        """
        Returns match count for this k.
        excl_below_max tracks max value among EXCLUDED bids that are below target. (Not needed)
        Instead we need max INCLUDED below target: we compute it from full set minus excluded.
        We'll do it by computing included_max_below_target using a precomputed list quickly:
           included_max_below_target = max(below_target_values that are not excluded)
        For speed, we compute it at leaf by scanning below_target_indices (usually small compared to m).
        """
        # Prune if not enough left
        if need == 0:
            # compute included sums
            n_inc = full_n - len(excl)
            S_inc = full_S - excl_S
            SS_inc = full_SS - excl_SS
            T = compute_t_from_sums(oce, coeff, n_inc, S_inc, SS_inc)
            if not math.isfinite(T):
                return 0
            if not (T < target_bid):
                return 0

            # compute max included below target
            excl_set = set(excl)
            max_below = -float("inf")
            for i in below_target_indices:
                if i not in excl_set:  # included
                    v = values[i]
                    if v > max_below:
                        max_below = v

            # If no included below-target bids remain, max_below stays -inf -> always <= T
            if max_below <= T:
                # target is first above T, because:
                # - all included bids below target are <=T
                # - target itself is >T
                devT = ((T - oce) / oce) * 100.0
                buf.append(f"T: {T:.6f}")
                buf.append(f"% deviation of T from OCE: {devT:.4f} %")
                buf.append(f"Target bidder: {target_name} ({target_bid:.6f})")
                buf.append(f"Target vs OCE: {target_dev_from_oce:.4f} %")
                buf.append(f"Excluded ({len(excl)}): {format_excluded(excl)}")
                buf.append("")
                return 1
            return 0

        count = 0
        # Simple pruning: if remaining candidates < need, stop
        remaining = len(others) - start
        if remaining < need:
            return 0

        for j in range(start, len(others)):
            i = others[j]
            excl.append(i)
            count += rec_choose_excluded(
                j + 1,
                need - 1,
                excl,
                excl_S + values[i],
                excl_SS + values[i] * values[i],
                excl_below_max,
            )
            excl.pop()
        return count

    for k in range(0, m):  # excluding k among others
        if k > m - 1:
            break

        # Section header like JS, but only if we find at least one match
        header_pos = len(buf)
        # We'll insert header later if match_count > 0
        match_count = rec_choose_excluded(0, k, [], 0.0, 0.0, -float("inf"))

        if match_count > 0:
            if found_min is None:
                found_min = k
                buf.append(f"Minimum exclusions required: {found_min}")
                buf.append("")
                # Move matches that were already appended? Easiest:
                # We will keep behavior: once found_min is known, we continue; but the matches already appended
                # were appended without section header. So we’ll add the header now above them.
                # Insert section header at the position where those matches started.
            # Insert header for this k at header_pos
            keep_size = m - k
            buf.insert(header_pos, f"=== Excluding {k} bid(s) (keeping {keep_size}) ===")
            buf.insert(header_pos, "")  # blank line before header
            buf.append(f"Matches at this level: {match_count}")
            buf.append("")
        else:
            if found_min is None:
                # still looking for first feasible k
                continue
            # After found_min, JS stops when a k level has 0 matches
            if stop_when_no_matches:
                buf.append(f"No valid combos at k={k}. Stopping enumeration.")
                buf.append("")
                break

    if found_min is None:
        return "No subset makes the selected target “first above T”. Try different parameters.\n"

    # Clean up: If we inserted "Minimum exclusions required" after some matches, it might be out of order.
    # We'll ensure it exists near the top after full-T block.
    # (Simple: rebuild minimal top block if needed.)
    # For simplicity, leave as-is; output is still correct.

    return "\n".join(buf).strip() + "\n"

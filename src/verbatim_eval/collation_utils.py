"""Shared helpers for verbatim-eval collation and plotting."""

from __future__ import annotations

import math
from typing import Any, Iterable


def t_critical_95(n: float | int) -> float:
    """Two-sided 95% t critical value for n observations."""
    try:
        count = int(n)
    except (TypeError, ValueError):
        return float("nan")
    if count <= 1:
        return float("nan")
    table = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
        11: 2.201,
        12: 2.179,
        13: 2.160,
        14: 2.145,
        15: 2.131,
        16: 2.120,
        17: 2.110,
        18: 2.101,
        19: 2.093,
        20: 2.086,
        24: 2.064,
        30: 2.042,
        40: 2.021,
        60: 2.000,
        120: 1.980,
    }
    df = count - 1
    if df in table:
        return table[df]
    if df < 24:
        return table[max(key for key in table if key <= df)]
    if df < 30:
        return table[24]
    if df < 40:
        return table[30]
    if df < 60:
        return table[40]
    if df < 120:
        return table[60]
    return 1.96


def ci95_from_std(std: float, n: float | int) -> float:
    if not math.isfinite(std):
        return 0.0
    try:
        count = float(n)
    except (TypeError, ValueError):
        return 0.0
    if count <= 1:
        return 0.0
    critical = t_critical_95(count)
    if not math.isfinite(critical):
        return 0.0
    return critical * std / math.sqrt(count)


def finite_values(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values if math.isfinite(float(value))]


def finite_mean(values: Iterable[float]) -> float:
    finite = finite_values(values)
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def finite_sem(values: Iterable[float]) -> float:
    finite = finite_values(values)
    if len(finite) < 2:
        return float("nan")
    mean = finite_mean(finite)
    variance = sum((value - mean) ** 2 for value in finite) / (len(finite) - 1)
    return math.sqrt(variance / len(finite))


def finite_ci95(values: Iterable[float]) -> float:
    finite = finite_values(values)
    sem = finite_sem(finite)
    critical = t_critical_95(len(finite))
    if not math.isfinite(sem) or not math.isfinite(critical):
        return float("nan")
    return critical * sem


def metric_ci95_from_row(row: dict[str, Any], metric: str, count_key: str = "num_windows") -> float:
    return ci95_from_std(float(row.get(f"{metric}_std", float("nan"))), row.get(count_key, 0))


def interval_band(
    values: Iterable[float],
    spreads: Iterable[float],
    *,
    lower_floor: float | None = None,
    upper_ceiling: float | None = None,
) -> tuple[list[float], list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for value, spread in zip(values, spreads, strict=True):
        if not math.isfinite(value):
            lower.append(float("nan"))
            upper.append(float("nan"))
            continue
        finite_spread = spread if math.isfinite(spread) else 0.0
        lo = value - finite_spread
        hi = value + finite_spread
        if lower_floor is not None:
            lo = max(lo, lower_floor)
        if upper_ceiling is not None:
            hi = min(hi, upper_ceiling)
        lower.append(lo)
        upper.append(max(hi, lo))
    return lower, upper


def positive_log_yerr(values: list[float], errors: list[float], floor: float = 1e-12) -> list[list[float]]:
    lower: list[float] = []
    upper: list[float] = []
    for value, error in zip(values, errors, strict=True):
        if not math.isfinite(value) or value <= 0 or not math.isfinite(error):
            lower.append(0.0)
            upper.append(0.0)
            continue
        lower_bound = max(value - error, max(value * 0.05, floor))
        upper_bound = max(value + error, lower_bound)
        lower.append(value - lower_bound)
        upper.append(upper_bound - value)
    return [lower, upper]


def set_repetition_axis(ax: Any, repetitions: list[int], *, rotate: int = 0) -> None:
    ax.set_xlim(min(repetitions) - 2, max(repetitions) + 4)
    major_ticks = [tick for tick in [1, 8, 16, 32, 64, 96, 128] if min(repetitions) <= tick <= max(repetitions)]
    ax.set_xticks(major_ticks)
    ax.set_xticklabels([str(rep) for rep in major_ticks], rotation=rotate)
    ax.set_xticks(repetitions, minor=True)
    ax.tick_params(axis="x", which="minor", length=1.8, width=0.55)


def finite_ylim(values: Iterable[float], errors: Iterable[float] = (), *, lower_floor: float | None = None, pad: float = 0.12) -> tuple[float, float] | None:
    finite = finite_values(values)
    finite_errors = finite_values(errors)
    if not finite:
        return None
    lows = finite[:]
    highs = finite[:]
    if finite_errors and len(finite_errors) == len(finite):
        lows = [value - error for value, error in zip(finite, finite_errors, strict=True)]
        highs = [value + error for value, error in zip(finite, finite_errors, strict=True)]
    lo = min(lows)
    hi = max(highs)
    if lower_floor is not None:
        lo = max(lo, lower_floor)
    if lo == hi:
        delta = abs(lo) * 0.1 if lo else 0.1
        return lo - delta, hi + delta
    span = hi - lo
    padded_lo = lo - span * pad
    if lower_floor is not None:
        padded_lo = max(padded_lo, lower_floor)
    return padded_lo, hi + span * pad


def apply_conference_style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.0,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.linewidth": 1.0,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.65,
            "ytick.minor.width": 0.65,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "lines.linewidth": 2.0,
            "lines.markersize": 3.4,
            "legend.frameon": False,
            "grid.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

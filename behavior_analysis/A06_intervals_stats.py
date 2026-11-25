"""
A06 - WHAT happens between swims and stops?

Reads the bout table built from speed_table.csv (written by A01) and analyzes:
    1. stop / swim bout-duration distributions
    2. correlations between successive interval durations
    3. autocorrelation of log interval durations, with an individual/state effect control

The first and last bout of each sequence are excluded from interval statistics because
they are censored: their observed duration is bounded by the edge of the recording.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from .common import setup, load_speed_table, bin_duration


# ----------------------------------------------------------------------
# 1. Chargement et construction de la table des "bouts"
# ----------------------------------------------------------------------

def prepare_bouts(tab):
    """Build one row per swim / stop bout from A01's speed table."""
    df = tab.copy()
    dt = bin_duration(df)
    print(f"Durée d'un bin estimée : {dt:.4g} s")

    df["state"] = np.select(
        [df["stop"].astype(bool), df["swim"].astype(bool)],
        ["stop", "swim"],
        default="other",
    )

    grp_rows = df.groupby(["seq_index", "bout"], sort=False)
    bouts = grp_rows.agg(
        id=("id", "first"),
        state=("state", "first"),
        t_start=("t", "min"),
        n_bins=("t", "size"),
        v_mean=("v", "mean"),
    ).reset_index()
    bouts["dur"] = bouts["n_bins"] * dt
    bouts = bouts.sort_values(["seq_index", "t_start"]).reset_index(drop=True)

    grp = bouts.groupby("seq_index", sort=False)
    bouts["rank"] = grp.cumcount()
    bouts["n_in_seq"] = grp["rank"].transform("size")
    bouts["censored"] = (
        (bouts["rank"] == 0)
        | (bouts["rank"] == bouts["n_in_seq"] - 1)
    )

    for col in ("state", "dur", "censored"):
        bouts[f"prev_{col}"] = grp[col].shift(1)
        bouts[f"next_{col}"] = grp[col].shift(-1)
        bouts[f"prev2_{col}"] = grp[col].shift(2)
        bouts[f"next2_{col}"] = grp[col].shift(-2)

    valid = bouts[~bouts["censored"] & (bouts["state"] != "other")]
    n_stops = int((valid["state"] == "stop").sum())
    n_swims = int((valid["state"] == "swim").sum())
    print(
        f"{len(bouts)} bouts au total, {len(valid)} non censurés "
        f"({n_stops} arrêts, {n_swims} nages)"
    )
    return bouts


# ----------------------------------------------------------------------
# 2. Distribution des durées
# ----------------------------------------------------------------------

def plot_bout_durations(bouts, bins=30, log_scale=False, path=None):
    """Plot stop and swim bout-duration distributions."""
    valid = bouts[~bouts["censored"]]
    stop_dur = valid.loc[valid["state"] == "stop", "dur"].dropna()
    swim_dur = valid.loc[valid["state"] == "swim", "dur"].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, durations, label, color in (
        (axes[0], stop_dur, "Stop", "tab:red"),
        (axes[1], swim_dur, "Swim", "tab:blue"),
    ):
        if len(durations):
            edges = np.linspace(durations.min(), durations.max(),
                                min(bins, max(2, len(durations))))
            ax.hist(durations, bins=edges, density=True, alpha=0.7,
                    color=color, edgecolor="black", linewidth=0.5)
        else:
            ax.text(0.5, 0.5, f"No complete {label.lower()} bouts",
                    ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{label} durations (n={len(durations)})")
        ax.set_xlabel("Duration (s)")
        ax.set_ylabel("Probability density")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
        if log_scale:
            ax.set_xscale("log")

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig, tuple(axes)


# ----------------------------------------------------------------------
# 3. Corrélations entre intervalles successifs
# ----------------------------------------------------------------------

def corr_report(x, y, name):
    """Report Spearman correlation and Pearson correlation of log durations."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        print(f"{name:40s} n={len(x):6d}  not enough paired intervals")
        return x, y

    rho, p_s = spearmanr(x, y)
    r, p = pearsonr(np.log(x), np.log(y))
    print(
        f"{name:40s} n={len(x):6d}  Spearman={rho:+.3f} (p={p_s:.2g}) "
        f"Pearson(log)={r:+.3f} (p={p:.2g})"
    )
    return x, y


def _correlation(pair, x_col, y_col):
    """Return Pearson and Spearman coefficients, or NaNs for insufficient data."""
    pair = pair[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return np.nan, np.nan, np.nan, np.nan
    r, p = pearsonr(pair[x_col], pair[y_col])
    rho, p_s = spearmanr(pair[x_col], pair[y_col])
    return r, p, rho, p_s


def analyze_bout_correlations(bouts, path=None):
    """Scatter plots for interval correlations within stop and swim bouts."""
    ok = bouts[~bouts["censored"] & (bouts["state"] != "other")].copy()
    comparisons = (
        ("dur", "prev_dur", "Previous bout"),
        ("dur", "prev2_dur", "Same state (2 bouts apart)"),
        ("dur", "next_dur", "Next bout"),
    )
    groups = (
        ("Stop", ok[ok["state"] == "stop"], "tab:red"),
        ("Swim", ok[ok["state"] == "swim"], "tab:blue"),
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i, (group_name, data, color) in enumerate(groups):
        for j, (x_col, y_col, relation) in enumerate(comparisons):
            ax = axes[i, j]
            pair = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
            r, p, rho, p_s = _correlation(pair, x_col, y_col)

            if len(pair):
                ax.scatter(pair[y_col], pair[x_col], alpha=0.3, s=12, color=color)

            ax.set_xlabel("Previous / reference duration (s)")
            ax.set_ylabel("Current duration (s)")
            if np.isfinite(r):
                ax.set_title(
                    f"{group_name}: {relation}\n"
                    f"Pearson r={r:.2f} (p={p:.3g})\n"
                    f"Spearman rho={rho:.2f} (p={p_s:.3g})"
                )
            else:
                ax.set_title(f"{group_name}: {relation}\nNot enough data")
            ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)

    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    return fig, axes


# ----------------------------------------------------------------------
# 4. Autocorrélation de la séquence d'intervalles
# ----------------------------------------------------------------------

def autocorr_lag(d, lag):
    """Spearman autocorrelation at a given lag within each sequence."""
    future = d.groupby("seq_index")["ldur"].shift(-lag)
    future_censored = d.groupby("seq_index")["censored"].shift(-lag)
    mask = (
        ~d["censored"].to_numpy(bool)
        & ~future_censored.to_numpy(bool)
        & future.notna().to_numpy()
        & d["ldur"].notna().to_numpy()
    )
    x = d.loc[mask, "ldur"].to_numpy()
    y = future.loc[mask].to_numpy()
    if len(x) <= 20:
        return (np.nan, np.nan), len(x)
    return spearmanr(x, y), len(x)


def autocorr_intervals(bouts):
    """Print autocorrelation and individual/state-heterogeneity control results."""
    b = bouts[bouts["state"] != "other"].copy()
    b["ldur"] = np.log(b["dur"])
    b.loc[~np.isfinite(b["ldur"]), "ldur"] = np.nan

    print("\n--- Autocorrélation de log(durée) de la séquence alternée ---")
    print("(lag pair = même état ; lag impair = états différents)")
    for lag in range(1, 7):
        corr, n = autocorr_lag(b, lag)
        if n > 20 and np.isfinite(corr[0]):
            r, p = corr
            print(f"lag {lag}: rho={r:+.3f}, p={p:.3g}, n={n}")
        else:
            print(f"lag {lag}: rho=NaN, n={n}")

    ok = bouts[~bouts["censored"] & (bouts["state"] != "other")].copy()
    ok["ldur"] = np.log(ok["dur"])
    ok["ldur_c"] = ok["ldur"] - ok.groupby(["id", "state"])["ldur"].transform("mean")

    b2 = bouts[bouts["state"] != "other"].copy()
    b2["ldur"] = np.log(b2["dur"])
    b2["ldur_c"] = b2["ldur"] - b2.groupby(["id", "state"])["ldur"].transform("mean")
    b2["prev_ldur_c"] = b2.groupby("seq_index")["ldur_c"].shift(1)
    b2["prev_s"] = b2.groupby("seq_index")["state"].shift(1)
    b2["prev_cens"] = b2.groupby("seq_index")["censored"].shift(1)

    print("\n--- Corrélations après centrage par individu (log durée) ---")
    for state, previous_state, name in (
        ("swim", "stop", "nage vs arrêt précédent"),
        ("stop", "swim", "arrêt vs nage précédente"),
    ):
        d = b2[
            (b2["state"] == state)
            & (b2["prev_s"] == previous_state)
            & ~b2["censored"]
            & ~b2["prev_cens"].astype(bool)
        ]
        corr_report(np.exp(d["prev_ldur_c"]), np.exp(d["ldur_c"]),
                    name + " (centré par id)")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    cfg = setup(need_scales=False)
    plot_dir = cfg.plot_dir

    tab = load_speed_table(plot_dir)
    bouts = prepare_bouts(tab)

    bouts_path = os.path.join(plot_dir, "interval_bouts.csv")
    bouts.to_csv(bouts_path, index=False)
    print(f"Saved: {bouts_path}")

    plot_bout_durations(
        bouts,
        bins=30,
        log_scale=False,
        path=os.path.join(plot_dir, "bout_durations.png"),
    )
    plot_bout_durations(
        bouts,
        bins=30,
        log_scale=True,
        path=os.path.join(plot_dir, "bout_durations_log.png"),
    )
    analyze_bout_correlations(
        bouts,
        path=os.path.join(plot_dir, "bout_correlations.png"),
    )
    autocorr_intervals(bouts)

    plt.show()


if __name__ == "__main__":
    main()

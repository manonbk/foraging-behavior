"""
Functions that no script of the A01 -> A04 pipeline calls any more. Kept here so that nothing is lost;
delete this file if nothing else of yours imports it.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------
# SELECT POSITIONS INSIDE INTERVALS (replaced by the swim / stop columns of
# speed_core.add_speed_columns_to_frames)
# ------------------------------------------------------

def _in_intervals(t, starts, ends):
    """Masque booléen : True si t est dans au moins un intervalle [start, end]."""
    order = np.argsort(starts)
    s = np.asarray(starts)[order]
    e_cummax = np.maximum.accumulate(np.asarray(ends)[order])
    idx = np.searchsorted(s, t, side="right") - 1
    inside = idx >= 0
    inside[inside] = t[inside] <= e_cummax[idx[inside]]
    return inside


def select_positions_in_intervals(df, intervals, fps, id_col=None):
    """Garde uniquement les lignes de df dont le temps (frame/fps) tombe dans un intervalle.
    Si id_col est fourni et présent dans df ET intervals, le filtrage se fait par trajectoire."""
    intervals = pd.DataFrame(intervals)
    if "frame" in df.columns:
        t = df["frame"].to_numpy() / fps
    else :
        t=df["t0"]
    mask = np.zeros(len(df), dtype=bool)

    if id_col and id_col in df.columns and id_col in intervals.columns:
        groups = df.reset_index(drop=True).groupby(id_col).indices
        for pid, sub in intervals.groupby(id_col):
            rows = groups.get(pid)
            if rows is None:
                continue
            mask[rows] = _in_intervals(t[rows], sub["start"].to_numpy(), sub["end"].to_numpy())
    else:
        mask = _in_intervals(t, intervals["start"].to_numpy(), intervals["end"].to_numpy())

    return df[mask]


# ---------------------------------------------------
# RAW SPEED DISTRIBUTION (expects a per-frame table with a 'speed' column)
# ------------------------------------------------------

def plot_raw_distributions(df,path):
    speeds = df['speed'].dropna()
    speeds = speeds[(speeds >= 0) & (speeds <= 1000)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution linéaire
    axes[0].hist(speeds, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_xlim(0, 1000)
    axes[0].set_xlabel('Speed (µm/s)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Linear Distribution')

    # Distribution logarithmique
    log_speeds = np.log10(speeds[speeds > 0])

    axes[1].hist(log_speeds, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('log10(Speed)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Log Distribution')

    plt.savefig(path)
    plt.tight_layout()
    plt.show()

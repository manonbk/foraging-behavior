"""
A02 - HOW LONG do swims and stops last?

    I.   Histograms of stop / swim durations from find_stops (A01's stops.csv / swims.csv).
    II.  Stops read in the speed table (A01's speed_table.csv): survival and hazard.
    III. Markov stops: durations of state == 0 episodes in the tracking file
         (needs a "state" column, e.g. from the HMM fits; skipped if absent).

Survival / hazard (plot_stop_dynamics) is the same for II and III, only the way the durations are
extracted differs.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trajectories.magic_loading import magic_load_trajectories
from trajectories.trajectory_analysis import scale_lengths, segments_from_table

from .common import setup, load_speed_table, bin_duration


#
# PARTIE 0. HISTOGRAMMES DES DURÉES (stops / swims de find_stops)
#

def histogram_swim_stop_durations(stops, swims, unit="min", only_complete=True, path=None):
    """Plot histograms of stop and swimming durations.

    only_complete=True drops censored intervals (touching the start or end of their sequence):
    their duration is limited by the sequence, not by the behaviour. This biases the histogram
    toward short durations when bouts are long compared with sequences (the longest bouts are
    the ones most often censored), so compare with only_complete=False and mind the sequence
    lengths after concatenation.
    """

    if unit not in ("s", "min"):
        raise ValueError("unit must be 's' or 'min'")
    factor = 60 if unit == "min" else 1
    keep = lambda d: not (only_complete and d.get("censored", False))
    stop_durations = np.array([d["duration"] / factor for d in stops if keep(d)])
    swim_durations = np.array([d["duration"] / factor for d in swims if keep(d)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, dur, msg in ((ax1, stop_durations, "No complete stops"),
                         (ax2, swim_durations, "No complete swims")):
        if len(dur) > 0:
            ax.hist(dur, bins=np.linspace(dur.min(), dur.max(), min(120, max(10, len(dur)))),
                    edgecolor="black")
        else:
            ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(f"Duration ({unit})")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", alpha=0.5)

    tag = "complete only" if only_complete else "all"
    ax1.set_title(f"Stop durations (n={len(stop_durations)}, {tag})")
    ax2.set_title(f"Swimming durations (n={len(swim_durations)}, {tag})")
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    return fig, (ax1, ax2)


#
# PARTIE I. LES STOPS DE A01 (table de vitesse)
#

def stop_durations(df):
    """Durée (en nombre de bins) de chaque épisode de stop complet.
    Le premier et le dernier bout de chaque séquence sont exclus (tronqués)."""
    durations = []
    for _, seq in df.groupby("seq_index"):
        seq = seq.sort_values("t")
        bouts = seq.groupby("bout", sort=False).agg(
            stop=("stop", "first"),
            n=("t", "size"),
        )
        bouts = bouts.iloc[1:-1]                   # on enlève les bouts tronqués
        durations.extend(bouts.loc[bouts["stop"].astype(bool), "n"])
    return np.array(durations, dtype=int)

#
# PARTIE II. LES STOPS MARKOVIENS
#


#
# PARTIE II. LES STOPS MARKOVIENS
#

def stop_state_durations(segments):
    """Durée (en frames, entiers) de chaque arrêt complet (state == 0).
    Le premier et le dernier épisode de chaque segment sont exclus (tronqués)."""
    durations = []
    for seg in segments:
        state = seg.sort_values("frame")["state"].to_numpy()
        new_bout = np.r_[True, state[1:] != state[:-1]]
        bout = np.cumsum(new_bout) - 1
        n = np.bincount(bout)
        first_state = state[new_bout]
        for k in range(1, len(n) - 1):
            if first_state[k] == 0:
                durations.append(n[k])
    return np.array(durations, dtype=int)


#
# SURVIE / HASARD (commun aux parties I et II)
#

def plot_stop_dynamics(n, dt, min_at_risk=20, logx=True, logy=True):
    """
    n  : durées des stops en nombre de bins/frames (entiers)
    dt : durée d'un bin/frame (s)
    Plot :
        1. fraction of stops lasting longer vs stop duration
        2. Probability of leaving vs time already spent in the stop
    """
    N = len(n)
    k = np.arange(1, n.max() + 1)
    n_leave = np.bincount(n, minlength=k[-1] + 1)[1:]
    at_risk = N - np.cumsum(n_leave) + n_leave
    t = k * dt

    surv = 1 - np.cumsum(n_leave) / N
    hazard = n_leave / at_risk

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ok = surv > 0
    axes[0].plot(t[ok], surv[ok], "o-", ms=3)
    axes[0].set_xlabel("stop duration (s)")
    axes[0].set_ylabel("fraction of stops lasting longer")
    axes[0].set_title(f"Survival ({N} stops)")

    ok = at_risk >= min_at_risk
    axes[1].plot(t[ok], hazard[ok], "o-", ms=3)
    axes[1].set_xlabel("time already spent in the stop (s)")
    axes[1].set_ylabel("probability of leaving (per bin)")
    axes[1].set_title(f"Hazard (at risk ≥ {min_at_risk})")

    for ax in axes:
        # une échelle log n'est possible que s'il y a des valeurs > 0 (ex: moins de min_at_risk stops -> axe vide)
        pos = lambda get: any(np.any(np.asarray(get(l), dtype=float) > 0) for l in ax.lines)
        if logx and pos(lambda l: l.get_xdata()): ax.set_xscale("log")
        if logy and pos(lambda l: l.get_ydata()): ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    return fig

#


#
# MAIN
#

def survival_hazard(n, dt, name, plot_dir):
    if len(n) == 0:
        print(f"{name}: no complete stop found, nothing to plot")
        return
    print(f"{name}: {len(n)} stops, dt = {dt:.4f} s, max duration = {n.max() * dt:.2f} s")
    fig = plot_stop_dynamics(n, dt)
    fig.savefig(f"{plot_dir}/{name}", dpi=200)


if __name__ == "__main__":
    cfg = setup()
    PLOT_DIR, FPS, PIXEL_SIZE_UM = cfg.plot_dir, cfg.fps, cfg.pixel_size_um

    # 0. histograms of durations (stops.csv / swims.csv written by A01)
    stops_path, swims_path = f"{PLOT_DIR}/stops.csv", f"{PLOT_DIR}/swims.csv"
    if os.path.exists(stops_path) and os.path.exists(swims_path):
        stops = pd.read_csv(stops_path).to_dict("records")
        swims = pd.read_csv(swims_path).to_dict("records")
        histogram_swim_stop_durations(stops, swims, only_complete=True, unit="min",
                                      path=f"{PLOT_DIR}/swim_stop_durations.png")
        histogram_swim_stop_durations(stops, swims, only_complete=False, unit="min",
                                      path=f"{PLOT_DIR}/swim_stop_durations_uncensored.png")
    else:
        print("stops.csv / swims.csv not found: run A01_swim_stop.py first. Histograms skipped.")

    # I. survival / hazard of the stops of the speed table
    tab = load_speed_table(PLOT_DIR)
    survival_hazard(stop_durations(tab), bin_duration(tab), "stop_dynamics.png", PLOT_DIR)

    # II. survival / hazard of the Markov stops (needs a 'state' column in the tracking file)
    df = magic_load_trajectories(cfg.filename, link=False)
    if "state" in df.columns:
        scale_lengths(df, PIXEL_SIZE_UM, FPS)
        n = stop_state_durations(segments_from_table(df))
        survival_hazard(n, 1 / FPS, "stop_states_dynamics.png", PLOT_DIR)
    else:
        print("No 'state' column in the tracking file: Markov stops skipped.")

    plt.show()

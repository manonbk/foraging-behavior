"""
A01 - DETECT swims and stops.

Computes the binned speed (speed_core.speed_sequences), glues segments separated by short detection
gaps, finds stops / swims (speed_core.find_stops) and writes, in <data folder>/movement_analysis/:
    speed_table.csv   one row per speed observation (seq_index, id, t, v, logv, stop, swim, bout)
    stops.csv         one row per detected stop  (start, end, duration, censored, ...)
    swims.csv         one row per detected swim
All the other scripts (A02, A03, A04) read these files.

Overview plots:
    - speed x time with stops in red   (plot_speed_sequences)
    - histogram of speed               (plot_binned_speed)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from trajectories.magic_loading import magic_load_trajectories
from trajectories.trajectory_analysis import scale_lengths

from .common import setup, SPEED_TABLE_NAME
from .speed_core import (BIN_FRAMES, speed_sequences, concatenate_segments, find_stops,
                         build_speed_table)

MAX_GAP_S = 20.0  # detection gaps shorter than this are glued (filled with NaN speed)
COMPARE_BINS = False  # True: also compare several bin sizes (plots only)


# ---------------------------------------------------
# PLOTS
# ------------------------------------------------------

### AMO'S SPEED x TIME + stops in red (adapted)

def plot_speed_sequences(seqs, stops, dt, bin_frames, window_min=2.0, 
                         max_gap_min=1.0, path=None, scale="linear"):
    """Plot rolling mean speed, grouping sequences separated by short gaps."""

    if scale not in ("linear", "log"):
        raise ValueError("scale must be 'linear' or 'log'")
    if window_min <= 0 or max_gap_min < 0:
        raise ValueError("window_min must be positive and max_gap_min non-negative")

    bin_duration = bin_frames * dt
    window_bins = max(1, int(round(window_min * 60 / bin_duration)))
    max_gap_sec = max_gap_min * 60

    # Prepare and sort non-empty sequences by starting time
    sequences = []
    for seq in seqs:
        v = np.asarray(seq["v"], dtype=float)
        if len(v) == 0:
            continue

        t = seq["t0"] + np.arange(len(v)) * bin_duration
        sequences.append({
            "t": t,
            "v": v,
            "end": t[-1] + bin_duration,
        })

    sequences.sort(key=lambda s: s["t"][0])

    # Group sequences separated by at most max_gap_min
    groups = []
    for seq in sequences:
        if (not groups or seq["t"][0] - groups[-1]["end"] > max_gap_sec):
            groups.append({
                "times": [seq["t"]],
                "speeds": [seq["v"]],
                "end": seq["end"]})
        else:
            groups[-1]["times"].append(seq["t"])
            groups[-1]["speeds"].append(seq["v"])
            groups[-1]["end"] = max(groups[-1]["end"], seq["end"])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Shade detected stops
    for stop in stops:
        ax.axvspan(
            stop["start"] / 60,
            stop["end"] / 60,
            color="lightcoral",
            alpha=0.75,
            linewidth=0,
            zorder=0,
        )

    # Calculate and plot rolling means separately for each group
    for group in groups:
        v = np.concatenate(group["speeds"])
        t = np.concatenate(group["times"])

        if len(v) < window_bins:
            continue

        valid = np.isfinite(v)
        values = np.where(valid, v, 0.0)
        cumsum = np.concatenate(([0.0], np.cumsum(values)))
        counts = np.concatenate(([0], np.cumsum(valid)))
        rolling_sum = cumsum[window_bins:] - cumsum[:-window_bins]
        rolling_count = counts[window_bins:] - counts[:-window_bins]
        rolling_speed = np.divide(
            rolling_sum,
            rolling_count,
            out=np.full(len(rolling_sum), np.nan),
            where=rolling_count > 0,
        )

        # Use the actual times at both ends of each window
        rolling_time = (
            t[:len(rolling_speed)]
            + t[window_bins - 1:window_bins - 1 + len(rolling_speed)]
        ) / 2
        plot_valid = np.isfinite(rolling_speed)
        if scale == "log":
            plot_valid &= rolling_speed > 0
        ax.plot(
            rolling_time[plot_valid] / 60,
            rolling_speed[plot_valid],
            color="black",
            linewidth=1.5,
            zorder=2,
        )

    ax.set_yscale(scale)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean speed (µm/s)")
    ax.set_title(f"Rolling mean speed ({window_min:g}-min window)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color="black", linewidth=1.5, label="Rolling mean speed")]

    if len(stops) > 0:
        handles.append(Patch(
                facecolor="lightcoral",
                edgecolor="none",
                alpha=0.75,
                label="Detected stops",
            ))

    ax.legend(handles=handles, loc="best", frameon=False)
    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    return fig, ax


def plot_binned_speed(seqs, gaps=None, path=None, floor=None):
    """Plot the distribution of speed (linear and log10 scales)."""

    # Concatenate all sequences
    z = np.concatenate([s['logv'][s['logv'] < 3] for s in seqs])
    z = z[np.isfinite(z)]

    # Recover original speeds
    v = np.concatenate([s['v'][s['v'] < 1000] for s in seqs])
    v = v[np.isfinite(v)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 1. Linear distribution
    bins_v = np.linspace(v.min(), v.max(), 100)

    axes[0].hist(v, bins=bins_v, density=True,
                 label='All observations',alpha=0.7, edgecolor='black')

    if floor is not None:
        axes[0].axvline(floor, color='k', ls=':',
                        label='Speed floor')

    axes[0].set_xlabel('Speed (um/s)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Linear speed distribution')
    axes[0].legend(fontsize=8)

    # 2. Log distribution
    bins_z = np.linspace(z.min(), z.max(), 100)

    axes[1].hist(z, bins=bins_z, density=True,
                 label='All observations',alpha=0.7, edgecolor='black')

    if floor is not None:
        axes[1].axvline(np.log10(floor), color='k', ls=':',
                        label='Speed floor',alpha=0.7)

    axes[1].set_xlabel('log10(speed / (um/s))')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Log speed distribution')
    axes[1].legend(fontsize=8)

    plt.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    return fig


def compare_bins(df, dt, pixel_size_um, plot_dir, bin_list=None):
    """Speed plots for several bin sizes (no stop detection: stops are not drawn)."""
    if bin_list is None:
        bin_list = np.linspace(4, 30, 10).astype(int)
    for bin_frames in bin_list:
        floor = 0.1 * pixel_size_um / (bin_frames * dt)
        seqs = speed_sequences(df, dt, bin_frames, floor=floor)
        plot_speed_sequences(seqs, [], dt, bin_frames, scale="linear",
                             path=f"{plot_dir}/speed_x_time_{bin_frames}_bins.png")
        plot_speed_sequences(seqs, [], dt, bin_frames, scale="log",
                             path=f"{plot_dir}/speed_log_x_time_{bin_frames}_bins.png")
        plot_binned_speed(seqs, floor=floor, path=f"{plot_dir}/log_speed_histogram_{bin_frames}_bins.png")
        plt.close("all")


# ---------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":
    """ STEP 1 : GET DATA and scale lengths"""
    cfg = setup()
    FPS, PIXEL_SIZE_UM, PLOT_DIR = cfg.fps, cfg.pixel_size_um, cfg.plot_dir
    dt = 1 / FPS

    df = magic_load_trajectories(cfg.filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    """ STEP 2 : SPEED, GAPS, STOPS / SWIMS"""
    speed_floor = 0.1 * PIXEL_SIZE_UM / (BIN_FRAMES * dt)
    seqs = speed_sequences(df, dt, BIN_FRAMES, floor=speed_floor)
    # Glue segments separated by short detection gaps (NaN speed in the gap)
    seqs = concatenate_segments(seqs, dt, BIN_FRAMES, max_gap_s=MAX_GAP_S)
    stops, swims = find_stops(seqs, dt, BIN_FRAMES)

    """ STEP 3 : SAVE (read by A02, A03, A04)"""
    # One table, one row per speed observation: id, seq_index, t, v, logv, swim, stop, bout
    tab = build_speed_table(seqs, stops, swims, dt, BIN_FRAMES)
    tab.to_csv(f"{PLOT_DIR}/{SPEED_TABLE_NAME}", index=False)
    pd.DataFrame(stops).to_csv(f"{PLOT_DIR}/stops.csv", index=False)
    pd.DataFrame(swims).to_csv(f"{PLOT_DIR}/swims.csv", index=False)

    """ STEP 4 : OVERVIEW PLOTS"""
    plot_speed_sequences(seqs, stops, dt, BIN_FRAMES, scale="linear", path=f"{PLOT_DIR}/speed_x_time.png")
    plot_speed_sequences(seqs, stops, dt, BIN_FRAMES, scale="log", path=f"{PLOT_DIR}/speed_log_x_time.png")
    plot_binned_speed(seqs, floor=speed_floor, path=f"{PLOT_DIR}/log_speed_histogram.png")

    if COMPARE_BINS:
        compare_bins(df, dt, PIXEL_SIZE_UM, PLOT_DIR)

    plt.show()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories
from trajectories.trajectory_analysis import scale_lengths

# ----------------------------------------------------------------------------
# PARAMETERS
# ----------------------------------------------------------------------------
JUMP_UM = 100          # fixed jump threshold in um per frame; None -> automatic (robust)
JUMP_MAD_K = 8          # automatic threshold = median + K * 1.4826 * MAD of displacement/frame
SIZE_JUMP_FRAC = 0.5    # relative change of 'length' between adjacent frames flagged as size jump (if you win/loose more than 50% of your length). Maybe should use width?
LOW_CONF = 0.5          # conf below this is "low confidence"
TIMELINE_BINS = 2000    # number of bins of the detection timeline


# ----------------------------------------------------------------------------
# CORE EVALUATION
# ----------------------------------------------------------------------------
def evaluate_tracking_quality(df, jump_um=JUMP_UM, jump_mad_k=JUMP_MAD_K,
                              size_jump_frac=SIZE_JUMP_FRAC):
    """
    Compute simple tracking quality metrics.

    Returns
    -------
    metrics : dict          (one scalar per metric -> one row of the summary table)
    details : dict of data  (counts per frame, segments, gaps, pair table with jumps)
    """
    df = df.sort_values("frame").reset_index(drop=True)

    # ---- Number of detections per frame ------------------------------------------
    counts = df.groupby("frame").size()
    f0, f1 = int(counts.index.min()), int(counts.index.max())
    counts = counts.reindex(np.arange(f0, f1 + 1), fill_value=0)
    n_frames = len(counts)
    n0 = int((counts == 0).sum())
    n1 = int((counts == 1).sum())
    nm = int((counts > 1).sum())

    # ---- Segments (contiguous frames with >= 1 detection) and gaps ------
    present = counts.index[counts > 0].values
    breaks = np.where(np.diff(present) > 1)[0]
    starts = np.r_[present[0], present[breaks + 1]]
    ends = np.r_[present[breaks], present[-1]]
    seg_len = ends - starts + 1
    segments = pd.DataFrame({"start": starts, "end": ends, "length": seg_len})

    gap_start = present[breaks] + 1
    gap_len = present[breaks + 1] - present[breaks] - 1
    gaps = pd.DataFrame({"start": gap_start, "length": gap_len})

    # ---- Jumps between consecutive frames with exactly 1 detection -----
    single = df[df["frame"].isin(counts.index[counts == 1])].sort_values("frame")
    fr = single["frame"].values
    x, y = single["x"].values, single["y"].values
    L = single["length"].values

    dframe = np.diff(fr)
    disp = np.hypot(np.diff(x), np.diff(y))
    disp_per_frame = disp / dframe
    size_change = np.abs(np.diff(L)) / np.maximum(L[:-1], 1e-9)

    pairs = pd.DataFrame({
        "frame_from": fr[:-1], "frame_to": fr[1:], "dframe": dframe,
        "x_from": x[:-1], "y_from": y[:-1], "x_to": x[1:], "y_to": y[1:],
        "disp": disp, "disp_per_frame": disp_per_frame,
        "size_change": size_change,
    })

    adj = pairs["dframe"] == 1
    if jump_um is None:
        ref = pairs.loc[adj, "disp_per_frame"].values
        if len(ref) > 0:
            med = np.median(ref)
            mad = np.median(np.abs(ref - med))
            thr = med + jump_mad_k * 1.4826 * mad
        else:
            thr = np.nan
    else:
        thr = jump_um
    pairs["jump"] = pairs["disp_per_frame"] > thr
    pairs["size_jump"] = pairs["size_change"] > size_jump_frac
    pairs["jump_and_size"] = pairs["jump"] & pairs["size_jump"]

    # ---- Confidence -----------------------------------------
    conf = df["conf"] if "conf" in df else pd.Series(dtype=float)

    # ---- 5. Summary ---------------------------------------------------------
    metrics = {
        "n_frames_total": n_frames,
        "n_detections": len(df),
        "n_frames_0det": n0,
        "n_frames_1det": n1,
        "n_frames_multidet": nm,
        "frac_0det": n0 / n_frames,
        "frac_1det": n1 / n_frames,
        "frac_multidet": nm / n_frames,
        "n_segments": len(segments),
        "seg_len_median": float(np.median(seg_len)),
        "seg_len_max": int(seg_len.max()),
        "frac_in_longest_segment": seg_len.max() / n_frames,
        "n_gaps": len(gaps),
        "gap_len_mean": float(gaps["length"].mean()) if len(gaps) else 0.0,
        "gap_len_max": int(gaps["length"].max()) if len(gaps) else 0,
        "jump_threshold_um_per_frame": float(thr),
        "n_jumps_adjacent": int((pairs["jump"] & adj).sum()),
        "n_jumps_across_gap": int((pairs["jump"] & ~adj).sum()),
        "n_size_jumps": int((pairs["size_jump"] & adj).sum()),
        "n_jump_and_size": int((pairs["jump_and_size"] & adj).sum()),
        "disp_per_frame_median": float(np.median(disp_per_frame[adj])) if adj.any() else np.nan,
        "conf_mean": float(conf.mean()) if len(conf) else np.nan,
        "frac_low_conf": float((conf < LOW_CONF).mean()) if len(conf) else np.nan,
    }
    details = {"counts": counts, "segments": segments, "gaps": gaps, "pairs": pairs,
               "thr": thr}
    return metrics, details

 
# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
def plot_quality(details, label, out_dir):
    counts, segments, gaps, pairs, thr = (details[k] for k in
                                          ("counts", "segments", "gaps", "pairs", "thr"))
    jumps = pairs[pairs["jump"]]
 
    # Fig 1: timeline + displacement per frame
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    # Binned fractions instead of one bar per frame (fast even for ~600k frames)
    n_bins = int(min(TIMELINE_BINS, len(counts)))
    bin_id = np.arange(len(counts)) * n_bins // len(counts)
    bin_size = np.bincount(bin_id, minlength=n_bins)
    frac0 = np.bincount(bin_id, weights=(counts.values == 0), minlength=n_bins) / bin_size
    fracm = np.bincount(bin_id, weights=(counts.values > 1), minlength=n_bins) / bin_size
    centers = counts.index.values[0] + (np.arange(n_bins) + 0.5) * len(counts) / n_bins
 
    ax[0].fill_between(centers, 0, frac0, step="mid", color="tab:red", linewidth=0,
                       label="no detection")
    if fracm.any():  # only draw / list in the legend if there really are >1 detection frames
        ax[0].fill_between(centers, frac0, frac0 + fracm, step="mid", color="tab:orange",
                           linewidth=0, label=">1 detection")
    ax[0].set_ylim(0, 1)

    ax[0].set_ylabel("fraction of frames\nper bin")
    ax[0].legend(loc="upper right", ncol=2, fontsize=8)
    ax[0].set_title(f"{label}\nMissing / multiple detections over time")
 
    ax[1].plot(pairs["frame_to"], pairs["disp_per_frame"], ".", ms=3, color="k",
               rasterized=True)
    ax[1].plot(jumps["frame_to"], jumps["disp_per_frame"], "o", mfc="none", color="r",
               label=f"jumps ({len(jumps)})")
    ax[1].axhline(thr, color="r", ls="--", lw=0.8, label="threshold")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("frame")
    ax[1].set_ylabel("displacement per frame (um)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_timeline.png"), dpi=150)
    plt.close(fig)
 
    # Fig 3: segment and gap length distributions
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].hist(segments["length"], bins=30)
    ax[0].set_xlabel("segment length (frames)"); ax[0].set_yscale("log")
    if len(gaps):
        ax[1].hist(gaps["length"], bins=30)
    ax[1].set_xlabel("gap length (frames)"); ax[1].set_yscale("log")
    adjp = pairs[pairs["dframe"] == 1]
    ax[2].hist(adjp["disp_per_frame"], bins=60)
    ax[2].axvline(thr, color="r", ls="--")
    ax[2].set_xlabel("displacement per frame (um)"); ax[2].set_yscale("log")
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{label}_distributions.png"), dpi=150)
    plt.close(fig)
 


# ----------------------------------------------------------------------------
# SUMMARY TABLE (one row per run, to compare trackers / preprocessing)
# ----------------------------------------------------------------------------
def update_summary(metrics, label, summary_path):
    row = pd.DataFrame([{"label": label, **metrics}]).set_index("label")
    if os.path.exists(summary_path):
        print("adding quality tests to existing file..")
        old = pd.read_csv(summary_path, sep="\t", index_col="label")
        old = old.drop(index=label, errors="ignore")  # overwrite same label
        row = pd.concat([old, row])
    row.to_csv(summary_path, sep="\t")
    return row


def compare_summaries(paths):
    """Concatenate several quality_summary.tsv (e.g. from different recordings)."""
    return pd.concat([pd.read_csv(p, sep="\t", index_col="label") for p in paths])


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    """ STEP 1 : GET DATA and scale lengths"""
    filename = ask_open_tracking_file()
    FPS, PIXEL_SIZE_UM = guess_or_ask_fps_and_pixelsize(filename)
    dt = 1 / FPS
    filepath, ext = os.path.splitext(filename)
    folder = os.path.dirname(os.path.abspath(filename))

    PLOT_DIR = os.path.abspath(f"{folder}/quality_evaluation")
    os.makedirs(PLOT_DIR, exist_ok=True)

    df = magic_load_trajectories(filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    label = os.path.splitext(os.path.basename(filename))[0]
    print("Evaluating tracking quality")
    metrics, details = evaluate_tracking_quality(df)

    print("Saving data..")
    update_summary(metrics, label, os.path.join(PLOT_DIR, "quality_summary.tsv"))
    plot_quality(details, label, PLOT_DIR)
    # details["pairs"][details["pairs"]["jump"]].to_csv(
    #         os.path.join(PLOT_DIR, f"{label}_jumps.tsv"), sep="\t", index=False)
    # details["gaps"].to_csv(os.path.join(PLOT_DIR, f"{label}_gaps.tsv"), sep="\t", index=False)

    summary = pd.read_csv(os.path.join(PLOT_DIR, "quality_summary.tsv"),
                          sep="\t", index_col="label")
    print(summary.T.to_string())
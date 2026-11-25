"""
Core computations of the swim / stop analysis (no I/O, no plots).

    speed_sequences      -> speed per bin of BIN_FRAMES frames, one sequence per contiguous segment
    concatenate_segments -> glue segments separated by short detection gaps (NaN speed in the gap)
    find_stops           -> stops (sliding windows) and swims (their complement)
    build_speed_table    -> one row per speed observation: seq_index, id, t, v, logv, stop, swim, bout
    add_speed_columns_to_frames -> copy v_binned / swim / stop onto the per-frame tracking table
"""

import numpy as np
import pandas as pd

from trajectories.trajectory_analysis import segments_from_table

# ---------------------------------------------------
# CONSTANTS
# ------------------------------------------------------

BIN_FRAMES = 4  # average position over BIN_FRAMES frames to compute speed

WINDOW_SEC = 10  # window length to verify stop
STEP_SEC = 5  # how long until computing new window (overlapping here)
SPEED_ZERO_THRESH = 20.0  # speed threshold under which you consider it stopped
ZERO_FRACTION_SHADE = 0.6  # fraction of time in the window that you have to spend under SPEED_ZERO_THRESH to consider it a stop
MIN_VALID_FRACTION = 0.5  # how much of the window has to contain a valid detection to be computed


# ---------------------------------------------------
# COMPUTE SPEED
# ------------------------------------------------------

def speed_sequences(table, dt, bin_frames=BIN_FRAMES, min_bins=5, floor=None):
    """
    Cuts `table` into contiguous segments, then computes for each one the speed (um/s)
    from NON-overlapping blocks of `bin_frames` frames:
        v_k = || mean(pos in block k+1) - mean(pos in block k) || / (bin_frames * dt)
    `floor` (um/s): exact zeros (quantised detections) are clipped so that log is defined.

    Returns a list of dicts: t0 (s, time of first observation), v (um/s), logv (log10).
    Each observation is bin_frames*dt seconds long.
    """
    seqs, n_obs, n_clip = [], 0, 0
    for seg in segments_from_table(table):                    # from trajectory_analysis.py
        x, y = seg['x'].to_numpy(), seg['y'].to_numpy()
        nid=seg["id"].iloc[0]
        n = len(x) // bin_frames
        if n - 1 < min_bins:
            continue
        mx = x[:n * bin_frames].reshape(n, bin_frames).mean(1)
        my = y[:n * bin_frames].reshape(n, bin_frames).mean(1)
        v = np.hypot(np.diff(mx), np.diff(my)) / (bin_frames * dt)
        if floor is not None:
            n_clip += int((v < floor).sum())
            v = np.maximum(v, floor)
        n_obs += len(v)
        seqs.append(dict(t0=(seg['frame'].iloc[0] + bin_frames) * dt, v=v, logv=np.log10(v), id=nid))
    print(f"  {len(seqs)} sequences, {n_obs} observations of {bin_frames * dt:.2f} s "
          f"({100 * n_clip / max(n_obs, 1):.2f} % clipped at the speed floor)")
    return seqs


# ---------------------------------------------------
# CONCATENATE SEGMENTS SEPARATED BY SHORT DETECTION GAPS
# ------------------------------------------------------

def concatenate_segments(seqs, dt, bin_frames=BIN_FRAMES, max_gap_s=10.0):
    """Glue consecutive segments of the same id when the detection gap between them is at most
    `max_gap_s` seconds. The gap is filled with NaN speeds (one NaN per missing observation),
    so each glued group is ONE regular time series, with holes = NaN.

    Done BEFORE find_stops on purpose: the state (swim / stop) of a segment is only known
    through windows of WINDOW_SEC seconds, so a segment shorter than a window has no state, and a stop cut in two
    by a hole is missed. Glued first, find_stops sees the whole bout (it ignores NaN bins).
    Gaps longer than `max_gap_s` are NOT filled (they may hide a long stop).

    Bins after a gap are placed on the time grid of the first segment of the group
    (time error <= bin_duration / 2).
    """
    bd = bin_frames * dt
    by_id = {}
    for sq in seqs:
        by_id.setdefault(sq["id"], []).append(sq)

    out, kept, cut = [], [], []

    def close(nid, t0, parts):
        v = np.concatenate(parts)
        out.append(dict(t0=t0, v=v, logv=np.log10(v), id=nid))

    for nid, segs in by_id.items():
        segs = sorted(segs, key=lambda q: q["t0"])
        t0, parts, n = segs[0]["t0"], [np.asarray(segs[0]["v"], float)], len(segs[0]["v"])
        for q in segs[1:]:
            gap = q["t0"] - (t0 + n * bd)          # end of last observation -> start of next one
            if gap <= max_gap_s:
                n_fill = max(0, int(round(gap / bd)))
                parts += [np.full(n_fill, np.nan), np.asarray(q["v"], float)]
                n += n_fill + len(q["v"])
                kept.append(gap)
            else:
                close(nid, t0, parts)
                cut.append(gap)
                t0, parts, n = q["t0"], [np.asarray(q["v"], float)], len(q["v"])
        close(nid, t0, parts)

    n_nan = sum(int(np.isnan(q["v"]).sum()) for q in out)
    n_all = sum(len(q["v"]) for q in out)
    print(f"  concatenate_segments: {len(seqs)} -> {len(out)} sequences; "
          f"{len(kept)} gaps filled with NaN ({100 * n_nan / max(n_all, 1):.1f} % of bins), "
          f"{len(cut)} gaps > {max_gap_s:g} s left as cuts")
    allg = np.array(kept + cut)
    if len(allg):
        q = np.percentile(allg, [25, 50, 75, 90])
        print(f"  gap duration quantiles 25/50/75/90 % = "
              f"{q[0]:.1f} / {q[1]:.1f} / {q[2]:.1f} / {q[3]:.1f} s, max {allg.max():.0f} s")
    return out


# ---------------------------------------------------
# FIND STOPS
# ------------------------------------------------------

def find_stops(seqs,dt,bin_frames=BIN_FRAMES,window_sec=WINDOW_SEC,step_sec=STEP_SEC,
               speed_zero_thresh=SPEED_ZERO_THRESH,zero_fraction_shade=ZERO_FRACTION_SHADE,
               min_valid_fraction=MIN_VALID_FRACTION):
    """Detect stops and derive swimming intervals.

    Times and durations are in seconds.
    Returns (stops, swims), each a list of dictionaries.

    NaN speeds (detection gaps filled by concatenate_segments) are ignored; a window is only
    tested if at least `min_valid_fraction` of its bins are valid.
    Each interval has a 'censored' flag: True when it touches the start or the end of its
    sequence, so its true duration is unknown (it is only a lower bound). Leave censored
    intervals out of duration histograms.
    """

    if window_sec <= 0 or step_sec <= 0:
        raise ValueError("window_sec and step_sec must be positive")
    if speed_zero_thresh <= 0:
        raise ValueError("speed_zero_thresh must be positive")
    if not 0 <= zero_fraction_shade <= 1:
        raise ValueError("zero_fraction_shade must be between 0 and 1")

    bin_duration = bin_frames * dt

    stops = []
    swims = []

    for seq_index, seq in enumerate(seqs):
        nid=seq["id"]
        v = np.asarray(seq["v"], dtype=float)
        if len(v) == 0:
            continue

        t = seq["t0"] + np.arange(len(v)) * bin_duration
        valid = np.isfinite(v)

        # Observed interval, consistent with the speed plot
        seq_start = t[0]
        seq_end = t[-1] + bin_duration

        # Find windows satisfying the stop criterion
        windows = []
        current_start = seq_start

        while current_start + window_sec <= seq_end:
            current_end = current_start + window_sec

            mask = ((t >= current_start) & (t < current_end) & valid)
            window_speeds = v[mask]

            if (len(window_speeds) >= max(1, min_valid_fraction * window_sec / bin_duration)
                and np.mean(window_speeds < speed_zero_thresh) >= zero_fraction_shade ):
                windows.append({
                    "start": current_start,
                    "end": current_end,
                })

            current_start += step_sec

        # Merge overlapping stop windows
        merged = []

        for window in windows:
            if merged and window["start"] <= merged[-1]["end"]:
                merged[-1]["end"] = max(
                    merged[-1]["end"], window["end"]
                )
            else:
                merged.append(window.copy())

        # Store stops and derive swims between them
        previous_end = seq_start

        # swims touching the sequence limits are censored; for stops, being within one step of
        # a limit is enough (the neighbouring window that would show it continues cannot be tested)
        cens_swim = lambda a, b: bool(a <= seq_start + 1e-9 or b >= seq_end - 1e-9)

        for stop in merged:
            stop["seq_index"] = seq_index
            stop["duration"] = stop["end"] - stop["start"]
            stop["id"]=nid
            stop["censored"] = bool(stop["start"] - seq_start < step_sec
                                    or seq_end - stop["end"] < step_sec)
            stops.append(stop)

            if stop["start"] > previous_end:
                swims.append({
                    "start": previous_end,
                    "end": stop["start"],
                    "duration": stop["start"] - previous_end,
                    "seq_index": seq_index,
                    "id" : nid,
                    "censored": cens_swim(previous_end, stop["start"]),
                })

            previous_end = max(previous_end, stop["end"])

        # Swimming after the last stop, or the whole segment if no stops
        if previous_end < seq_end:
            swims.append({
                "start": previous_end,
                "end": seq_end,
                "duration": seq_end - previous_end,
                "seq_index": seq_index,
                "id" : nid,
                "censored": cens_swim(previous_end, seq_end),
            })

    stops.sort(key=lambda s: (s["seq_index"], s["start"]))
    swims.sort(key=lambda s: (s["seq_index"], s["start"]))

    return stops, swims


# ---------------------------------------------------
# BUILD ONE TABLE : 1 row = 1 speed observation (1 bin)
# ------------------------------------------------------

def _in_half_open(t, starts, ends):
    """True si start <= t < end pour un des intervalles (supposés disjoints, comme ceux de find_stops)."""
    t = np.asarray(t, dtype=float)
    if len(starts) == 0:
        return np.zeros(len(t), dtype=bool)
    order = np.argsort(starts)
    s = np.asarray(starts, dtype=float)[order]
    e = np.asarray(ends, dtype=float)[order]
    idx = np.searchsorted(s, t, side="right") - 1
    return (idx >= 0) & (t < e[np.clip(idx, 0, None)])


def build_speed_table(seqs, stops, swims, dt, bin_frames=BIN_FRAMES):
    """Un DataFrame 'long' : une ligne par observation de vitesse.

    Colonnes :
        seq_index : index du segment contigu dans `seqs` (unique, contrairement a `id`)
        id        : identifiant de la paramecie
        t         : temps (s) du debut de l'observation (meme convention que find_stops)
        v, logv   : vitesse (um/s) et log10(v), calculees sur blocs de `bin_frames` frames
        stop      : True si l'observation est dans un arret
        swim      : True si elle est dans une nage
        bout      : numero de l'episode (suite consecutive d'observations de meme etat
                    dans le meme segment) -> utile pour ne jamais traverser un arret
    """
    bin_duration = bin_frames * dt
    parts = []
    for seq_index, seq in enumerate(seqs):
        n = len(seq["v"])
        if n == 0:
            continue
        parts.append(pd.DataFrame({
            "seq_index": seq_index,
            "id": seq["id"],
            "t": seq["t0"] + np.arange(n) * bin_duration,
            "v": np.asarray(seq["v"], dtype=float),
            "logv": np.asarray(seq["logv"], dtype=float),
        }))
    tab = pd.concat(parts, ignore_index=True)

    stops_by_seq = {k: g for k, g in pd.DataFrame(stops).groupby("seq_index")} if len(stops) else {}
    swims_by_seq = {k: g for k, g in pd.DataFrame(swims).groupby("seq_index")} if len(swims) else {}

    t_all = tab["t"].to_numpy()
    stop = np.zeros(len(tab), dtype=bool)
    swim = np.zeros(len(tab), dtype=bool)
    for seq_index, rows in tab.groupby("seq_index").indices.items():
        if seq_index in stops_by_seq:
            g = stops_by_seq[seq_index]
            stop[rows] = _in_half_open(t_all[rows], g["start"].to_numpy(), g["end"].to_numpy())
        if seq_index in swims_by_seq:
            g = swims_by_seq[seq_index]
            swim[rows] = _in_half_open(t_all[rows], g["start"].to_numpy(), g["end"].to_numpy())
    tab["stop"], tab["swim"] = stop, swim

    n_bad = int((stop == swim).sum())          # doit etre 0 : swim = complement de stop
    if n_bad:
        print(f"  WARNING: {n_bad} observations are neither swim nor stop (or both)")

    new_bout = (tab["seq_index"].ne(tab["seq_index"].shift())
                | tab["stop"].ne(tab["stop"].shift()))
    tab["bout"] = new_bout.cumsum() - 1

    print(f"  speed table: {len(tab)} rows, {100 * stop.mean():.1f} % in stops, "
          f"{tab['bout'].nunique()} bouts")
    return tab


def add_speed_columns_to_frames(df, tab, dt, bin_frames=BIN_FRAMES):
    """Recopie v_binned, swim, stop sur le DataFrame de tracking (1 ligne par frame).

    Chaque observation couvre `bin_frames` frames : ses valeurs sont donc dupliquees sur
    ces frames. Les frames sans observation (premier bloc de chaque segment, reste de la
    division par bin_frames) valent NaN / <NA>.
    A utiliser pour filtrer / colorer des positions, PAS pour calculer des autocorrelations
    ou des cartes de retour (utiliser `tab`, ou les lags seraient faux).
    """
    f0 = np.round(tab["t"].to_numpy() / dt).astype(int)
    frames = (f0[:, None] + np.arange(bin_frames)[None, :]).ravel()
    rep = lambda col: np.repeat(tab[col].to_numpy(), bin_frames)
    ext = pd.DataFrame({"id": rep("id"), "frame": frames, "v_binned": rep("v"),
                        "swim": rep("swim"), "stop": rep("stop")})
    stale = [c for c in ("v_binned", "swim", "stop") if c in df.columns]
    out = df.drop(columns=stale).merge(ext, on=["id", "frame"], how="left")
    out[["swim", "stop"]] = out[["swim", "stop"]].astype("boolean")
    return out

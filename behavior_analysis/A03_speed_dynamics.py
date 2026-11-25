"""
A03 - HOW does the speed change? Speed dynamics, all built on the table written by A01_swim_stop.py
(speed_table.csv). One row = one speed observation (one bin).
Columns: seq_index, id, t, v, logv, stop, swim, bout.

    1. find_transitions / plot_transitions : how long does the speed take to go swim <-> stop ?
    2. lagged_pairs / plot_return_map      : v(t+Delta) vs v(t), fixed points
    3. autocorrelation / plot_acf          : autocorrelation of the speed (swim only, stop only, all)

Rows are consecutive bins inside a sequence (seq_index), so shifting by n rows inside a
sequence is an exact lag of n * bin_duration. Nothing here ever crosses a sequence boundary.

Detection gaps glued by concatenate_segments are rows with NaN speed: they are kept in the time
grid and ignored by every average / correlation below.

scale="log" works on logv, scale="linear" on v (thresholds included). For the return map and the
autocorrelation the same switch is the `col` argument ("logv" or "v").
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

from .common import setup, load_speed_table, bin_duration, lagged, COL, col_of_scale, scale_of_col


# ---------------------------------------------------
# 1. TRANSITIONS SWIM <-> STOP
# ------------------------------------------------------
# Why not just align on the stop boundaries of find_stops ? Those boundaries come from 2-min
# windows moved every 30 s, so they are only accurate to tens of seconds. We use them to know
# WHERE to look (reach_s before/after the boundary, on the stop side), then measure the
# transition on the speed itself.

def speed_levels(tab, frac=(0.2, 0.8), scale="log"):
    """Typical speed of stops and swims (median of logv, or of v if scale='linear'), and the
    two crossing levels in between. lo/hi sit at `frac` of the gap (20-80 % by default: the
    transition time is measured between these two levels, like an engineering rise time)."""
    col = col_of_scale(scale)
    L_swim = tab.loc[tab["swim"], col].median()
    L_stop = tab.loc[tab["stop"], col].median()
    gap = L_swim - L_stop
    return dict(L_stop=L_stop, L_swim=L_swim,
                lo=L_stop + frac[0] * gap, hi=L_stop + frac[1] * gap)


def _transition_in_window(w, kind, lo, hi):
    """down : last 'fast' bin (>= hi), then first 'slow' bin (<= lo) after it.
       up   : last 'slow' bin (<= lo), then first 'fast' bin (>= hi) after it.
    Returns (i, j, 'ok') with positions in w, or (None, None, reason) with reason
      'no_start' : the window never contains the starting level
                   (down: no bin >= hi, up: no bin <= lo),
      'no_end'   : after the last starting-level bin, the end level is never reached
                   (down: never <= lo, up: never >= hi) before the window ends."""
    ref, tgt = (w >= hi, w <= lo) if kind == "down" else (w <= lo, w >= hi)
    i_all = np.flatnonzero(ref)
    if len(i_all) == 0:
        return None, None, "no_start"
    i = i_all[-1]
    j_all = np.flatnonzero(tgt[i + 1:])
    if len(j_all) == 0:
        return None, None, "no_end"
    return i, i + 1 + j_all[0], "ok"


def _smoothed(tab, smooth_bins, col="logv"):
    """Running median of `col` inside each sequence. Bins that are NaN stay NaN."""
    sm = np.array(tab.groupby("seq_index")[col]
                     .transform(lambda x: x.rolling(smooth_bins, center=True, min_periods=1).median()),
                  dtype=float)                           # np.array: writable copy
    sm[np.isnan(tab[col].to_numpy())] = np.nan
    return sm


def find_transitions(tab, frac=(0.2, 0.8), reach_s=60.0, margin_s=10.0, smooth_bins=3,
                     scale="log"):
    """One row per swim->stop ('down') and stop->swim ('up') boundary.

    The transition is searched in [boundary - margin, boundary + reach] for 'down' and
    [boundary - reach, boundary + margin] for 'up' (the true transition lies on the stop side
    of the boundary found by find_stops), restricted to the two bouts around the boundary.
    The speed is first smoothed by a running median of `smooth_bins` bins (1 = no smoothing).
    It removes isolated spikes but blurs the transition: smooth_bins * bin_duration should stay
    well below the durations you measure, so check the sensitivity (1, 3, 5). The smoothed
    signal is only used to FIND the transition; the plots show the raw speed.

    scale : 'log' (levels and crossings on logv) or 'linear' (on v). Run both to compare.

    A detection hole (NaN bins) between the start and the end of a transition does not discard
    it: its duration is then an upper bound (it includes the unobserved time) and `n_missing`
    counts the NaN bins in between. Use ev[ev.n_missing == 0] to keep fully observed ones.

    Columns: kind, seq_index, id, row_boundary, t_boundary, row_start, row_end, t_start,
    duration (NaN when no clean transition was found), n_missing, reason ('ok', 'no_start',
    'no_end', see _transition_in_window), row_lo / row_hi (rows of the search window).
    """
    bd = bin_duration(tab)
    col = col_of_scale(scale)
    lev = speed_levels(tab, frac, scale)
    raw = tab[col].to_numpy()
    R, M = int(round(reach_s / bd)), int(round(margin_s / bd))
    n = len(tab)

    s = _smoothed(tab, smooth_bins, col)
    seq, stop, bout = tab["seq_index"].to_numpy(), tab["stop"].to_numpy(), tab["bout"].to_numpy()
    t, ids = tab["t"].to_numpy(), tab["id"].to_numpy()

    change = np.r_[True, bout[1:] != bout[:-1]]          # bouts are contiguous runs of rows
    kb = np.cumsum(change) - 1                           # bout rank of each row
    bstart = np.flatnonzero(change)
    bend = np.r_[bstart[1:] - 1, n - 1]

    same = np.r_[False, seq[1:] == seq[:-1]]
    prev_stop = np.r_[False, stop[:-1]]
    boundaries = [("down", np.flatnonzero(same & stop & ~prev_stop)),
                  ("up",   np.flatnonzero(same & ~stop & prev_stop))]

    out = []
    for kind, rows in boundaries:
        for b in rows:
            lo, hi = (b - M, b + R) if kind == "down" else (b - R, b + M)
            lo = max(lo, bstart[kb[b - 1]])              # stay within previous bout ...
            hi = min(hi, bend[kb[b]]) + 1                # ... and current bout
            i, j, reason = _transition_in_window(s[lo:hi], kind, lev["lo"], lev["hi"])
            rec = dict(kind=kind, seq_index=seq[b], id=ids[b], row_boundary=b, t_boundary=t[b],
                       row_start=np.nan, row_end=np.nan, t_start=np.nan, duration=np.nan,
                       n_missing=np.nan, reason=reason, row_lo=lo, row_hi=hi)
            if i is not None:
                i, j = lo + i, lo + j
                rec.update(row_start=i, row_end=j, t_start=t[i], duration=(j - i) * bd,
                           n_missing=int(np.isnan(raw[i:j + 1]).sum()))
            out.append(rec)
    ev = pd.DataFrame(out)
    ev.attrs["levels"] = lev
    ev.attrs["smooth_bins"] = smooth_bins
    ev.attrs["scale"] = scale
    return ev


def summarize_transitions(ev):
    lev = ev.attrs.get("levels", {})
    if lev:
        unit = "log10(speed)" if ev.attrs.get("scale", "log") == "log" else "speed (um/s)"
        print(f"  {unit} levels: stop={lev['L_stop']:.2f}  lo={lev['lo']:.2f}  "
              f"hi={lev['hi']:.2f}  swim={lev['L_swim']:.2f}")
    for kind in ("down", "up"):
        e = ev[ev["kind"] == kind]
        d = e["duration"].dropna()
        if len(d):
            q1, q2, q3 = np.percentile(d, [25, 50, 75])
            n_gap = int((e["n_missing"] > 0).sum())
            print(f"  {kind:>4}: {len(d)}/{len(e)} resolved, duration median {q2:.3f} s "
                  f"(IQR {q1:.3f}-{q3:.3f}); {n_gap} with a detection hole inside")
        else:
            print(f"  {kind:>4}: 0/{len(e)} resolved")
        bad = e.loc[e["duration"].isna(), "reason"].value_counts()
        if len(bad):
            print("        unresolved: " + ", ".join(f"{k} = {v}" for k, v in bad.items()))


def event_triggered(tab, rows, pre_s, post_s, col="logv"):
    """Matrix (n_events, n_times) of `col` around each row in `rows`; rel = time axis (s).
    Events whose window leaves their sequence are dropped."""
    bd = bin_duration(tab)
    pre, post = int(round(pre_s / bd)), int(round(post_s / bd))
    offs = np.arange(-pre, post + 1)
    rows = np.asarray(rows, dtype=int)
    idx = rows[:, None] + offs[None, :]
    ok = (idx[:, 0] >= 0) & (idx[:, -1] < len(tab))
    idx, rows = idx[ok], rows[ok]
    seq = tab["seq_index"].to_numpy()
    ok = (seq[idx[:, 0]] == seq[rows]) & (seq[idx[:, -1]] == seq[rows])
    return offs * bd, tab[col].to_numpy()[idx[ok]]


def _mean_band(ax, rel, M, color, label):
    with np.errstate(invalid="ignore", divide="ignore"):
        cnt = np.isfinite(M).sum(0)
        m = np.nanmean(M, 0)
        sem = np.nanstd(M, 0) / np.sqrt(cnt)
    ax.plot(rel, m, color=color, label=f"{label} (n={len(M)})")
    ax.fill_between(rel, m - sem, m + sem, color=color, alpha=0.25, linewidth=0)


def plot_transitions(tab, ev, reach_s=60.0, path=None):
    """(a) average log-speed aligned on the find_stops boundary,
       (b) average aligned on the start of each measured transition,
       (c) distribution of transition durations."""
    colors = {"down": "tab:red", "up": "tab:blue"}
    names = {"down": "swim -> stop", "up": "stop -> swim"}
    lev = ev.attrs.get("levels", {})
    scale = ev.attrs.get("scale", "log")
    col = col_of_scale(scale)
    ylab = "log10(speed)" if scale == "log" else "speed (um/s)"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))

    for kind in ("down", "up"):
        e = ev[ev["kind"] == kind]
        if len(e) == 0:
            continue
        rel, M = event_triggered(tab, e["row_boundary"], 1.5 * reach_s, 1.5 * reach_s, col)
        if len(M):
            if len(M) <= 25:                             # few events: show each one
                ax[0].plot(rel, M.T, color=colors[kind], lw=0.5, alpha=0.3)
            _mean_band(ax[0], rel, M, colors[kind], names[kind])
        es = e.dropna(subset=["row_start"])
        if len(es):
            rel, M = event_triggered(tab, es["row_start"], 10.0, 15.0, col)
            if len(M):
                _mean_band(ax[1], rel, M, colors[kind], names[kind])
        d = e["duration"].dropna()
        if len(d):
            bins = np.linspace(0, np.percentile(d, 99), 40)
            ax[2].hist(d, bins=bins, alpha=0.55, color=colors[kind], label=names[kind])

    for a in ax[:2]:
        a.axvline(0, color="k", ls=":")
        if lev:
            a.axhline(lev["lo"], color="0.6", ls="--", lw=0.8)
            a.axhline(lev["hi"], color="0.6", ls="--", lw=0.8)
        a.set_ylabel(ylab)
        a.legend(frameon=False, fontsize=8)
    ax[0].set_xlabel("time from stop-detection boundary (s)")
    ax[0].set_title("Aligned on find_stops boundary (all events)")
    ax[1].set_xlabel("time from start of transition (s)")
    ax[1].set_title("Aligned on measured start (resolved events only)")
    ax[2].set_xlabel("transition duration (s)")
    ax[2].set_ylabel("count")
    ax[2].set_title(f"Duration (lo <-> hi level, {scale} scale)")
    ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig, ax


def plot_unresolved(tab, ev, max_events=12, path=None):
    """Look at the events for which no clean transition was found: raw (grey) and smoothed
    log-speed, search window shaded, 'lo' / 'hi' levels dashed, t = 0 at the find_stops boundary.
    Shows whether the real transition is outside the window, the speed never reaches a level,
    or a dip/hesitation breaks the 'last start-level bin' rule."""
    bad = ev[ev["duration"].isna()].head(max_events)
    if len(bad) == 0:
        print("  no unresolved events")
        return None
    lev = ev.attrs["levels"]
    scale = ev.attrs.get("scale", "log")
    col = col_of_scale(scale)
    s = _smoothed(tab, ev.attrs.get("smooth_bins", 1), col)
    raw, t, seq = tab[col].to_numpy(), tab["t"].to_numpy(), tab["seq_index"].to_numpy()
    ncol = 4
    nrow = int(np.ceil(len(bad) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow), squeeze=False)
    for ax_, (_, r) in zip(axes.ravel(), bad.iterrows()):
        lo, hi, b = int(r["row_lo"]), int(r["row_hi"]), int(r["row_boundary"])
        ext = max(hi - lo, int(round(90.0 / bin_duration(tab))))   # >= 90 s of context
        a = max(lo - ext, np.searchsorted(seq, r["seq_index"], "left"))
        z = min(hi + ext, np.searchsorted(seq, r["seq_index"], "right"))
        tt = t[a:z] - t[b]
        ax_.plot(tt, raw[a:z], color="0.7", lw=0.8)
        ax_.plot(tt, s[a:z], color="tab:red" if r["kind"] == "down" else "tab:blue", lw=1.2)
        ax_.axvspan(t[lo] - t[b], t[hi - 1] - t[b], color="y", alpha=0.12, lw=0)
        ax_.axvline(0, color="k", ls=":")
        ax_.axhline(lev["lo"], color="0.5", ls="--", lw=0.8)
        ax_.axhline(lev["hi"], color="0.5", ls="--", lw=0.8)
        ax_.set_title(f"{r['kind']}, seq {int(r['seq_index'])}: {r['reason']}", fontsize=8)
        ax_.tick_params(labelsize=7)
    for ax_ in axes.ravel()[len(bad):]:
        ax_.axis("off")
    fig.supxlabel("time from find_stops boundary (s)", fontsize=9)
    fig.supylabel("log10(speed)" if scale == "log" else "speed (um/s)", fontsize=9)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig


# ---------------------------------------------------
# 2. RETURN MAP : v(t + Delta) vs v(t)
# ------------------------------------------------------

def lagged_pairs(tab, n, col="logv"):
    """(x, y) = (col(t), col(t + n bins)), same sequence only."""
    x = tab[col]
    y = lagged(tab, col, n)
    ok = (x.notna() & y.notna()).to_numpy()
    return x.to_numpy()[ok], y.to_numpy()[ok]


def conditional_mean(x, y, bins=40, min_count=30):
    """E[y | x] on uniform bins of x (0.5-99.5 percentile range). Returns centers, mean, sem."""
    edges = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99.5), bins + 1)
    which = np.digitize(x, edges) - 1
    c, m, s = [], [], []
    for k in range(bins):
        sel = which == k
        if sel.sum() >= min_count:
            c.append(0.5 * (edges[k] + edges[k + 1]))
            m.append(y[sel].mean())
            s.append(y[sel].std() / np.sqrt(sel.sum()))
    return np.array(c), np.array(m), np.array(s)


def fixed_points(c, m, max_gap=1.5):
    """Zero crossings of m(c) - c (linear interpolation).
    Returns a list of (x*, slope, status):
      'stable'   : local slope of m is < 1,
      'unstable' : local slope of m is > 1,
      'no data'  : the crossing falls in a gap (the two bins around it are more than `max_gap`
                   bin-widths apart, because that region is almost never visited, which is
                   typical of the barrier between two states). It is only an interpolation."""
    d = m - c
    spacing = np.median(np.diff(c))
    out = []
    for k in np.flatnonzero(np.sign(d[:-1]) * np.sign(d[1:]) < 0):
        x_star = c[k] - d[k] * (c[k + 1] - c[k]) / (d[k + 1] - d[k])
        slope = (m[k + 1] - m[k]) / (c[k + 1] - c[k])
        if c[k + 1] - c[k] > max_gap * spacing:
            status = "no data"
        else:
            status = "stable" if slope < 1 else "unstable"
        out.append((x_star, slope, status))
    return out


def plot_return_map(tab, lags=(2, 5, 10, 25), col="logv", bins=40, path=None):
    """(a) 2D histogram of (x(t), x(t+Delta)) for the shortest lag, (b) E[x(t+Delta)|x(t)] for
    several Delta, (c) the drift E[x(t+Delta) - x(t) | x(t)] (its zeros = fixed points)."""
    bd = bin_duration(tab)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    x, y = lagged_pairs(tab, lags[0], col)
    h = ax[0].hist2d(x, y, bins=80, cmap="viridis", norm=LogNorm())
    lim = [x.min(), x.max()]
    ax[0].plot(lim, lim, "w--", lw=1)
    ax[0].set_title(f"{col}(t+Delta) vs {col}(t), Delta={lags[0] * bd:.2g} s")
    fig.colorbar(h[3], ax=ax[0], label="count")

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(lags)))
    results = {}
    for n, col_ in zip(lags, colors):
        x, y = lagged_pairs(tab, n, col)
        c, m, s = conditional_mean(x, y, bins)
        lab = f"Delta={n * bd:.2g} s"
        ax[1].errorbar(c, m, s, color=col_, marker="o", ms=3, lw=1, label=lab)
        ax[2].plot(c, m - c, color=col_, marker="o", ms=3, lw=1, label=lab)
        fps_ = fixed_points(c, m)
        results[n] = fps_
        for x_star, _, kind in fps_:
            if kind != "no data":
                ax[1].plot(x_star, x_star, marker="s" if kind == "stable" else "x",
                           color=col_, mfc="none", ms=9)
    ax[1].plot(lim, lim, "k--", lw=1)
    ax[2].axhline(0, color="k", ls="--", lw=1)
    ax[1].set_xlabel(f"{col}(t)")
    ax[1].set_ylabel(f"<{col}(t+Delta)>")
    ax[1].set_title("Conditional mean (squares = stable, x = unstable)")
    ax[2].set_xlabel(f"{col}(t)")
    ax[2].set_ylabel(f"<{col}(t+Delta) - {col}(t)>")
    ax[2].set_title("Drift")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")

    for n, fps_ in results.items():
        desc = ", ".join(f"{p:.2f} ({k}, slope {s:.2f})" for p, s, k in fps_) or "none"
        print(f"  Delta={n * bd:.2g} s: fixed points = {desc}")
    return fig, ax, results


# ---------------------------------------------------
# 3. AUTOCORRELATION
# ------------------------------------------------------

def autocorrelation(tab, col="logv", state="swim", max_lag_s=20.0, center="global",
                    max_slow_frac=0.05):
    """Autocorrelation pooled over segments, never crossing a state change.

    state : 'swim' or 'stop' -> only bins of that state, each bout kept as its own segment
            None             -> all bins, each sequence (seq_index) is one segment
    center: 'global' (one mean for everything), or 'id' (one mean per paramecium).
    max_slow_frac: (state='swim' only) drop swim bouts in which more than this fraction of bins
            is below the 'lo' speed level, i.e. bouts hiding a stop that find_stops missed.
            A few such bins dominate the variance and fake a very long correlation time.
            The number of dropped bouts is printed. Use None to keep everything.
    Returns lags (s), rho, cnt (number of pairs per lag), n_segments.
    """
    bd = bin_duration(tab)
    maxlag = int(round(max_lag_s / bd))
    if state is None:
        sub, group = tab, "seq_index"
    else:
        sub, group = tab[tab[state]], "bout"
        if state == "swim" and max_slow_frac is not None:
            lo = speed_levels(tab, scale=scale_of_col(col))["lo"]
            n_slow = (sub[col] < lo).groupby(sub["bout"]).sum()
            n_ok = sub[col].notna().groupby(sub["bout"]).sum()
            slow = n_slow / n_ok.clip(lower=1)
            drop = slow[slow > max_slow_frac].index
            if len(drop):
                print(f"  ACF: dropped {len(drop)}/{slow.size} swim bouts containing slow episodes "
                      f"(> {100 * max_slow_frac:.0f} % of bins below the 'lo' level)")
            sub = sub[~sub["bout"].isin(drop)]
    x = sub[col].to_numpy(dtype=float)
    if center == "global":
        x = x - np.nanmean(x)
    elif center == "id":
        x = x - sub.groupby("id")[col].transform("mean").to_numpy()
    else:
        raise ValueError("center must be 'global' or 'id'")

    num, cnt = np.zeros(maxlag + 1), np.zeros(maxlag + 1)
    segs = sub.reset_index(drop=True).groupby(group).indices
    for rows in segs.values():
        xs = x[rows]
        ok = np.isfinite(xs).astype(float)               # NaN bins (detection holes) are skipped
        xs = np.where(ok > 0, xs, 0.0)
        L = len(xs)
        for k in range(min(maxlag + 1, L)):
            num[k] += xs[:L - k] @ xs[k:]
            cnt[k] += ok[:L - k] @ ok[k:]                 # number of pairs with both bins valid
    rho = num / np.maximum(cnt, 1)
    rho = rho / rho[0]
    return np.arange(maxlag + 1) * bd, rho, cnt, len(segs)


def fit_exponential(lags, rho, cnt, fit_from=2, min_count=200):
    """Fit rho(lag) = A exp(-lag/tau) for lag index >= fit_from.
    A < 1 absorbs the white measurement noise that only affects lag 0.
    fit_from=2 skips lag 1, where the speeds of two successive bins share a block mean
    (correlated detection noise). Returns A, tau, tau_err."""
    sel = (np.arange(len(lags)) >= fit_from) & (cnt >= min_count)
    f = lambda t, A, tau: A * np.exp(-t / tau)
    p0 = (max(rho[sel][0], 0.05), max(lags[sel][-1] / 3, lags[1]))
    popt, pcov = curve_fit(f, lags[sel], rho[sel], p0=p0,
                           bounds=((0, 1e-6), (1.0, np.inf)))
    return popt[0], popt[1], np.sqrt(pcov[1, 1])


def plot_acf(tab, col="logv", max_lag_s=20.0, max_lag_all_s=300.0, center="global",
             max_slow_frac=0.05, min_count=200, path=None):
    """(a, b) ACF inside swim bouts (linear / semilog) with exponential fit,
       (c) ACF over all bins (dominated by the swim/stop switching, hence much longer lags)."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    panels = [("swim", max_lag_s, "tab:green", "swim only", ax[:2]),
              (None, max_lag_all_s, "k", "all bins", ax[2:])]
    for state, mlag, color, label, axes in panels:
        lags, rho, cnt, nseg = autocorrelation(tab, col, state, mlag, center, max_slow_frac)
        ok = cnt >= min_count
        for a in axes:
            a.plot(lags[ok], rho[ok], "o", ms=3, color=color, label=f"{label} ({nseg} segments)")
        if state is None:      # switching between two states: not a single exponential
            continue
        try:
            A, tau, err = fit_exponential(lags, rho, cnt, min_count=min_count)
            tt = np.linspace(lags[2], lags[ok][-1], 200)
            for a in axes:
                a.plot(tt, A * np.exp(-tt / tau), "-", color="tab:orange", lw=1.2,
                       label=f"fit: tau={tau:.3g} s")
            print(f"  {label}: tau = {tau:.3g} +/- {err:.2g} s  (A = {A:.2f})")
        except Exception as exc:
            print(f"  {label}: exponential fit failed ({exc})")
    ax[1].set_yscale("symlog", linthresh=0.05)
    titles = ["Swim: linear", "Swim: semilog (exponential = line)", "All bins"]
    for a, title in zip(ax, titles):
        a.axhline(0, color="0.5", lw=0.8)
        a.set_xlabel("lag (s)")
        a.set_ylabel(f"autocorrelation of {col}")
        a.set_title(title)
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
        print(f"Saved: {path}")
    return fig, ax


# ---------------------------------------------------
# MAIN
# ------------------------------------------------------

if __name__ == "__main__":
    cfg = setup(need_scales=False)
    PLOT_DIR = cfg.plot_dir

    tab = load_speed_table(PLOT_DIR)
    print(f"{len(tab)} observations of {bin_duration(tab):.3f} s, "
          f"{tab['seq_index'].nunique()} sequences")

    for scale in ("log", "linear"):
        col = COL[scale]
        print(f"=== scale: {scale}")

        print("1. Transitions")
        ev = find_transitions(tab, scale=scale)
        ev.to_csv(f"{PLOT_DIR}/transitions_{scale}.csv", index=False)
        summarize_transitions(ev)
        plot_transitions(tab, ev, path=f"{PLOT_DIR}/transitions_{scale}.png")
        plot_unresolved(tab, ev, path=f"{PLOT_DIR}/transitions_unresolved_{scale}.png")

        print("2. Return map")
        plot_return_map(tab, lags=(2, 5, 10, 25), col=col, path=f"{PLOT_DIR}/return_map_{scale}.png")

        print("3. Autocorrelation")
        plot_acf(tab, col=col, max_lag_s=20.0, path=f"{PLOT_DIR}/acf_{scale}.png")

    plt.show()

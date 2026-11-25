#!/usr/bin/env python3
"""
Analyse du "retour" d'une paramécie vers son point de départ :
  - condition "stop" : départ = sortie d'un arrêt (stop -> plus stop)
  - condition "swim" : départ = point au milieu d'une nage établie (contrôle)

Pour chaque départ et chaque horizon, on regarde si la trajectoire
  1) s'éloigne au-delà de r_out  (left)
  2) puis repasse en dessous de r_in du point de départ (returned)

Les IC sont obtenus par bootstrap sur les id (pas sur les départs).

"""
import argparse

import numpy as np
import pandas as pd

from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories
from trajectories.trajectory_analysis import scale_lengths, segments_from_table
import os


# -------------------------------
# CONSTANTS
# -------------------------------

R_IN=100
R_OUT=200
SMOOTH=4
MIN_STOP=10 #minimum de durée dun stop, s
MIN_SWIM=10 #minimum de durée dun swim, s
SWIM_STEP=10 #espacement des départs, s
horizons=[10,20,30,50,100] #temps de trajectoire a analyser, s
mode="time"
N_BOOT=200
SEED=0
OUT="return_results"
block_seconds=100
cluster="segment"
N_PERM=5000
MIN_REMAINING=100  # s : durée min d'enregistrement restante après un départ (limite le biais de censure ; 0 = désactivé)


# --------------------------------------------------------------------------
# Préparation
# --------------------------------------------------------------------------
def iter_pieces(segments):
    """Recoupe chaque segment aux éventuels trous de frame."""
    for seg in segments:
        seg = seg.sort_values("frame")
        frame = seg["frame"].to_numpy()
        breaks = np.where(np.diff(frame) != 1)[0] + 1
        for idx in np.split(np.arange(len(seg)), breaks):
            if len(idx) > 1:
                yield seg.iloc[idx]


def smooth(a, win):
    if win <= 1:
        return a.astype(float)
    return (
        pd.Series(a).rolling(win, center=True, min_periods=1).median().to_numpy()
    )


def find_stop_exits(stop, m):
    """Indices i où stop[i] est False et stop[i-m:i] est entièrement True."""
    return [
        i for i in range(m, len(stop)) if (not stop[i]) and stop[i - m : i].all()
    ]


def find_swim_starts(swim, m, step):
    """Indices en pleine nage (swim vrai sur m frames avant), espacés de `step`."""
    out, last = [], -10**9
    for i in range(m, len(swim)):
        if swim[i - m : i + 1].all() and i - last >= step:
            out.append(i)
            last = i
    return out


def window_metrics(x, y, i, end, r_in, r_out):
    d = np.hypot(x[i + 1 : end + 1] - x[i], y[i + 1 : end + 1] - y[i])
    out = np.where(d > r_out)[0]
    if len(out) == 0:
        return dict(left=False, returned=False, dmin_after=np.nan, dmax=d.max())
    after = d[out[0] :]
    return dict(
        left=True,
        returned=bool((after < r_in).any()),
        dmin_after=float(after.min()),
        dmax=float(d.max()),
    )


# --------------------------------------------------------------------------
# Extraction des départs
# --------------------------------------------------------------------------
def process_piece(piece, fps, seg_idx=0):
    x = smooth(piece["x"].to_numpy(), SMOOTH)
    y = smooth(piece["y"].to_numpy(), SMOOTH)
    swim = piece["swim"].to_numpy(bool)
    stop = piece["stop"].to_numpy(bool)
    pid = piece["id"].iloc[0]
    frames = piece["frame"].to_numpy()
    n = len(x)
    block_f = max(1, int(round(block_seconds * fps)))

    step = np.hypot(np.diff(x), np.diff(y))
    cum = np.concatenate([[0.0], np.cumsum(step)])  # longueur parcourue cumulée

    m_stop = max(1, int(round(MIN_STOP * fps)))
    m_swim = max(1, int(round(MIN_SWIM * fps)))
    s_step = max(1, int(round(SWIM_STEP * fps)))

    starts = [("stop", i) for i in find_stop_exits(stop, m_stop)]
    starts += [("swim", i) for i in find_swim_starts(swim, m_swim, s_step)]

    rows = []
    for cond, i in starts:
        for h in horizons:
            if mode == "time":
                end = i + max(1, int(round(h * fps)))
            else:
                end = int(np.searchsorted(cum, cum[i] + h))
            if end >= n or end <= i:
                continue  # fenêtre incomplète : on n'inclut pas (même critère pour les 2 conditions)
            r = window_metrics(x, y, i, end, R_IN, R_OUT)
            if cluster == "id":
                cl = str(pid)
            elif cluster == "segment":
                cl = f"{pid}_seg{seg_idx}"
            else:  # blocs de temps
                cl = f"{pid}_blk{int(frames[i]) // block_f}"
            rows.append(
                dict(condition=cond, id=pid, cluster=cl, frame=int(frames[i]),
                     horizon=h, **r)
            )
    return rows


# --------------------------------------------------------------------------
# Statistiques
# --------------------------------------------------------------------------
def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def wilson(k, n, z=1.96):
    """IC de Wilson pour une proportion k/n. Ne s'effondre pas quand k=0 ou k=n."""
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_diff(p1, l1, u1, p2, l2, u2):
    """IC de la différence p1-p2 à partir des IC de Wilson (Newcombe)."""
    d = p1 - p2
    return (d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def union(a_lo, a_hi, b_lo, b_hi):
    los = [v for v in (a_lo, b_lo) if np.isfinite(v)]
    his = [v for v in (a_hi, b_hi) if np.isfinite(v)]
    return (min(los) if los else np.nan, max(his) if his else np.nan)


def pct(x):
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) else (np.nan, np.nan)


def stratified_perm_test(d, n_perm, rng):
    """
    Test de permutation stratifié par cluster, sur les départs 'partis' (left=True).
    H0 : le type de départ (stop/swim) n'influence pas la probabilité de retour.
    On permute les étiquettes stop/swim *à l'intérieur de chaque cluster* et on
    ne garde que les clusters qui contiennent les deux types : on compare donc
    des stops et des nages proches dans le temps (même individu, même zone).
    Renvoie (p-value, nb de clusters mixtes).
    """
    strata = []   # (retours 0/1 du cluster, étiquettes stop observées)
    for _, g in d.groupby("cluster"):
        is_stop = (g["condition"] == "stop").to_numpy()
        if is_stop.any() and (~is_stop).any():
            strata.append((g["returned"].to_numpy(float), is_stop))
    if not strata:
        return np.nan, 0
    n_stop = sum(int(m.sum()) for _, m in strata)
    n_swim = sum(int((~m).sum()) for _, m in strata)
    tot = sum(r.sum() for r, _ in strata)

    def stat(s_stop):   # P(retour|stop) - P(retour|swim), sur les clusters mixtes
        return s_stop / n_stop - (tot - s_stop) / n_swim

    obs = stat(sum(r[m].sum() for r, m in strata))
    count = 0
    for _ in range(n_perm):
        s = 0.0
        for r, m in strata:
            s += r[rng.choice(len(r), size=int(m.sum()), replace=False)].sum()
        if abs(stat(s)) >= abs(obs) - 1e-12:
            count += 1
    return (1 + count) / (n_perm + 1), len(strata)


def summarize(df, n_boot, rng, n_perm=5000):
    rows = []
    for h, dh in df.groupby("horizon"):
        ids = dh["cluster"].unique()
        counts = {}
        for cond in ("stop", "swim"):
            g = (
                dh[dh.condition == cond]
                .groupby("cluster")
                .agg(left=("left", "sum"), ret=("returned", "sum"), n=("left", "size"))
                .reindex(ids, fill_value=0)
            )
            counts[cond] = g[["ret", "left", "n"]].to_numpy(float)
        draws = rng.integers(0, len(ids), size=(n_boot, len(ids)))

        res = {}
        for cond in ("stop", "swim"):
            c = counts[cond]
            k, nl = c[:, 0].sum(), c[:, 1].sum()
            point = float(ratio(k, nl))
            boot = ratio(c[draws, 0].sum(axis=1), c[draws, 1].sum(axis=1))
            blo, bhi = pct(boot)                 # bootstrap par cluster (peut s'effondrer à 0 ou 1)
            wlo, whi = wilson(k, nl)             # Wilson (ne s'effondre pas, mais suppose l'indépendance)
            lo, hi = union(blo, bhi, wlo, whi)   # on affiche le plus prudent des deux
            res[cond] = dict(p=point, lo=lo, hi=hi, wl=wlo, wh=whi, boot=boot,
                             n=int(c[:, 2].sum()), nleft=int(nl),
                             fl=float(ratio(nl, c[:, 2].sum())),
                             ncl=int((c[:, 2] > 0).sum()))
        S, W = res["stop"], res["swim"]
        diff = S["p"] - W["p"]
        blo, bhi = pct(S["boot"] - W["boot"])
        nlo, nhi = newcombe_diff(S["p"], S["wl"], S["wh"], W["p"], W["wl"], W["wh"])
        dlo, dhi = union(blo, bhi, nlo, nhi)
        p_perm, n_mixed = stratified_perm_test(dh[dh["left"]], n_perm, rng)
        rows.append(
            dict(
                horizon=h,
                n_stop=S["n"], n_stop_left=S["nleft"], n_clusters_stop=S["ncl"],
                n_swim=W["n"], n_swim_left=W["nleft"], n_clusters_swim=W["ncl"],
                frac_left_stop=S["fl"], frac_left_swim=W["fl"],
                p_return_stop=S["p"], stop_lo=S["lo"], stop_hi=S["hi"],
                p_return_swim=W["p"], swim_lo=W["lo"], swim_hi=W["hi"],
                diff=diff, diff_lo=dlo, diff_hi=dhi,
                p_perm=p_perm, n_clusters_mixtes=n_mixed,
            )
        )
    return pd.DataFrame(rows).sort_values("horizon")


def plot(summary, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for cond, col in (("stop", "tab:red"), ("swim", "tab:blue")):
        p = summary[f"p_return_{cond}"]
        err = [p - summary[f"{cond}_lo"], summary[f"{cond}_hi"] - p]
        ax.errorbar(summary["horizon"], p, yerr=err, marker="o", capsize=3,
                    color=col, label=f"départ {cond}")
    for _, r in summary.iterrows():
        ax.annotate(f"n={int(r.n_stop_left)}", (r.horizon, r.p_return_stop),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7, color="tab:red")
    ax.set_xlabel("horizon (s)" if mode == "time" else "longueur parcourue")
    ax.set_ylabel("P(retour | parti)")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)

# --------------------------------------------------------------------------
# P(s'arrêter au même endroit), sur tout l'enregistrement (même id, segments ignorés)
# --------------------------------------------------------------------------
def _stop_bouts(frame, stop, x, y, m):
    """Arrêts = séries d'au moins m frames consécutives avec stop vrai (sans trou de détection).
    Renvoie (indice de début, x médian, y médian) de chaque arrêt."""
    bouts, n, i = [], len(frame), 0
    while i < n:
        if stop[i]:
            j = i
            while j + 1 < n and stop[j + 1] and frame[j + 1] - frame[j] == 1:
                j += 1
            if j - i + 1 >= m:
                bouts.append((i, float(np.median(x[i : j + 1])), float(np.median(y[i : j + 1]))))
            i = j + 1
        else:
            i += 1
    return bouts


def stop_return_events(table, fps, r_in=R_IN, r_out=R_OUT, min_remaining=MIN_REMAINING):
    """
    Pour chaque départ (sortie d'arrêt 'stop' ou point de nage établie 'swim'), regarde TOUT
    le reste de l'enregistrement du même id (on peut changer de segment) :
      - left          : la paramécie s'est éloignée à plus de r_out
      - passed_near   : après ça, elle repasse à moins de r_in (sans forcément s'arrêter)
      - stopped_there : après ça, elle fait un arrêt (>= MIN_STOP) dont la position est à moins de r_in
      - stopped_there_skip : pareil, mais en ignorant le PREMIER arrêt qui suit le départ (évite de compter
                             un redémarrage bref suivi d'un nouvel arrêt, ou un flag stop qui clignote)
      - delay_s / delay_skip_s : temps (s) entre le départ et le début de cet arrêt de retour
    """
    table = table.dropna(subset=["id"])
    m_stop = max(1, int(round(MIN_STOP * fps)))
    m_swim = max(1, int(round(MIN_SWIM * fps)))
    s_step = max(1, int(round(SWIM_STEP * fps)))
    rows = []
    for pid, g in table.groupby("id"):
        pieces = list(iter_pieces([g]))   # lissage fait par morceau continu
        if not pieces:
            continue
        frame = np.concatenate([p["frame"].to_numpy() for p in pieces])
        x = np.concatenate([smooth(p["x"].to_numpy(), SMOOTH) for p in pieces])
        y = np.concatenate([smooth(p["y"].to_numpy(), SMOOTH) for p in pieces])
        stop = np.concatenate([p["stop"].to_numpy(bool) for p in pieces])
        swim = np.concatenate([p["swim"].to_numpy(bool) for p in pieces])
        n = len(frame)

        # départs : mêmes critères que précédemment, mais en exigeant des frames consécutives
        starts = [("stop", i) for i in range(m_stop, n)
                  if (not stop[i]) and stop[i - m_stop : i].all() and frame[i] - frame[i - m_stop] == m_stop]
        last = -10**12
        for i in range(m_swim, n):
            if (swim[i - m_swim : i + 1].all() and frame[i] - frame[i - m_swim] == m_swim
                    and frame[i] - last >= s_step):
                starts.append(("swim", i))
                last = frame[i]

        bouts = _stop_bouts(frame, stop, x, y, m_stop)
        b_idx = np.array([b[0] for b in bouts], dtype=int)
        b_x = np.array([b[1] for b in bouts])
        b_y = np.array([b[2] for b in bouts])

        for cond, i in starts:
            if (frame[-1] - frame[i]) / fps < min_remaining:
                continue   # pas assez d'enregistrement restant (même critère pour les 2 conditions)
            d = np.hypot(x[i + 1 :] - x[i], y[i + 1 :] - y[i])
            out = np.flatnonzero(d > r_out)
            left = len(out) > 0
            passed = stopped = stopped_skip = False
            delay = delay_skip = np.nan
            if left:
                j0 = i + 1 + out[0]                       # première frame au-delà de r_out
                passed = bool((d[out[0] :] < r_in).any())
                if len(b_idx):
                    ok = (b_idx > j0) & (np.hypot(b_x - x[i], b_y - y[i]) < r_in)
                    if ok.any():
                        stopped = True
                        delay = (frame[b_idx[ok][0]] - frame[i]) / fps
                    after = np.flatnonzero(b_idx > i)          # arrêts qui commencent après le départ
                    if len(after):
                        ok_skip = ok.copy()
                        ok_skip[after[0]] = False              # on ignore le premier d'entre eux
                        if ok_skip.any():
                            stopped_skip = True
                            delay_skip = (frame[b_idx[ok_skip][0]] - frame[i]) / fps
            rows.append(dict(condition=cond, id=pid, frame=int(frame[i]), left=left,
                             passed_near=passed, stopped_there=stopped, delay_s=delay,
                             stopped_there_skip=stopped_skip, delay_skip_s=delay_skip))
    return pd.DataFrame(rows)


def summarize_stop_return(events, col="stopped_there"):
    """P(arrêt au même endroit | parti), par type de départ, avec IC de Wilson (et erreur standard)."""
    rows = []
    for cond in ("stop", "swim"):
        e = events[events["condition"] == cond] if len(events) else events
        el = e[e["left"]] if len(e) else e
        n = len(el)
        k = int(el[col].sum()) if n else 0
        kp = int(el["passed_near"].sum()) if n else 0
        p = k / n if n else np.nan
        lo, hi = wilson(k, n)
        rows.append(dict(
            condition=cond, n_starts=len(e), n_left=n, n_ids=int(el["id"].nunique()) if n else 0,
            k_stopped_there=k, p_stop_there=p,
            se=float(np.sqrt(p * (1 - p) / n)) if n else np.nan, ci_lo=lo, ci_hi=hi,
            p_pass_near=kp / n if n else np.nan,
        ))
    return pd.DataFrame(rows)


def print_return_times(events):
    """Temps médian (s) entre le départ et le nouvel arrêt au même endroit (parmi les départs qui y reviennent).
    Attention : calculé seulement sur les retours observés (conditionné au retour, et limité par la fin de
    l'enregistrement), donc plutôt sous-estimé pour les retours lents."""
    for label, scol, dcol in (("tous les arrêts", "stopped_there", "delay_s"),
                              ("sans le 1er arrêt", "stopped_there_skip", "delay_skip_s")):
        print(f"\nTemps pour s'arrêter à nouveau au même endroit ({label}) :")
        for cond in ("stop", "swim"):
            el = events[(events["condition"] == cond) & events["left"]]
            e = el[el[scol]]
            if len(e) == 0:
                print(f"  départ {cond:4s} : aucun retour (sur {len(el)} départs)")
                continue
            q1, med, q3 = np.percentile(e[dcol], [25, 50, 75])
            print(f"  départ {cond:4s} : médiane {med:.1f} s  (Q1-Q3 : {q1:.1f}-{q3:.1f} s ; {len(e)} retours sur {len(el)} départs)")


def plot_stop_return(summary, path, error="wilson"):
    """error='wilson' (IC 95 %, conseillé) ou 'se' (écart type de la proportion, sqrt(p(1-p)/n))."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.5, 4))
    labels = {"stop": "départ stop", "swim": "départ swim"}
    for xpos, (_, r) in enumerate(summary.iterrows()):
        if error == "se":
            yerr = [[r.se], [r.se]]
        else:
            yerr = [[r.p_stop_there - r.ci_lo], [r.ci_hi - r.p_stop_there]]
        ax.errorbar(xpos, r.p_stop_there, yerr=yerr, marker="o", capsize=4,
                    color="tab:red" if r.condition == "stop" else "tab:blue")
        ax.annotate(f"n={int(r.n_left)}", (xpos, r.p_stop_there), textcoords="offset points",
                    xytext=(10, 4), fontsize=8)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([labels[c] for c in summary["condition"]])
    ax.set_xlim(-0.5, len(summary) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(s'arrêter au même endroit)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def run_stop_return(table, fps, plot_dir):
    events = stop_return_events(table, fps)
    if events.empty:
        print("Stop-return : aucun départ valide (MIN_REMAINING trop grand ?)")
        return
    events.to_csv(f"{plot_dir}/{OUT}_stop_return_events.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        for tag, col, label in (("all", "stopped_there", "tous les arrêts"),
                                ("skip_first", "stopped_there_skip", "sans le 1er arrêt")):
            summ = summarize_stop_return(events, col)
            summ.to_csv(f"{plot_dir}/{OUT}_stop_return_summary_{tag}.csv", index=False)
            print(f"\nP(arrêt au même endroit) - {label}")
            print(summ)
            plot_stop_return(summ, f"{plot_dir}/{OUT}_stop_return_plot_{tag}.png")
    print_return_times(events)


def main():
    filename = ask_open_tracking_file()
    FPS, PIXEL_SIZE_UM = guess_or_ask_fps_and_pixelsize(filename)
    folder = os.path.dirname(os.path.abspath(filename))
    PLOT_DIR = os.path.abspath(f"{folder}/return_analysis")
    os.makedirs(PLOT_DIR, exist_ok=True)
    table = magic_load_trajectories(filename, link=False)
    scale_lengths(table, PIXEL_SIZE_UM, FPS)

    run_stop_return(table, FPS, PLOT_DIR)

    segments = segments_from_table(table)
    rows = []
    for k, piece in enumerate(iter_pieces(segments)):
        rows += process_piece(piece, FPS, k)
    if not rows:
        raise SystemExit("Aucun départ valide trouvé (horizons trop longs ? min-stop trop grand ?)")

    df = pd.DataFrame(rows)
    df.to_csv(f"{PLOT_DIR}/{OUT}_events.csv", index=False)

    rng = np.random.default_rng(SEED)
    summary = summarize(df, N_BOOT, rng, N_PERM)
    summary.to_csv(f"{PLOT_DIR}/{OUT}_summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(summary)

    try:
        plot(summary, f"{PLOT_DIR}/{OUT}_plot.png")
    except ImportError:
        print("matplotlib absent : pas de figure")


if __name__ == "__main__":
    main()
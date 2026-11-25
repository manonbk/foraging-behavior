"""
HMM maison sur la vitesse, avec K = 2 ou 3 états (argument K).

  K=2 : arret (Rayleigh)                          | nage (Normale)
  K=3 : arret1, arret2 (deux Rayleigh, sigma croissants) | nage (Normale)

Convention : les K-1 premiers états sont de type Rayleigh (vitesse = 0 + bruit de mesure),
le dernier est l'état de nage : vitesse ~ Normale(mu_N + beta * tau, sigma_N),
avec tau = temps normalisé dans [0, 1] (la vitesse moyenne de nage varie linéairement).

Paramètres ajustés en maximisant directement la vraisemblance (algorithme forward).
Valeurs manquantes : toutes les lignes (triées par frame) forment UNE séquence.
 - vitesse NaN  -> émission = 1 pour tous les états (aucune information sur l'état)
 - frames sans détection -> la chaîne avance de `gap` pas : transition A^gap
   (si le trou dépasse max_gap frames, on repart de la loi stationnaire)

Pour ajouter un autre K ou un autre type d'état, il suffit de modifier la partie « Modèle »
(STATE_NAMES, unpack, emission_logprob, initial_theta).
"""
import os
import json
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.stats import norm, rayleigh
import matplotlib.pyplot as plt

from .HMM_fit import plot_speed_vs_states
from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories, magic_save_trajectories
from trajectories.trajectory_analysis import scale_lengths

EPS = 1e-6  # évite log(0) pour une vitesse exactement nulle

STATE_NAMES = {
    2: ['arret', 'nage'],
    3: ['arret1', 'arret2', 'nage'],
}


# ------------------------------------------------------------------
# Modèle : seule cette partie dépend de K
# ------------------------------------------------------------------
# theta = [ log-incréments des sigma Rayleigh (K-1) , mu_N, log sigma_N, beta,
#           logits de transition hors-diagonale (K*(K-1), ligne par ligne) ]
#
#  - sigma Rayleigh : sigma_1 = exp(theta[0]), sigma_2 = sigma_1 + exp(theta[1]), ...
#    -> toujours croissants, donc pas d'échange d'étiquettes entre arret1 et arret2.
#  - transitions : softmax sur chaque ligne (diagonale de logit 0). Contrairement à des
#    sigmoïdes indépendantes, ça garantit que chaque ligne est une vraie loi de probabilité.

def transition_matrix(logits, K):
    L = np.zeros((K, K))
    L[~np.eye(K, dtype=bool)] = logits  # hors-diagonale, ligne par ligne
    E = np.exp(L - L.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def stationary(A):
    """Distribution stationnaire : pi A = pi, sum(pi) = 1."""
    K = len(A)
    M = A.T - np.eye(K)
    M[-1] = 1.0
    b = np.zeros(K)
    b[-1] = 1.0
    pi = np.clip(np.linalg.solve(M, b), 1e-300, None)
    return pi / pi.sum()


def unpack(theta, K):
    n_ray = K - 1
    sigmas = np.cumsum(np.exp(theta[:n_ray]))  # sigma Rayleigh croissants
    mu_N, sigma_N, beta = theta[n_ray], np.exp(theta[n_ray + 1]), theta[n_ray + 2]
    A = transition_matrix(theta[n_ray + 3:], K)
    return sigmas, mu_N, sigma_N, beta, A, stationary(A)


def emission_logprob(x, tau, theta, K):
    """log p(vitesse | état), tableau (n_obs, K).
    Vitesse NaN -> log p = 0 pour tous les états (émission constante, aucune information)."""
    sigmas, mu_N, sigma_N, beta, _, _ = unpack(theta, K)
    missing = np.isnan(x)
    xc = np.maximum(np.where(missing, 1.0, x), EPS)
    logB = np.empty((len(x), K))
    for k, s in enumerate(sigmas):  # états Rayleigh
        logB[:, k] = np.log(xc) - 2 * np.log(s) - xc**2 / (2 * s**2)
    logB[:, K - 1] = norm.logpdf(xc, mu_N + beta * tau, sigma_N)  # nage : normale à moyenne linéaire
    logB[missing] = 0.0
    return logB


def initial_theta(xv, K):
    """Initialisation à partir des quantiles des vitesses (xv = vitesses sans NaN)."""
    bands = {2: [(0.0, 0.3)], 3: [(0.0, 0.3), (0.0, 0.3)]}[K]  # bandes « lentes » -> Rayleigh
    sig = []
    for lo_q, hi_q in bands:
        sel = xv[(xv >= np.quantile(xv, lo_q)) & (xv <= np.quantile(xv, hi_q))]
        sig.append(np.sqrt(np.mean(sel**2) / 2))  # MLE Rayleigh
    log_incr = [np.log(sig[0])]
    for i in range(1, len(sig)):
        log_incr.append(np.log(max(sig[i] - sig[i - 1], 0.2 * sig[i - 1])))
    hi = xv[xv >= np.quantile(xv, 0.5)]
    return np.r_[log_incr,
                 np.median(hi),            # mu_N
                 np.log(hi.std()),         # sigma_N
                 0.0,                      # beta (pas de dérive au départ)
                 np.full(K * (K - 1), -3.5)]  # probas de transition ~ 3 % par frame


# ------------------------------------------------------------------
# Algorithmes HMM génériques (forward et Viterbi), accélérés par numba
# Une seule séquence ; gidx[t] = indice dans la table des puissances de A
# pour passer de l'observation t-1 à t (0 = on repart de la loi stationnaire).
# ------------------------------------------------------------------
def transition_powers(A, pi, max_gap):
    """Table P[g] = A^g pour g = 1..max_gap ; P[0] = matrice « stationnaire » (lignes = pi),
    utilisée pour les trous trop longs (on repart de la loi stationnaire)."""
    K = len(pi)
    P = np.empty((max_gap + 1, K, K))
    P[0] = np.tile(pi, (K, 1))
    P[1] = A
    for g in range(2, max_gap + 1):
        P[g] = P[g - 1] @ A
    return P


@njit(cache=True)
def forward_loglik(logB, gidx, Apow, pi):
    N, K = logB.shape
    total = 0.0
    alpha = np.zeros(K)
    for t in range(N):
        b = logB[t]
        m = b.max()
        new = np.zeros(K)
        for j in range(K):
            if t == 0:
                prior = pi[j]
            else:
                prior = 0.0
                for i in range(K):
                    prior += alpha[i] * Apow[gidx[t], i, j]
            new[j] = prior * np.exp(b[j] - m)
        s = new.sum()
        alpha = new / s
        total += np.log(s) + m
    return total


@njit(cache=True)
def viterbi(logB, gidx, logApow, logpi):
    N, K = logB.shape
    delta = np.empty((N, K))
    psi = np.zeros((N, K), np.int64)
    delta[0] = logpi + logB[0]
    for t in range(1, N):
        for j in range(K):
            best = -np.inf
            arg = 0
            for i in range(K):
                v = delta[t - 1, i] + logApow[gidx[t], i, j]
                if v > best:
                    best = v
                    arg = i
            delta[t, j] = best + logB[t, j]
            psi[t, j] = arg
    states = np.zeros(N, np.int64)
    s = np.argmax(delta[N - 1])
    states[N - 1] = s
    for t in range(N - 2, -1, -1):
        s = psi[t + 1, s]
        states[t] = s
    return states


# ------------------------------------------------------------------
# Fit
# ------------------------------------------------------------------
def fit_hmm_homemade(df, K=2, dt=1.0, save_dir=None, max_gap=10):
    """Ajuste le modèle à K états (2 ou 3), ajoute 'state' (0..K-1) et 'state_name' au df.
    max_gap : nombre max de frames d'écart entre deux lignes consécutives pour lesquelles
    on fait « avancer » la chaîne (A^gap). Au-delà, on repart de la loi stationnaire.
    max_gap=1 : séquences indépendantes (équivalent à découper aux trous).
    Renvoie (df, params)."""
    assert K in STATE_NAMES, f"K doit être dans {list(STATE_NAMES)}"
    d = df.sort_values('frame')
    frames = d['frame'].to_numpy(dtype=float)
    x = d['speed'].to_numpy(dtype=float)  # peut contenir des NaN
    gaps = np.diff(frames).astype(np.int64)
    assert (gaps >= 1).all(), "frames dupliquées : une seule ligne par frame attendue"

    # indice de transition pour chaque ligne (la 1re ligne n'en utilise pas)
    gidx = np.r_[0, np.where(gaps <= max_gap, gaps, 0)].astype(np.int64)

    print(f"K = {K} états")
    print(f"{len(x)} lignes dont {int(np.isnan(x).sum())} sans vitesse (NaN)")
    print(f"trous : {int(((gaps > 1) & (gaps <= max_gap)).sum())} comblés (<= {max_gap} frames), "
          f"{int((gaps > max_gap).sum())} plus longs (redémarrage)")

    # Temps normalisé dans [0, 1] : beta = variation totale de vitesse de nage sur l'enregistrement
    frame0 = frames.min()
    span = frames.max() - frames.min()
    tau = (frames - frame0) / span

    def neg_loglik(theta):
        _, _, _, _, A, pi = unpack(theta, K)
        Apow = transition_powers(A, pi, max_gap)
        return -forward_loglik(emission_logprob(x, tau, theta, K), gidx, Apow, pi) / len(x)

    xv = x[~np.isnan(x)]
    res = minimize(neg_loglik, initial_theta(xv, K), method='L-BFGS-B')
    theta = res.x
    sigmas, mu_N, sigma_N, beta, A, pi = unpack(theta, K)
    loglik = -res.fun * len(x)
    bic = len(theta) * np.log(len(xv)) - 2 * loglik

    params = {
        'K': K,
        'state_names': STATE_NAMES[K],
        'sigma_rayleigh_um_s': sigmas.tolist(),   # un par état d'arrêt
        'mu_N_um_s': mu_N,                        # vitesse de nage moyenne au début
        'sigma_N_um_s': sigma_N,
        'beta_um_s_total': beta,                  # variation de la vitesse de nage sur tout l'enregistrement
        'beta_um_s_per_min': beta / (span * dt / 60),
        'frame0': frame0,
        'frame_span': span,
        'max_gap_frames': max_gap,
        'transition_matrix_per_frame': A.tolist(),
        'mean_durations_s': (dt / (1 - np.diag(A))).tolist(),
        'stationary_distribution': pi.tolist(),
        'log_likelihood': loglik,
        'BIC': bic,
        'converged': bool(res.success),
    }
    print(json.dumps(params, indent=2))

    # États les plus probables (Viterbi) : y compris sur les lignes à vitesse NaN
    Apow = transition_powers(A, pi, max_gap)
    states = viterbi(emission_logprob(x, tau, theta, K), gidx, np.log(Apow), np.log(pi))
    df = df.copy()
    df['state'] = np.nan
    df.loc[d.index, 'state'] = states
    df['state_name'] = df['state'].map(dict(enumerate(STATE_NAMES[K])))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"hmm_homemade_K{K}_parameters.json"), 'w') as f:
            json.dump(params, f, indent=2)
    return df, params


# ------------------------------------------------------------------
# Plot : distributions par état vs lois ajustées
# ------------------------------------------------------------------
def state_pdf(k, grid, tau_k, params):
    """Loi marginale de la vitesse dans l'état k, évaluée sur grid.
    tau_k = temps normalisé des observations dans l'état k : pour l'état de nage,
    la moyenne change avec le temps, donc on moyenne la loi normale sur ces instants."""
    if k < params['K'] - 1:
        return rayleigh.pdf(grid, scale=params['sigma_rayleigh_um_s'][k])
    mu = params['mu_N_um_s'] + params['beta_um_s_total'] * tau_k
    mu = mu[::max(1, len(mu) // 300)]  # sous-échantillon, suffisant pour la courbe
    return norm.pdf(grid[:, None], mu[None, :], params['sigma_N_um_s']).mean(axis=1)


def plot_state_distributions(df, params, bins=100):
    """Histogramme des vitesses par état (états décodés par Viterbi)
    vs lois théoriques ajustées, pondérées par la fraction de temps passée dans l'état."""
    d = df.dropna(subset=['state', 'speed'])
    x = d['speed'].to_numpy(dtype=float)
    s = d['state'].to_numpy(dtype=int)
    tau = (d['frame'].to_numpy(dtype=float) - params['frame0']) / params['frame_span']
    colors = plt.get_cmap('tab10').colors

    grid = np.linspace(0, np.quantile(x, 0.999), 500)
    edges = np.linspace(0, grid[-1], bins + 1)
    width = edges[1] - edges[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(x, bins=edges, density=True, color='lightgray', label='toutes les données')
    total = np.zeros_like(grid)
    for k, name in enumerate(params['state_names']):
        xk = x[s == k]
        if len(xk) == 0:
            continue
        w = len(xk) / len(x)
        # aire de l'histogramme = fraction de temps dans l'état
        ax.hist(xk, bins=edges, weights=np.full(len(xk), w / (len(xk) * width)),
                color=colors[k], alpha=0.35, label=f'{name} (données, {w:.0%})')
        pdf = w * state_pdf(k, grid, tau[s == k], params)
        ax.plot(grid, pdf, color=colors[k], lw=2, label=f'{name} (modèle)')
        total += pdf
    ax.plot(grid, total, 'k--', lw=1.5, label='somme des modèles')
    ax.set_xlabel('Speed (µm/s)')
    ax.set_ylabel('Densité')
    ax.legend()
    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    K = 3         # 2 ou 3 états : c'est le seul réglage à changer
    MAX_GAP = 40  # trous (en frames) comblés par la chaîne

    filename = ask_open_tracking_file()
    FPS, PIXEL_SIZE_UM = guess_or_ask_fps_and_pixelsize(filename)
    dt = 1 / FPS
    filepath, ext = os.path.splitext(filename)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "HMM")
    os.makedirs(out_dir, exist_ok=True)

    df = magic_load_trajectories(filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    df, params = fit_hmm_homemade(df, K=K, dt=dt, save_dir=out_dir, max_gap=MAX_GAP)
    magic_save_trajectories(df, filepath + f"_HMM_K{K}", ext)

    fig, ax = plot_speed_vs_states(df, dt, names=STATE_NAMES[K])
    fig.savefig(os.path.join(out_dir, f"speed_v_states_homemade_K{K}_unique_init.png"), dpi=150)

    fig2, ax2 = plot_state_distributions(df, params)
    fig2.savefig(os.path.join(out_dir, f"state_distributions_homemade_K{K}_unique_init.png"), dpi=150)
    plt.show()
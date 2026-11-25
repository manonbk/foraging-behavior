"""
Fit d'un HMM gaussien sur les vitesses + plot de contrôle.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hmmlearn.hmm import GaussianHMM
from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories, magic_save_trajectories
from trajectories.trajectory_analysis import scale_lengths, segments_from_table
from scipy.stats import norm

STATE_NAMES = {2: ['arret', 'nage'],
               3: ['arret', 'nage_lente', 'nage_rapide']}


def valid_segments(df):
    """Segments continus, sans les lignes où la vitesse est NaN
    (typiquement la 1re frame de chaque trajectoire, où la vitesse n'est pas définie)."""
    segs = [seg[seg['speed'].notna()] for seg in segments_from_table(df)]
    return [seg for seg in segs if len(seg) > 0]


def fit_hmm(df, K=2, dt=1.0, save_dir=None, n_init=3):
    """Fit un HMM à K états sur la colonne 'speed'.
    Chaque segment continu est une séquence indépendante.
    Les états sont numérotés par vitesse moyenne croissante (0 = le plus lent).
    Ajoute les colonnes 'state' (0..K-1) et 'state_name' au df."""

    segments = valid_segments(df)
    sequences = [seg['speed'].to_numpy().reshape(-1, 1) for seg in segments]
    lengths = [len(s) for s in sequences]
    X = np.concatenate(sequences)
    print(f"{len(X)} observations, {len(lengths)} segments continus")

    # Plusieurs initialisations, on garde la meilleure
    best, best_score = None, -np.inf
    for seed in range(n_init):
        m = GaussianHMM(n_components=K, covariance_type='diag',
                        n_iter=200, random_state=seed)
        m.fit(X, lengths)
        score = m.score(X, lengths)
        if score > best_score:
            best, best_score = m, score
    model = best

    # Renumérotation des états par vitesse moyenne croissante
    order = np.argsort(model.means_.ravel())
    rank = np.empty(K, dtype=int)
    rank[order] = np.arange(K)
    states = rank[model.predict(X, lengths)]

    # Remettre les états dans le df (on suppose que les segments gardent l'index de df)
    df = df.copy()
    df['state'] = np.nan  # reste NaN là où la vitesse est NaN
    for seg, s in zip(segments, np.split(states, np.cumsum(lengths)[:-1])):
        df.loc[seg.index, 'state'] = s
    df['state_name'] = df['state'].map(dict(enumerate(STATE_NAMES.get(K, range(K)))))

    # Paramètres du modèle (dans l'ordre trié)
    transmat = model.transmat_[order][:, order]
    n_params = K**2 + 2 * K - 1
    params = {
        'K': K,
        'dt_seconds': dt,
        'means_um_s': model.means_.ravel()[order].tolist(),
        'stds_um_s': np.sqrt(model.covars_[:, 0, 0])[order].tolist(),
        'start_probabilities': model.startprob_[order].tolist(),
        'transition_matrix': transmat.tolist(),
        'mean_durations_s': (dt / (1 - np.diag(transmat))).tolist(),
        'log_likelihood': best_score,
        'BIC': n_params * np.log(len(X)) - 2 * best_score,
        'converged': bool(model.monitor_.converged),
    }
    print(json.dumps(params, indent=2))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"hmm_parameters_K{K}.json"), 'w') as f:
            json.dump(params, f, indent=2)

    return df, model


def plot_speed_vs_states(df, dt, names=None):
    """Vitesse en noir, couleur de fond = état. Les trous (frames sans détection)
    restent blancs."""
    K = int(df['state'].max()) + 1  # max() ignore les NaN
    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(15, 5))
 
    for seg in valid_segments(df):
        t = seg['frame'].to_numpy(dtype=float) * dt / 60  # minutes
        s = seg['state'].to_numpy(dtype=int)
        ax.plot(t, seg['speed'], color='black', lw=0.4)
 
        # un fond coloré par plage d'états constants
        starts = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
        ends = np.r_[starts[1:], len(s)] - 1
        for a, b in zip(starts, ends):
            ax.axvspan(t[a], t[b] + dt / 60, color=colors[s[a] % 10],
                       alpha=0.2, linewidth=0)
 
    names = names or STATE_NAMES.get(K, [f'State {k}' for k in range(K)])
    ax.legend(handles=[Patch(facecolor=colors[k % 10], alpha=0.3, label=names[k])
                       for k in range(K)])
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Speed (µm/s)')
    ax.set_title(f'HMM - {K} états')
    fig.tight_layout()
    return fig, ax



def plot_state_distrib(df, means, stds, bins=100):
    d = df.dropna(subset=['state', 'speed'])
    speeds = d['speed'].to_numpy(dtype=float)
    states = d['state'].to_numpy(dtype=int)
    valid = np.isfinite(speeds) & (speeds >= 0) & (speeds <= 1000)
    speeds = speeds[valid]
    states = states[valid]
    fig, ax = plt.subplots(figsize=(10, 5))
    # Histogram
    edges = np.linspace(0, 1000, bins + 1)
    ax.hist(speeds, bins=edges, density=True,
            alpha=0.4, edgecolor='black', label='Data')
    # Gaussian distributions
    x = np.linspace(0, 1000, 500)
    total = np.zeros_like(x)
    for k, (mu, std) in enumerate(zip(means, stds)):
        w = np.mean(states == k)
        y = w * norm.pdf(x, loc=mu, scale=std)
        ax.plot(x, y, label=f'State {k}')
        total += y

    ax.plot(x, total, 'k--', label='Total model')
    ax.set_xlim(0, 1000)
    ax.set_xlabel('Speed (µm/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed distributions')
    ax.legend()
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    K = 3  # 2 ou 3

    filename = ask_open_tracking_file()
    FPS, PIXEL_SIZE_UM = guess_or_ask_fps_and_pixelsize(filename)
    dt = 1 / FPS
    filepath, ext = os.path.splitext(filename)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "HMM")
    os.makedirs(out_dir, exist_ok=True)

    df = magic_load_trajectories(filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    df, model = fit_hmm(df, K=K, dt=dt, save_dir=out_dir)
    magic_save_trajectories(df, filepath + f"_HMM_K{K}", ext)

    order = np.argsort(model.means_.ravel())
    means=model.means_.ravel()[order].tolist()
    stds = np.sqrt(model.covars_[:, 0, 0])[order].tolist()
    fig, ax= plot_state_distrib(df, means, stds)
    fig.savefig(os.path.join(out_dir, f"distrib_states_K{K}.png"), dpi=150)
    plt.show()

    fig, ax = plot_speed_vs_states(df, dt)
    fig.savefig(os.path.join(out_dir, f"speed_v_states_K{K}.png"), dpi=150)
    plt.show()



    # with open(r"C:\Users\manon\paramecium\recordings\20261002\tracking\HMM\hmm_parameters_K2.json", "r", encoding="utf-8") as f:
    #     params = json.load(f)
    # means = params['means_um_s']
    # stds = params['stds_um_s']
    # fig, ax= plot_state_distrib(df, means, stds)
    # fig.savefig(os.path.join(out_dir, f"distrib_states_K{K}.png"), dpi=150)
    # plt.show()
    

    
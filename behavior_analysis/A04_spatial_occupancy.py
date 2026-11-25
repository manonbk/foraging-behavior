"""
A04 - WHERE are the swims and stops? Occupancy heatmaps of all positions, of swims or of stops.

Reads speed_table.csv (A01) and copies swim / stop onto the per-frame tracking table, so the
three maps share exactly the same bins. Outputs go to <data folder>/movement_analysis/spatial_occupancy/.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from trajectories.magic_loading import magic_load_trajectories
from trajectories.trajectory_analysis import scale_lengths

from .common import setup, load_speed_table, bin_duration
from .speed_core import add_speed_columns_to_frames

BIN_SIZE_UM = 10
SMOOTHING_MODE = "hist"  # or "hist_gauss"
GAUSS_SIGMA_UM = 150


# ----------------------------------------------------------------------
# Grille de densité
# ----------------------------------------------------------------------
def compute_density_grid(df, label, path,
                         bin_um=BIN_SIZE_UM, smoothing_mode=SMOOTHING_MODE,
                         gauss_sigma_um=GAUSS_SIGMA_UM):
    """label="all" -> occupation globale ;
    sinon uniquement les positions pour lesquelles la colonne `label` ("swim" ou "stop") est True."""

    # La grille est toujours calculée sur TOUTES les positions, pour que les
    # cartes global / swims / stops aient exactement les mêmes bins (comparables).
    xs_all = df["x"].to_numpy()
    ys_all = df["y"].to_numpy()
    x_edges = np.arange(np.nanmin(xs_all), np.nanmax(xs_all) + bin_um, bin_um)
    y_edges = np.arange(np.nanmin(ys_all), np.nanmax(ys_all) + bin_um, bin_um)
    nx = len(x_edges) - 1
    ny = len(y_edges) - 1

    # Sous-sélection éventuelle
    if label != "all":
        if label not in df.columns:
            raise ValueError(
                f"Error: '{label}' not in the data. "
                "Run A01_swim_stop.py first."
            )
        df=df.dropna(subset=[label])
        df = df[df[label]==True]
        print(f"[{label}] -> {len(df):,} positions retenues")
    else:
        print(f"[{label}] toutes les positions : {len(df):,}")

    xs = df["x"].to_numpy()
    ys = df["y"].to_numpy()

    print(f"  Grille : {nx} x {ny} bins de {bin_um:.2f} µm")
    print(f"  x : {x_edges[0]:.2f} – {x_edges[-1]:.2f} µm | y : {y_edges[0]:.2f} – {y_edges[-1]:.2f} µm")

    # Histogramme 2D : counts de shape (nx, ny)
    counts, _, _ = np.histogram2d(xs, ys, bins=[x_edges, y_edges])
    total_counts = counts.sum()
    print(f"  Total counts (frames) : {int(total_counts):,}")

    if total_counts == 0:
        print(f"  Aucune donnée pour '{label}' : grille non générée.", file=sys.stderr)
        return None

    # Densité de probabilité (1/µm²)
    probability = counts / total_counts
    density_raw = probability / (bin_um ** 2)

    # Lissage
    print(f"  Smoothing mode : {smoothing_mode}")
    if smoothing_mode == "hist":
        density_smoothed = density_raw.copy()
    elif smoothing_mode == "hist_gauss":
        from scipy.ndimage import gaussian_filter
        sigma_bins = gauss_sigma_um / bin_um
        density_smoothed = gaussian_filter(density_raw, sigma=sigma_bins)
        total_density = density_smoothed.sum() * bin_um ** 2
        if total_density == 0:
            print("  Densité lissée nulle : abandon.", file=sys.stderr)
            return None
        density_smoothed = density_smoothed / total_density
    else:
        print(f"  Mode de lissage inconnu '{smoothing_mode}' : histogramme brut utilisé.")
        density_smoothed = density_raw.copy()

    # Tableau : une ligne par bin (ordre x externe, y interne)
    X0, Y0 = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(x_edges[1:], y_edges[1:], indexing="ij")
    grid_df = pd.DataFrame({
        "x_min": X0.ravel(),
        "x_max": X1.ravel(),
        "y_min": Y0.ravel(),
        "y_max": Y1.ravel(),
        "count": counts.astype(int).ravel(),
        "probability": probability.ravel(),
        "density_raw": density_raw.ravel(),
        "density_smoothed": density_smoothed.ravel(),
    })

    grid_df.to_csv(path, sep="\t", index=False, float_format="%.6e")
    print(f"  Grille sauvegardée : {path}")
    return grid_df


# ----------------------------------------------------------------------
# Heatmap
# ----------------------------------------------------------------------
def plot_heatmap(grid_df, label, path):
    print(f"\nHeatmap d'occupation spatiale : {label}")

    # Reconstruction de la matrice (nx, ny) depuis le tableau
    nx = grid_df["x_min"].nunique()
    ny = grid_df["y_min"].nunique()
    density = grid_df["density_smoothed"].to_numpy().reshape(nx, ny)

    nonzero_vals = density[density > 0]
    vmax = float(np.percentile(nonzero_vals, 99)) if nonzero_vals.size > 0 else None

    x_left, x_right = grid_df["x_min"].min(), grid_df["x_max"].max()
    y_bottom, y_top = grid_df["y_min"].min(), grid_df["y_max"].max()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(
        density.T,  # imshow attend (y, x)
        origin="lower",
        extent=[x_left, x_right, y_bottom, y_top],
        aspect="equal",
        cmap="inferno",
        interpolation="nearest",
        vmax=vmax,
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability density (1/µm²)")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Spatial occupancy density – {label}")
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap sauvegardée : {path}")


# ----------------------------------------------------------------------
def main():
    """ STEP 1 : GET DATA and scale lengths"""
    cfg = setup()
    FPS, PIXEL_SIZE_UM = cfg.fps, cfg.pixel_size_um
    df = magic_load_trajectories(cfg.filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    """ STEP 2 : ADD swim / stop (from the speed table written by A01)"""
    tab = load_speed_table(cfg.plot_dir)
    dt = 1 / FPS
    bin_frames = int(round(bin_duration(tab) / dt))
    df = add_speed_columns_to_frames(df, tab, dt, bin_frames)
    print(f"Loaded rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    """ STEP 3 : CREATE OUTPUT FOLDER"""
    PLOT_DIR = os.path.join(cfg.plot_dir, "spatial_occupancy")
    os.makedirs(PLOT_DIR, exist_ok=True)

    """ STEP 4 : spatial occupancy (global, swims, stops)"""
    for label in ["all", "swim", "stop"]:
        grid_path = os.path.join(PLOT_DIR, f"density_grid_{label}.tsv")
        plot_path = os.path.join(PLOT_DIR, f"spatial_occupancy_{label}.png")

        grid_df = compute_density_grid(df, label, grid_path)
        if grid_df is not None:
            plot_heatmap(grid_df, label, plot_path)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LogNorm
from interface.filedialogs import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories, magic_save_trajectories
from trajectories.trajectory_analysis import scale_lengths

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

def plot_angle_distributions(df, path):
    angles = df['angle'].to_numpy(dtype=float)
    angular_speeds = df['angular_speed'].to_numpy(dtype=float)
    angles = angles[np.isfinite(angles)]
    angular_speeds = angular_speeds[np.isfinite(angular_speeds)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(angles, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Angle (rad)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Angle Distribution')

    axes[1].hist(angular_speeds, bins=100, density=True, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Angular speed (rad/frame)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Angular Speed Distribution')

    fig.tight_layout()
    fig.savefig(path)
    plt.show()
    return fig, axes

def calculate_orientation_alignment(df):
    angles = df['angle'].to_numpy(dtype=float)
    vx = df['vx'].to_numpy(dtype=float)
    vy = df['vy'].to_numpy(dtype=float)
    speeds = np.hypot(vx, vy)

    valid = (
        np.isfinite(angles)
        & np.isfinite(vx)
        & np.isfinite(vy)
        & (speeds > 0)
    )
    alignment = np.full(len(df), np.nan)
    trajectory_angles = np.arctan2(vy[valid], vx[valid])
    # alignment[valid] = np.abs(np.cos(angles[valid] - trajectory_angles))
    alignment[valid] = np.cos(angles[valid] - trajectory_angles)

    return pd.Series(alignment, index=df.index, name='orientation_alignment')

def plot_orientation_alignment(alignment, path):
    values = alignment.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("No valid orientation-alignment values to plot")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=np.linspace(-1, 1, 101), density=True,
            alpha=0.7, edgecolor='black')
    ax.set_xlim(-1, 1)
    ax.set_xlabel('cos(body angle - trajectory angle)')
    ax.set_ylabel('Density')
    ax.set_title('Body Orientation Alignment')

    fig.tight_layout()
    fig.savefig(path)
    plt.show()
    return fig, ax

def plot_speed_alignment_heatmap(df, path):
    speeds = df['speed'].to_numpy(dtype=float)
    alignments = df['angular_speed'].to_numpy(dtype=float)
    valid = (
        np.isfinite(speeds)
        & np.isfinite(alignments)
        & (speeds >= 0)
        & (alignments >= -1.6)
        & (alignments <= 1.6)
    )
    if not np.any(valid):
        raise ValueError("No valid speed and orientation-alignment pairs to plot")

    fig, ax = plt.subplots(figsize=(8, 6))
    histogram = ax.hist2d(
        speeds[valid],
        alignments[valid],
        bins=(100, 50),
        cmap='viridis',
        norm=LogNorm(),
    )
    fig.colorbar(histogram[3], ax=ax, label='Number of observations')
    ax.set_xlabel('Speed (µm/s)')
    ax.set_ylabel('Angular speed')
    ax.set_ylim(-1.6, 1.6)
    ax.set_title('Speed vs. Angular speed')

    fig.tight_layout()
    fig.savefig(path)
    plt.show()
    return fig, ax



if __name__ == "__main__":
    filename = ask_open_tracking_file()
    FPS, PIXEL_SIZE_UM = guess_or_ask_fps_and_pixelsize(filename)
    dt = 1 / FPS
    filepath, ext = os.path.splitext(filename)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(filename)), "mouvement_analysis")
    os.makedirs(out_dir, exist_ok=True)

    df = magic_load_trajectories(filename, link=False)
    scale_lengths(df, PIXEL_SIZE_UM, FPS)

    # path=os.path.join(out_dir, f"raw_speed_distrib.png")
    # plot_raw_distributions(df, path)

    # path = os.path.join(out_dir, "raw_angle_distrib.png")
    # plot_angle_distributions(df, path)

    # df['orientation_alignment'] = calculate_orientation_alignment(df)
    # path = os.path.join(out_dir, "raw_orientation_alignment_relative.png")
    # plot_orientation_alignment(df['orientation_alignment'], path)

    # df['orientation_alignment'] = calculate_orientation_alignment(df)
    path = os.path.join(out_dir, "speed_vs_angular_speed.png")
    plot_speed_alignment_heatmap(df, path)
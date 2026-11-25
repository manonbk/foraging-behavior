"""
Shared helpers for the movement analysis scripts (A01 -> A04).

Pipeline
    A01_swim_stop.py          detect: speed per bin, stops / swims  -> speed_table.csv, stops.csv, swims.csv
    A02_durations.py          how long: stop / swim durations, survival and hazard of stops
    A03_speed_dynamics.py     how: swim <-> stop transitions, return map, autocorrelation
    A04_spatial_occupancy.py  where: occupancy maps (all / swim / stop)

Everything after A01 only reads what A01 wrote in <data folder>/movement_analysis/.
The computations themselves live in speed_core.py.
"""

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize

OUTPUT_DIRNAME = "movement_analysis"
SPEED_TABLE_NAME = "speed_table.csv"

# scale -> column of the speed table
COL = {"log": "logv", "linear": "v"}


# ---------------------------------------------------
# SETUP (the block that used to be copy-pasted in every main)
# ------------------------------------------------------

@dataclass
class Setup:
    filename: str
    folder: str
    plot_dir: str
    fps: Optional[float] = None
    pixel_size_um: Optional[float] = None


def setup(need_scales=True):
    """Ask for the tracking file, create the output folder.

    need_scales=True  also asks / guesses FPS and pixel size (needed when the tracking file is
                      loaded, e.g. A01, A04). The scripts that only read speed_table.csv
                      (A02 table part, A03) can pass False and skip that dialog.
    """
    filename = ask_open_tracking_file()
    folder = os.path.dirname(os.path.abspath(filename))
    plot_dir = os.path.abspath(os.path.join(folder, OUTPUT_DIRNAME))
    os.makedirs(plot_dir, exist_ok=True)
    cfg = Setup(filename, folder, plot_dir)
    if need_scales:
        cfg.fps, cfg.pixel_size_um = guess_or_ask_fps_and_pixelsize(filename)
    return cfg


# ---------------------------------------------------
# SPEED TABLE (written by A01, read by A02, A03, A04)
# ------------------------------------------------------

def col_of_scale(scale):
    if scale not in COL:
        raise ValueError("scale must be 'log' or 'linear'")
    return COL[scale]


def scale_of_col(col):
    return "log" if col == "logv" else "linear"


def prepare_table(tab):
    """Sort, reset the index (row position = index, used everywhere below), check columns."""
    needed = {"seq_index", "id", "t", "v", "logv", "stop", "swim", "bout"}
    missing = needed - set(tab.columns)
    if missing:
        raise ValueError(f"missing columns in speed table: {sorted(missing)}")
    tab = tab.sort_values(["seq_index", "t"]).reset_index(drop=True)
    tab["stop"] = tab["stop"].astype(bool)
    tab["swim"] = tab["swim"].astype(bool)
    return tab


def bin_duration(tab):
    """Duration of one observation (s) = median time step inside sequences."""
    return float(tab.groupby("seq_index")["t"].diff().median())


def lagged(tab, col, n):
    """Value of `col` n bins later in the same sequence (NaN when it would leave the sequence)."""
    return tab.groupby("seq_index")[col].shift(-n)


def load_speed_table(plot_dir):
    """Read and prepare the table written by A01_swim_stop.py."""
    path = os.path.join(plot_dir, SPEED_TABLE_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run A01_swim_stop.py first.")
    return prepare_table(pd.read_csv(path))

"""
All pipeline to preprocess data from single cell recording 
(P01 (ROI) then P02 (dust remove) then P03 (interpolate)) and then compute amo's data
"""

import os

import pandas as pd

from interface import ask_open_tracking_file, guess_or_ask_fps_and_pixelsize
from trajectories.magic_loading import magic_load_trajectories, magic_save_trajectories
from trajectories.trajectory_analysis import filter_shape, scale_lengths

from . import P01_ROI
from . import P02_clean_detection as clean
from .P03_interpolate import interpolate

""" CONSTANTS"""

MIN_DIAMETER = 100  # um
MASK_DIAMETER = 200  # um
MAX_GAP_FRAMES = 10  # for interpolation
UNIT = "um"


def count_cells_per_frame(data, expected_frames):
    n_detections = data.groupby("frame").size()
    n_detections = n_detections.value_counts()
    n_detections = pd.concat(
        [
            n_detections,
            pd.Series(
                {0: expected_frames - data["frame"].nunique()},
                index=[0],
            ),
        ]
    )
    print(n_detections)


def main():
    """Run the interactive single-cell preprocessing pipeline."""
    filename = ask_open_tracking_file()
    if not filename:
        raise RuntimeError("No tracking file was selected.")

    fps, pixel_size = guess_or_ask_fps_and_pixelsize(filename)
    filepath, ext = os.path.splitext(filename)
    folder = os.path.dirname(os.path.abspath(filename))

    print(f"Starting preprocessing {filename}")
    print(f"fps: {fps}, pixel size: {pixel_size}")

    data = magic_load_trajectories(filename, link=False)
    expected_frames = data["frame"].max() - data["frame"].min() + 1
    initial_frames = data["frame"].nunique()

    scale_lengths(data, pixel_size, fps)
    data = filter_shape(data)
    frames_filtered = data["frame"].nunique()
    before_roi = len(data)

    print("Expected total frames:", expected_frames)
    print("Initial frames:", initial_frames)
    print("Filtered frames:", frames_filtered)
    print("Frames removed by scaling/filtering:", initial_frames - frames_filtered)
    print("Cell counts in initial table:")
    count_cells_per_frame(data, expected_frames)

    roi = P01_ROI.select_circular_roi(data, unit=UNIT, max_frame=10000)
    if roi is not None:
        cx, cy, radius = roi
        print(
            f"ROI circle: center=({cx}, {cy}) {UNIT}, "
            f"radius={radius} {UNIT}"
        )
        print("Removing detections outside ROI")
        data = P01_ROI.filter_remove_outside_roi(data, cx, cy, radius)
        print(f"Rows removed: {before_roi - len(data):,}")
        print(f"Rows remaining: {len(data):,}")

    print("Saving ROI-filtered trajectories")
    magic_save_trajectories(data, filepath + "_ROI_filtered", ext)

    data = clean.label_segments(data, MIN_DIAMETER)
    data, masks = clean.remove_competing_segments(data)
    print("Cell counts after removing competing stationary segments:")
    count_cells_per_frame(data, expected_frames)

    data = clean.remove_segments_in_mask(
        data, masks, mask_diameter=MASK_DIAMETER
    )
    print("Cell counts after removing segments in dust masks:")
    count_cells_per_frame(data, expected_frames)

    data = clean.enforce_one_id_per_frame(data)
    print("Cell counts after enforcing one ID per frame:")
    count_cells_per_frame(data, expected_frames)
    magic_save_trajectories(data, filepath + "_cleaned", ext)

    print("Interpolating missing frames...")
    data = interpolate(data, MAX_GAP_FRAMES, 1 / fps)
    print("Final cell counts after interpolation:")
    count_cells_per_frame(data, expected_frames)
    output_file = os.path.join(
        folder, f"tracking_dust_removed_{fps}_fps_{pixel_size}_um"
    )
    magic_save_trajectories(data, output_file, ext)
    return output_file


if __name__ == "__main__":
    main()



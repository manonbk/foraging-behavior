import numpy as np
import pandas as pd


def add_kinematics(cell):
    """
    Recomputes vx, vy, speed, ax, ay, acceleration for a cell, per segment
    """
    cell = cell.copy()

    # Un nouveau segment démarre à chaque saut de frame (long gap non interpolé)
    seg = (cell["frame"].diff() != 1).cumsum()
    gx = cell.groupby(seg)["x"]
    gy = cell.groupby(seg)["y"]

    x_next, x_prev = gx.shift(-1), gx.shift(1)
    y_next, y_prev = gy.shift(-1), gy.shift(1)

    # Vitesse : NaN sur le dernier point de chaque segment
    cell["vx"] = x_next - cell["x"]
    cell["vy"] = y_next - cell["y"]
    cell["speed"] = (cell["vx"] ** 2 + cell["vy"] ** 2) ** 0.5

    # Accélération : NaN sur le premier et le dernier point de chaque segment
    cell["ax"] = x_next - 2 * cell["x"] + x_prev
    cell["ay"] = y_next - 2 * cell["y"] + y_prev
    cell["acceleration"] = (cell["ax"] ** 2 + cell["ay"] ** 2) ** 0.5

    return cell


def interpolate(df, max_frames_gap, dt, max_speed=1000):
    """
    !! df in um
    Interpolate short gaps (gap frames < max_frames_gap) in each cell trajectory, 
        only if interpolated speed < 1000 um/s. 
    Recomputes vx, vy, speed, ax, ay, acceleration on the interpolated trajectory.

    Returns : dataframe with original detections plus interpolated points.
        Includes an 'interpolated' boolean column.
    """
    results = []

    for cell_id, cell in df.groupby("id"):
        # Keep valid detections and sort by frame
        cell = cell.dropna(subset=["frame", "x", "y"]).copy()

        cell["frame"] = cell["frame"].astype(int)
        cell = cell.sort_values("frame").drop_duplicates("frame", keep="first")

        if cell.empty:
            continue

        # Create a continuous sequence of frames
        frames = np.arange(cell["frame"].iloc[0], cell["frame"].iloc[-1] + 1)

        cell = cell.set_index("frame").reindex(frames)

        x = cell["x"].to_numpy(dtype=float).copy()
        y = cell["y"].to_numpy(dtype=float).copy()

        missing = np.isnan(x)
        interpolated = np.zeros(len(frames), dtype=bool)

        # Detect consecutive runs of missing frames
        edges = np.diff(np.r_[0, missing.astype(int), 0])
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]

        for s, e in zip(starts, ends):
            gap_length = e - s

            # Interpolate only internal gaps below the threshold
            if gap_length < max_frames_gap:
                dx = x[e] - x[s - 1]
                dy = y[e] - y[s - 1]
                distance = np.sqrt(dx**2 + dy**2)
                deltaT = (e - (s - 1)) * dt
                speed = distance / deltaT
                #only interpolate if speed < max_speed
                if speed <= max_speed:
                    idx = np.arange(s, e)
                    x[s:e] = np.interp(idx, [s - 1, e], [x[s - 1], x[e]])
                    y[s:e] = np.interp(idx, [s - 1, e], [y[s - 1], y[e]])
                    interpolated[s:e] = True

        # Keep original detections and newly interpolated points.
        # Long gaps are still represented by missing rows.
        keep = ~np.isnan(x)
        out = cell.reindex(frames[keep]).copy()

        out["frame"] = frames[keep]
        out["x"] = x[keep]
        out["y"] = y[keep]
        out["id"] = cell_id
        out["interpolated"] = interpolated[keep]

        # Recompute kinematics on the (partially) interpolated trajectory
        out = add_kinematics(out)

        results.append(out)

    cols = ["frame", "x", "y", "id", "interpolated", "length", "width",
            "vx", "vy", "speed", "ax", "ay", "acceleration"]

    if not results:
        return pd.DataFrame(columns=cols)

    out = pd.concat(results, ignore_index=True).sort_values(["id", "frame"]).reset_index(drop=True)
    print(f"Pourcentage interpolated: {100 * out['interpolated'].mean():.2f}%")

    return out
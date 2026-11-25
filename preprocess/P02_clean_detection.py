"""
P02_clean_detections.py

Cleaning up false detections (stationary dust particles) in a single-cell
tracking table, based on validated "moving" segments/trajectories.

Pipeline:
1. label_segments            : adds segment_id + moving_segment
2. remove_competing_segments : in frames where a moving segment exists, removes
other detections AND their entire segment; 
stores the position of these segments (= dust)
3. remove_segments_in_mask   : removes unvalidated segments located entirely within
a mask (disk) centered on each known dust particle
4. enforce_one_id_per_frame  : verifies/enforces 1 detection per frame

Assumptions: columns "frame", "x", "y", "id", "conf"; x and y in the same unit
as MIN_DIAMETER / MASK_DIAMETER; `data` index is unique.
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from trajectories.trajectory_analysis import (
    segments_from_table,
    trajectories_from_table,
    diameter,
)

""" CONSTANTES  """
MIN_DIAMETER = 100    # um : un segment plus "étendu" que ça est considéré mobile
MASK_DIAMETER = 200   # um : diamètre du masque posé sur chaque poussière


# ----------------------------------------------------------------------------
# 1. Labelling des segments
# ----------------------------------------------------------------------------
def label_segments(data, min_diameter=MIN_DIAMETER):
    """Add 'segment_id' (int) et 'moving_segment' (bool) to data.
    si use_segments alors segments else trajectories"""
    pieces = segments_from_table(data)

    seg_ids = np.full(len(data), -1, dtype=np.int64)
    moving = np.zeros(len(data), dtype=bool)

    for i, seg in enumerate(pieces):
        pos = data.index.get_indexer(seg.index)   # positions (rapide), pas de .loc en boucle
        seg_ids[pos] = i
        moving[pos] = diameter(seg) > min_diameter

    orphans = seg_ids == -1
    if orphans.any():
        print(f"[labels] {orphans.sum()} lines with no segments -> segments singletons")
        seg_ids[orphans] = np.arange(len(pieces), len(pieces) + orphans.sum())

    #out = data.copy()
    data["segment_id"] = seg_ids
    data["moving_segment"] = moving

    n_moving = len({s for s, m in zip(seg_ids, moving) if m})
    print(f"[labels] {len(pieces)} segments, {n_moving} moving, {len(pieces) - n_moving} still")
    return data


# ----------------------------------------------------------------------------
# 2. Suppression des détections concurrentes d'un segment mobile
# ----------------------------------------------------------------------------
def remove_competing_segments(data):
    """
    For frames containing a validated moving segment, any unvalidated detection
    is discarded, along with the ENTIRE segment to which it belongs (even in
    frames without a moving segment). 

    Returns dust positions [N,2].
    """
    moving_frames = data.loc[data["moving_segment"], "frame"].unique()
    competing = (~data["moving_segment"]) & data["frame"].isin(moving_frames)

    kill_ids = data.loc[competing, "segment_id"].unique()
    kill = data["segment_id"].isin(kill_ids)

    # 1 mask position per segment (median : robuste au jitter du détecteur)
    dust = data.loc[kill].groupby("segment_id")[["x", "y"]].median().to_numpy()

    print(f"[compete] {len(kill_ids)} segments deleted ,({kill.sum()} lines), "
          f"{len(dust)} mask positions")
    data=data.loc[~kill]
    return data, dust


# ----------------------------------------------------------------------------
# 3. Masque sur les poussières
# ----------------------------------------------------------------------------
def remove_segments_in_mask(data, dust, mask_diameter=MASK_DIAMETER):
    """
   Removes unvalidated segments for which ALL detections lie within a
    disk (of diameter mask_diameter) centered on a known dust particle. 
    Moving segments are never affected.
    """
    if len(dust) == 0:
        return data

    radius = mask_diameter / 2
    tree = cKDTree(dust)

    cand = ~data["moving_segment"].to_numpy()
    if not cand.any():
        return data

    xy = data.loc[cand, ["x", "y"]].to_numpy()
    dist, _ = tree.query(xy, distance_upper_bound=radius)   # inf si hors rayon
    inside = np.isfinite(dist)

    seg_inside = pd.Series(inside).groupby(data.loc[cand, "segment_id"].to_numpy()).all()
    kill_ids = seg_inside.index[seg_inside.to_numpy()]
    kill = data["segment_id"].isin(kill_ids) & ~data["moving_segment"]

    print(f"[mask] {len(kill_ids)} segments entièrement dans le masque supprimés ({kill.sum()} lignes)")
    return data.loc[~kill]


# ----------------------------------------------------------------------------
# 4. Vérification : une seule détection par frame
# ----------------------------------------------------------------------------
def enforce_one_id_per_frame(data, resolve=True):
    """
    Affiche les frames avec >1 détection. Si resolve=True, garde par frame la
    détection mobile avec la meilleure 'conf'.
    Finit par mettre id=1 à toutes les detections
    """
    counts = data.groupby("frame").size()
    multi = counts[counts > 1]
    print(f"[check] frames avec >1 détection : {len(multi)} / {len(counts)}")

    if len(multi) and resolve:
        data = (data.sort_values(["frame", "moving_segment", "conf"],
                                 ascending=[True, False, False])
                    .drop_duplicates("frame", keep="first")
                    .sort_values("frame"))
        print(f"[check] résolu : {len(data)} lignes restantes")
    data["id"]=1
    return data


# ----------------------------------------------------------------------------
# Pipeline complet
# ----------------------------------------------------------------------------
def clean_detections(data,
                     min_diameter=MIN_DIAMETER, mask_diameter=MASK_DIAMETER,
                     resolve_duplicates=True):
    n0 = len(data)
    data = label_segments(data, min_diameter)
    data, dust = remove_competing_segments(data)
    data = remove_segments_in_mask(data, dust, mask_diameter)
    data = enforce_one_id_per_frame(data, resolve=resolve_duplicates)
    print(f"[done] {n0} -> {len(data)} lignes ({n0 - len(data)} supprimées)")
    return data, dust
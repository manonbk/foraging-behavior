'''
Analysis of trajectories obtained with T-Rex

Default variables:
'ACCELERATION#pcentroid', 'ANGLE', 'ACCELERATION#wcentroid', 'ANGULAR_A#centroid', 'AX', 'ANGULAR_V#centroid', 'BORDER_DISTANCE#pcentroid', 'AY', 'MIDLINE_OFFSET',
'SPEED#wcentroid', 'SPEED#pcentroid', 'SPEED', 'VX', 'VY', 'X#wcentroid', 'X', 'Y#wcentroid', 'Y', 'frame', 'midline_length', 'midline_x', 'midline_y', 'missing',
'normalized_midline', 'num_pixels', 'segment_length', 'time', 'timestamp', 'frame_segments', 'segment_vxys'

Others are possible, see https://trex.run/docs/formats.html

T-Rex also has posture data, with outlines. See:
https://github.com/mooch443/trex/blob/main/docs/scripts/plot_posture_output.py
'''
import numpy as np
import pandas as pd
import os
import re

__all__ = ['load_trex_file', 'load_trex_files']

def load_trex_file(filename, id):
    M = np.load(filename)
    valid = M['X#wcentroid'] != np.inf
    return pd.DataFrame(data={'x': M['X#wcentroid'][valid],
                               'y': M['Y#wcentroid'][valid],
                               'angle': M['ANGLE'][valid],  # not the right angle
                               'length': M['midline_length'][valid], # or segment_length
                               #'width': None,
                               #'eccentricity': None,
                               'frame': M['frame'][valid],
                               'id': id})

def load_trex_files(path):
    '''
    Loads a sequence of trajectory files from a folder.
    '''
    # Loads all .npz files; last digits are the id
    pattern = re.compile(r'(\d+)\.npz')
    traj = []
    for f in os.scandir(path):
        filename = f.path
        result = pattern.search(filename)
        if result is not None:
            i = int(result.group(1))
            traj.append(load_trex_file(filename, i))

    return pd.concat(traj)

if __name__ == '__main__':
    import os
    from pylab import *
    from trajectories.trajectory_analysis import *

    #filename = os.path.expanduser('~/Downloads/TRex/05102022/movie4x.mp4_fish0.npz')
    data = load_trex_files(os.path.expanduser('~/Downloads/TRex/20220805_18'))
    #filename = os.path.expanduser('~/Downloads/TRex/20220805_18/20220805_18_fish185.npz')
    #M = np.load(filename)

    segments = segments_from_table(data)
    traj = segments[1]
    print(np.array(traj['y']))
    calculate_features(traj)

    x, y = traj['x'], traj['y']
    v = traj['speed']
    angle = traj['angle']

    subplot(211)
    plot(v)
    subplot(212)
    plot(angle)

    show()

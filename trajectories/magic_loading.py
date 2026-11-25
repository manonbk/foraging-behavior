from .fasttrack import load_fasttrack
import os
import pandas as pd
from .linking import trackpy_track
from tqdm import tqdm

__all__ = ['magic_load_trajectories', 'magic_save_trajectories']

__chunk_size = 50000

def read_with_progress(filename, load_function=pd.read_csv, nframes=1e12, **kwd): # A better way would be to report the number of frames
    '''
    Reads a tracking csv/tsv file and displays the number of frames read in a progress bar
    '''
    previous_n = 0
    chunks = []
    with tqdm(desc="Loading") as pbar:
        for chunk in load_function(filename, chunksize=__chunk_size, **kwd):
            n = chunk['frame'].max()
            pbar.update(n-previous_n)
            previous_n = n
            if (n<nframes+1):
                chunks.append(chunk)
            else:
                chunks.append(chunk[chunk['frame']<=nframes])
                break

    return pd.concat(chunks)

def magic_load_trajectories(filename, link=False, nframes=1e12):
    '''
    Magically loads a trajectory file by identifying its format.
    '''
    _, ext = os.path.splitext(filename)
    if ('.tsv' in filename): # table with header
        data = read_with_progress(filename, load_function=pd.read_table, nframes=nframes)
        # Older files have wrong names
        data = data.rename(
            columns={'centroid-0': 'y', 'centroid-1': 'x', 'major_axis_length': 'length', 'minor_axis_length': 'width',
                     'orientation': 'angle'})
        # Check if there are ids
    elif ('.csv' in filename):
        data = read_with_progress(filename, load_function=pd.read_csv, nframes=nframes)
    elif (ext == '.h5'):
        if nframes<1e12:
            data = pd.read_hdf(filename, key='df', stop=nframes*100) # not great, assuming 100 cells
            data = data[data['frame']<=nframes]
        else:
            data = pd.read_hdf(filename, key='df')
    elif os.path.basename(filename) == 'tracking.txt': # Fasttrack
        data = load_fasttrack(filename)
    else:
        raise OSError("Unknown format")

    try:
        if data['length'].mean() < data['width'].mean():  # Check whether it's inverted
            temp = data['length'].copy()
            data['length'] = data['width']
            data['width'] = temp
    except KeyError:
        pass

    if ('id' not in data) and link:
        t = trackpy_track(data)
        data["id"] = t['particle']

    return data


def magic_save_trajectories(data, filepath, ext):
    """
    Magically saves a trajectory table in the specified format.
    """
    filename = filepath + ext

    if ext == '.tsv':
        data.to_csv(filename, sep='\t', index=False)

    elif ext == '.csv':
        data.to_csv(filename, index=False)

    elif ext == '.zip':
        data.to_csv(
            filename,
            index=False,
            compression='zip'
        )

    elif ext == '.h5':
        data.to_hdf(
            filename,
            key='df',
            mode='w',
            format='table'
        )

    else:
        raise OSError(f"Unknown format: {ext}")

    print(f"Trajectories saved to: {filename}")

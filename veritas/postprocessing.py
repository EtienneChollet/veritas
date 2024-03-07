import scipy
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import label


def volumeFilter(in_arr:np.array, min_vx:int=20, max_vx:int=None, verbose:bool=True):
    """
    Filter a binary volume based on volume of connected components.

    Parameters
    ----------
    in_arr : array-like
        Array to filter by volume. 
    min_vx : int
        Minimum volume connected component to KEEP.

    Returns
    -------

    """
    
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_arr, n_components = label(in_arr, structure)

    # Putting labeled_arr into torch and onto GPU for fast removal.
    out_arr = torch.from_numpy(labeled_arr).to('cuda')

    hist = scipy.ndimage.histogram(
        labeled_arr,
        min=0,
        max=n_components,
        bins=n_components+1
        )

    df = pd.DataFrame(
        data=hist,
        columns=['volume'],
        index=range(0, n_components+1)
        )

    # Getting all of the volumes we want to keep based on volume.
    highpass = []
    lowpass = []
    if min_vx is not None:
        lowpass = list(df[df.volume <= min_vx].index)
        if verbose: print(f'Filtering out connected components under {min_vx} voxels...')
    if max_vx is not None:
        highpass = list(df[df.volume >= max_vx].index)
        df = df[df.volume <= max_vx]
        if verbose: print(f'Filtering out connected components above {max_vx} voxels...')
    df = df.drop(lowpass + highpass)
    # Consolidating all the label ID's we want to keep.
    labels_to_remove = list(highpass+lowpass)
    print(f'Removing {len(labels_to_remove)} connected components...')
    
    for id in labels_to_remove:
        out_arr[out_arr == id] = 0

    out_arr[out_arr > 0] = 1
    out_arr = out_arr.cpu().numpy()
    return out_arr
    
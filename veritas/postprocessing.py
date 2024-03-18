import scipy
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import label
from veritas.data import RealOct
from torchmetrics.classification import Dice


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
    out_arr : torch.Tensor
    """

    # Checking if in_arr is numpy array. If not, convert.
    if isinstance(in_arr, np.ndarray):
        pass
    elif isinstance(in_arr, torch.Tensor):
        if in_arr.device == 'cpu':
            in_arr = in_arr.numpy()
        else:
            in_arr = in_arr.cpu().numpy()

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
    return out_arr


def get_confusion(ground_truth, prediction, tissue_mask=None):
    """
    Compute and save TP, TN, FP, FN tensor.

    Parameters
    ----------
    ground_truth : tensor-like, str
        Path to ground truth volume.
    prediction : tensor-like, str
        Path to prediction volume.
    tissue_mask : tensor-like, str
        Tissue mask volume where tissue label is 0.
    """
    # Load ground truth and prediction volumes.
    ground_truth = RealOct(input=ground_truth, binarize=True, dtype=torch.bool).tensor
    prediction = RealOct(input=prediction, binarize=True, dtype=torch.bool).tensor
    tissue_mask = RealOct(input=tissue_mask, binarize=True, dtype=torch.bool).tensor
    # Applying volume filter
    ground_truth = volumeFilter(ground_truth)
    prediction = volumeFilter(prediction)
    # Applying tissue mask to all volumes
    ground_truth[tissue_mask == 1] = 0
    prediction[tissue_mask == 1] = 0
    # Getting dice score
    dice = Dice(num_classes=1, multiclass=False).to('cuda')
    dice = dice(ground_truth, prediction)
    dice = round(dice.item(), 3)
    print(dice)
    # Confusion matrix
    true_positives = torch.mul(ground_truth, prediction).to(torch.uint8) * 17
    false_positives = torch.less(ground_truth, prediction).to(torch.uint8) * 3
    false_negatives = torch.greater(ground_truth, prediction).to(torch.uint8) * 6
    out_vol = true_positives + false_positives + false_negatives

version = 2
step_size = 64

if __name__ == '__main__':
    get_confusion(
        ground_truth='/autofs/cluster/octdata2/users/epc28/veritas/output_old/models/version_8/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii',
        prediction=f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-{step_size}.nii.gz',
        tissue_mask='/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.pia.nii'
    )
    
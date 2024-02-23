import cv2
import torch
import pandas as pd
import numpy as np
import nibabel as nib
import scipy
from scipy.ndimage import label
import sys
import time

# Loading and preparing input tensor from caa data.
path = "/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_2561/predictions/v2561_I46_Somatosensory_20um_crop-prediction_stepsz-64_thresh-0.5.nii.gz"
#path = "/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_2561/predictions/caa17_occipital-prediction_stepsz-64.nii"


in_nifti = nib.load(path)
aff = in_nifti.affine
arr = torch.from_numpy(in_nifti.get_fdata()).to('cuda')
arr[arr < 0.5] = 0
arr[arr >= 0.5] = 1
arr = arr.to(torch.uint8).cpu().numpy()

# Labeling each connected component an individual value using 8-connected algorithm.
structure = np.ones((3, 3, 3), dtype=np.uint8)
labeled, ncomponents = label(arr, structure)
print(labeled.shape)
print(ncomponents)


# Computing frequency of each unique label. We don't want to calculate histogram for background (id=0)
hist = scipy.ndimage.histogram(labeled, min=0, max=ncomponents, bins=ncomponents+1)


# Making index for histogram which correlates labels and frequency.
# Shifting the label id's by 1 because we are missing zero.
df = pd.DataFrame(hist, columns=['vx'], index=range(0, ncomponents+1))
# Clipping volumes below 6000 voxels
df = df[df.vx > 6000]
# Sorting values by voxelin
df = df.sort_values('vx', ascending=False)
df = df[1:]
print(df)
labeled = torch.from_numpy(labeled).to('cuda')

# Validating that the volume corresponds to the correct vessel id's
i = 0
label_ids = list(df.index)
print(label_ids[i])
print(torch.count_nonzero(labeled[labeled == label_ids[i]]))

# Renumbering
labels_to_keep = df.index
out_vol = torch.zeros(labeled.shape, dtype=torch.int).to('cuda')
for i in range(len(labels_to_keep)):
    out_vol[labeled == labels_to_keep[i]] = i

objects = scipy.ndimage.find_objects(out_vol.cpu().numpy())
i = 0
object = out_vol[objects[i]]
#object[object != i] = 0

obj = object.cpu().numpy().astype(np.uint8)
nift = nib.nifti1.Nifti1Image(obj, affine=aff)
nib.save(nift, '/autofs/cluster/octdata2/users/epc28/veritas/sandbox/first-branch.nii.gz')
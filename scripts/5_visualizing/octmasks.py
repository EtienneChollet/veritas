import torch
import skimage
import sklearn
import numpy as np
import nibabel as nib
import sklearn.cluster
from veritas.data import RealOct


path = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa6/frontal/caa6_frontal.nii'

mask = RealOct(path, device='cpu').make_mask()

print(mask)
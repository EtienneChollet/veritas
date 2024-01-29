import torch
import numpy as np
import pandas as pd
import nibabel as nib
from veritas.data import RealOctPatchLoader
oct_path = '/autofs/cluster/octdata2/users/epc28/data/mri-oct_patching/Stacked_Retardance.nii'
#oct_path = '/autofs/cluster/kosmo/OCT_temp/Emi_5yrold_5380_P20/StackNii/Stacked_Retardance.nii.gz'
#mri_path = '/autofs/cluster/kosmo/OCT_temp/Emi_5yrold_5380_P20/StackNii/transfer/vecreg_DG/dti_fa.crop.2oct.0p15.nii.gz'
#oct_path = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii'


oct = RealOctPatchLoader(
    pad_it=False,
    input=oct_path,
    patch_size=21,
    step_size=21,
    device='cpu'
    ).random_patch_sampler(
        n_patches=10,
        seed=3 ,
        name_prefix='patch',
        output='coords')

#for i in oct:
#    print(i)
import torch
from veritas import RealOct
import nibabel as nib
from veritas.postprocessing import volumeFilter


base_path = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa17/occipital'

in_path = '/autofs/cluster/octdata2/users/epc28/veritas/output_old/models/version_8/predictions/best/caa17_occipital-prediction_masked.nii.gz'
out_path = '/autofs/cluster/octdata2/users/epc28/veritas/output_old/models/version_8/predictions/best/caa17_occipital-prediction_masked_filtered.nii.gz'

print('loading prediction...')
in_nifti = nib.load(in_path)
in_arr = torch.from_numpy(in_nifti.get_fdata()).to('cuda')
affine = in_nifti.affine

#print('masking volume...')
#s = prediction_tensor.shape
#pia_mask_volume[:s[0], :s[1], :s[2]] *= prediction_tensor

filtered_volume = volumeFilter(in_arr)
filtered_volume = filtered_volume.cpu().numpy()

out_nifti = nib.nifti1.Nifti1Image(dataobj=filtered_volume, affine=affine)
nib.save(out_nifti, out_path)
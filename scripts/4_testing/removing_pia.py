import torch
from veritas import RealOct
import nibabel as nib


base_path = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa17/occipital'

prediction_path = '/autofs/cluster/octdata2/users/epc28/veritas/output_old/models/version_8/predictions/best/caa17_occipital-prediction_THRESH-0.5.nii.gz' #f'{base_path}/caa17_occipital.nii.gz'
pia_mask_path = f'{base_path}/caa17_occipital_mask_5x-dilate.nii.gz'
out_path = '/autofs/cluster/octdata2/users/epc28/veritas/output_old/models/version_8/predictions/best/caa17_occipital-prediction_masked.nii.gz' #f'{base_path}/caa17_occipital-masked.nii.gz'

#out_path = prediction_path.split('/')[:-1]
#out_path.append(prediction_path.split('/')[-1].strip('.nii') + '_NO-PIA.nii')
#out_path = '/'.join(out_path)
#print(out_path)

print('loading prediction...')
prediction = nib.load(prediction_path)
prediction_tensor = torch.from_numpy(prediction.get_fdata()).to('cuda')
affine = prediction.affine

#prediction= RealOct(
#    input=prediction_path,
#)
print('prediction loaded')

pia_mask_volume = RealOct(
    input=pia_mask_path
).tensor

print('masking volume...')
s = prediction_tensor.shape
pia_mask_volume[:s[0], :s[1], :s[2]] *= prediction_tensor

masked_volume = prediction_tensor

#masked_volume = prediction.tensor
#masked_volume[pia_mask_volume > 0] = 0
#masked_volume = masked_volume.cpu().numpy()

out_nifti = nib.nifti1.Nifti1Image(dataobj=masked_volume, affine=prediction.affine)
nib.save(out_nifti, out_path)
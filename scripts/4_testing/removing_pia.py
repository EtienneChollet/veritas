from veritas import RealOct
import nibabel as nib

prediction_path = '/autofs/cluster/octdata2/users/epc28/veritas/output/paper_models_small/version_4/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-16.nii'
out_path = prediction_path.split('/')[:-1]
out_path.append(prediction_path.split('/')[-1].strip('.nii') + '_NO-PIA.nii')
out_path = '/'.join(out_path)
print(out_path)

pia_mask_path = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.pia.nii'

prediction= RealOct(
    input=prediction_path
)

pia_mask_volume = RealOct(
    input=pia_mask_path
).tensor

masked_volume = prediction.tensor
masked_volume[pia_mask_volume > 0] = 0
masked_volume = masked_volume.cpu().numpy()

out_nifti = nib.nifti1.Nifti1Image(dataobj=masked_volume, affine=prediction.affine)
nib.save(out_nifti, out_path)
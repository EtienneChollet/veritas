import numpy as np
import pandas as pd
import nibabel as nib


version = 2568
out_dtype = np.uint8
threshold = 0.5

# GROUND TRUTH
ground_truth = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/ground_truth.nii'
ground_truth = nib.load(ground_truth)
affine = ground_truth.affine
ground_truth = ground_truth.get_fdata()
ground_truth[ground_truth < 1] = 0
ground_truth[ground_truth >= 1] = 1
ground_truth = ground_truth.astype(out_dtype)

# TISSUE MASK
tissue_mask = nib.load('/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.pia.nii').get_fdata().astype(out_dtype)

# PREDICTION
prediction = nib.load(f'/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_{version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii').get_fdata()
prediction[prediction < threshold] = 0
prediction[prediction >= threshold] = 1
prediction = prediction.astype(out_dtype)

# VERSION 8 PREDICTION
v8_prediction = nib.load('/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii').get_fdata()
v8_prediction[v8_prediction < threshold] = 0
v8_prediction[v8_prediction >= threshold] = 1
v8_prediction = v8_prediction.astype(out_dtype)


# Applying tissue mask
prediction[tissue_mask >= 1] = 0
v8_prediction[tissue_mask >= 1] = 0
# Saving binarized prediction
prediction_binary_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_{version}/predictions/v{version}_I46_Somatosensory_20um_crop-prediction_stepsz-64_thresh-0.5.nii.gz'
prediction_binary_out_nifti = nib.nifti1.Nifti1Image(dataobj=prediction, affine=affine)
nib.save(prediction_binary_out_nifti, prediction_binary_out)

# CONFUSION FOR GROUND TRUTH
# Yellow
tp = (ground_truth * prediction).astype(out_dtype) * 17
# Red
fp = (ground_truth < prediction).astype(out_dtype) * 3
# Green
fn = (ground_truth > prediction).astype(out_dtype) * 6
print((2* np.count_nonzero(tp)) / (np.count_nonzero(prediction) + np.count_nonzero(ground_truth)))
out_vol = tp + fp + fn

confusion_gt_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_{version}/predictions/v{version}-confusion_gt.nii.gz'
out_nifti = nib.nifti1.Nifti1Image(dataobj=out_vol, affine=affine)
nib.save(out_nifti, confusion_gt_out)

# CONFUSION FOR VERSION 8
# Yellow
tp = (v8_prediction * prediction).astype(out_dtype) * 17
# Red
fp = (v8_prediction < prediction).astype(out_dtype) * 3
# Green
fn = (v8_prediction > prediction).astype(out_dtype) * 6
print((2* np.count_nonzero(tp)) / (np.count_nonzero(prediction) + np.count_nonzero(v8_prediction)))
out_vol = tp + fp + fn

confusion_v8_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_{version}/predictions/v{version}-confusion_v8.nii.gz'
out_nifti = nib.nifti1.Nifti1Image(dataobj=out_vol, affine=affine)
nib.save(out_nifti, confusion_v8_out)


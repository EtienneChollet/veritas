import argparse
import numpy as np
import nibabel as nib
from veritas.postprocessing import volumeFilter


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='TP+FP+FN Volume Constructor')

    parser.add_argument('--model', type=int, default=1,
                        help='model version used to make the prediction (default: 1)')
    parser.add_argument('--step-size', type=int, default=64,
                        help='step size used to make the prediction (default: 64)')
    

    args = parser.parse_args()
    version = args.model
    out_dtype = np.uint8
    threshold = 0.5
    step_size = 64


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
    prediction = nib.load(f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-{step_size}.nii.gz').get_fdata()
    prediction[prediction < threshold] = 0
    prediction[prediction >= threshold] = 1
    prediction = prediction.astype(out_dtype)
    #prediction = volumeFilter(prediction)

    # VERSION 8 PREDICTION
    v8_prediction = nib.load('/autofs/cluster/octdata2/users/epc28/veritas/output_old_v1/models/version_8/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii').get_fdata()
    v8_prediction[v8_prediction < threshold] = 0
    v8_prediction[v8_prediction >= threshold] = 1
    v8_prediction = v8_prediction.astype(out_dtype)
    #v8_prediction = volumeFilter(v8_prediction)


    # Applying tissue mask
    prediction[tissue_mask >= 1] = 0
    v8_prediction[tissue_mask >= 1] = 0
    # Saving binarized prediction

    prediction = prediction#.cpu().numpy()
    v8_prediction = v8_prediction#.cpu().numpy()

    prediction_binary_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/v{version}_I46_Somatosensory_20um_crop-prediction_stepsz-{step_size}_thresh-0.5.nii.gz'
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

    confusion_gt_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/v{version}-confusion_gt.nii.gz'
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

    confusion_v8_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/v{version}-confusion_v8.nii.gz'
    out_nifti = nib.nifti1.Nifti1Image(dataobj=out_vol, affine=affine)
    nib.save(out_nifti, confusion_v8_out)


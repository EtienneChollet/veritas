import argparse
import numpy as np
import nibabel as nib
from veritas.postprocessing import volumeFilter


if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description='TP+FP+FN Volume Constructor')

    parser.add_argument('--version', type=int, default=1,
                        help='model version used to make the prediction (default: 1)')
    parser.add_argument('--step-size', type=int, default=32,
                        help='step size used to make the prediction (default: 32)')
    parser.add_argument('--filter', type=bool, default=False)

    args = parser.parse_args()
    version = args.version
    step_size = args.step_size
    filter = args.filter
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
    prediction = nib.load(f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-{step_size}.nii.gz').get_fdata()
    prediction[prediction < threshold] = 0
    prediction[prediction >= threshold] = 1
    prediction = prediction.astype(out_dtype)
    if filter:
        prediction = volumeFilter(prediction).cpu().numpy()

    # VERSION 8 PREDICTION
    v8_prediction = nib.load('/autofs/cluster/octdata2/users/epc28/veritas/output_old_v1/models/version_8/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii').get_fdata()
    #v8_prediction = nib.load('/autofs/cluster/octdata2/users/epc28/data/caroline_data/etis-modded-v8_prediction.nii').get_fdata()
    v8_prediction[v8_prediction < threshold] = 0
    v8_prediction[v8_prediction >= threshold] = 1
    v8_prediction = v8_prediction.astype(out_dtype)
    if filter:
        v8_prediction = volumeFilter(v8_prediction).cpu().numpy()


    # Applying tissue mask
    prediction[tissue_mask >= 1] = 0
    v8_prediction[tissue_mask >= 1] = 0
    # Saving binarized prediction

    #prediction = prediction
    #v8_prediction = v8_prediction

    prediction_binary_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/v{version}_I46_Somatosensory_20um_crop-prediction_stepsz-{step_size}_thresh-0.5.nii.gz'
    prediction_binary_out_nifti = nib.nifti1.Nifti1Image(dataobj=prediction, affine=affine)
    nib.save(prediction_binary_out_nifti, prediction_binary_out)

    # CONFUSION FOR GROUND TRUTH
    # Yellow
    tp = (ground_truth * prediction).astype(out_dtype) * 17
    n_tp = np.count_nonzero(tp)
    # Red
    fp = (ground_truth < prediction).astype(out_dtype) * 3
    n_fp = np.count_nonzero(fp)
    # Green
    fn = (ground_truth > prediction).astype(out_dtype) * 6
    n_fn = np.count_nonzero(fn)

    n = np.size(ground_truth)
    n_tn = n - (n_tp + n_fp + n_fn)
    dice_gt = (2 * n_tp) / ((2 * n_tp) + n_fp + n_fn)
    fpr = n_fp / (n_fp + n_fn + n_tp) #(n_fp + n_tn)
    fnr = n_fn / (n_fp + n_fn + n_tp) #(n_fn + n_tp)

    print(dice_gt)
    print(fpr)
    print(fnr)


    #dice_gt = (2* np.count_nonzero(tp)) / ((2* np.count_nonzero(tp)) + np.count_nonzero(fp) + np.count_nonzero(fn))
    #print(f'Dice GT: {round(dice_gt, 3)}')
    #print('\nTP,FP,FN')
    #print(f'{np.count_nonzero(tp)}\n{np.count_nonzero(fp)}\n{np.count_nonzero(fn)}')
    

    #print(n)


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
    dice_v8 = (2* np.count_nonzero(tp)) / (np.count_nonzero(prediction) + np.count_nonzero(v8_prediction))
    print(f'\nDice V8: {round(dice_v8, 3)}')
    out_vol = tp + fp + fn

    #print(f'{round(dice_gt, 3)}\n{round(dice_v8, 3)}')
    #print(f'{round(dice_gt, 3)},{round(dice_v8, 3)}')

    confusion_v8_out = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{version}/predictions/v{version}-confusion_v8.nii.gz'
    out_nifti = nib.nifti1.Nifti1Image(dataobj=out_vol, affine=affine)
    nib.save(out_nifti, confusion_v8_out)


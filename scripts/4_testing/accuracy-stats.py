import torch
import numpy as np
import nibabel as nib


class IRV(object):

    def __init__(self, gt_path, prediction_path, threshold, tissue_mask=None):
        self.gt = nib.load(gt_path).get_fdata()
        #self.gt = np.clip(self.gt, 0, 1).astype(np.uint8)

        self.gt[self.gt < threshold] = 0
        self.gt[self.gt >= threshold] = 1

        self.prediction = nib.load(prediction_path).get_fdata()
        self.prediction[self.prediction < threshold] = 0
        self.prediction[self.prediction >= threshold] = 1
        self.prediction = self.prediction.astype(np.uint8)

        if tissue_mask is not None:
            self.tissue_mask = nib.load(tissue_mask).get_fdata()
            self.tissue_mask = np.clip(self.tissue_mask, 0, 1).astype(np.uint8)
            self.prediction[self.tissue_mask > 0] = 0
            self.gt[self.tissue_mask > 0] = 0
        else:
            pass

    def compute_dice(self):
        intersection = (self.gt * self.prediction).sum()
        hits_in_prediction = self.prediction.sum()
        hits_in_gt = self.gt.sum()
        dice = (2 * intersection) / (hits_in_prediction + hits_in_gt)
        print(dice)

    def inter_rater_variability(self):
        intersection = (self.gt1 * self.gt2).sum()
        dice = (2 * intersection) / (self.gt1.sum() + self.gt2.sum())
        print(dice)


model_dir = 'lets_get_small_vessels-v2'
version = 2568


prediction = f'/autofs/cluster/octdata2/users/epc28/veritas/output/{model_dir}/version_{version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii'
ground_truth = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/ground_truth.nii'
tissue_mask = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.pia.nii'
irv = IRV(ground_truth, prediction, threshold=0.5, tissue_mask=tissue_mask)
irv.compute_dice()

ground_truth = '/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-64.nii'
irv = IRV(ground_truth, prediction, threshold=0.5, tissue_mask=tissue_mask)
irv.compute_dice()

#patch_n = [3]
#version_n = [11, 12, 13, 14]
#roi = 'occipital'
#case_n = 26
#
#for i in patch_n:
#    for n in version_n:
#        #rater_1 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/james/seg_DJE_patch_{i}.nii'
#        rater_1 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/james/seg_caa{case_n}-{roi}_patch_{i}.mgz'
#        rater_2 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/etienne/gt_{i}.nii'
#        prediction = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/predictions/version_{n}/patches/prediction_{i}.nii'
#        irv = IRV(rater_1, rater_2, prediction, 0.90)
#        irv.compute_dice()
#    irv.inter_rater_variability()
#    print('\n')

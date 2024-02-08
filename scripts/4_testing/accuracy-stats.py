import torch
import numpy as np
import nibabel as nib


class IRV(object):

    def __init__(self, rater_1, rater_2, prediction, threshold):
        self.gt1 = nib.load(rater_1).get_fdata()
        self.gt2 = nib.load(rater_2).get_fdata()
        self.prediction = nib.load(prediction).get_fdata()
        self.prediction[self.prediction <= threshold] = 0
        self.prediction[self.prediction >= threshold] = 1
        self.prediction = self.prediction.astype(np.int8)
        #self.gt_master = np.clip(self.gt1+self.gt2, 0, 1).astype(np.int8)
        self.gt_master = (self.gt1 * self.gt2).astype(np.int8)
        #self.gt_master = self.gt2

    def compute_dice(self):
        intersection = (self.gt_master * self.prediction).sum()
        hits_in_prediction = self.prediction.sum()
        hits_in_gt = self.gt_master.sum()
        dice = (2 * intersection) / (hits_in_prediction + hits_in_gt)
        print(dice)

    def inter_rater_variability(self):
        intersection = (self.gt1 * self.gt2).sum()
        dice = (2 * intersection) / (self.gt1.sum() + self.gt2.sum())
        print(dice)
        
patch_n = [3]
version_n = [11, 12, 13, 14]
roi = 'occipital'
case_n = 26

for i in patch_n:
    for n in version_n:
        #rater_1 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/james/seg_DJE_patch_{i}.nii'
        rater_1 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/james/seg_caa{case_n}-{roi}_patch_{i}.mgz'
        rater_2 = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/ground_truth/etienne/gt_{i}.nii'
        prediction = f'/autofs/cluster/octdata2/users/epc28/data/CAA/caa{case_n}/{roi}/predictions/version_{n}/patches/prediction_{i}.nii'
        irv = IRV(rater_1, rater_2, prediction, 0.90)
        irv.compute_dice()
    irv.inter_rater_variability()
    print('\n')
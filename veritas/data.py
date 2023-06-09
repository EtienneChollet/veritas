import os
#os.getcwd()

import time
from glob import glob
import nibabel as nib
from torchmetrics.functional import dice

import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("cornucopia")
import cornucopia as cc

#sys.path.append("veritas")
#from veritas import models
import models

class OctVolume(Dataset):

    def __init__(self, volume_path, trainee, tile_size, step_size, subset=-1, transform=None, target_transform=None):
        self.volume_path = volume_path
        self.device = 'cuda'
        self.tile_size = tile_size
        self.step_size = step_size
        self.volume_dtype = torch.float
        self.imprint_dtype = torch.float
        self.trainee = trainee

        # Get all volume specific things
        with torch.no_grad():
            self.volume_nifti = nib.load(self.volume_path)
            self.volume_affine = self.volume_nifti.affine
            self.volume_header = self.volume_nifti.header
            self.volume_tensor = torch.tensor(self.volume_nifti.get_fdata(), device=self.device, dtype=self.volume_dtype)
            self.raw_volume_shape = self.volume_tensor.shape   
            
        # Pad each dimension individually
        #for dim in range(len(self.raw_volume_shape)):
        self.pad_dimension()
        self.imprint_tensor = torch.zeros(self.volume_tensor.shape, dtype=self.imprint_dtype, device=self.device)

        # Partition volume into overlapping 3d patches
        self.get_frame_coords(step_size=self.step_size)
        


    def __len__(self):
        return len(self.coordlist)


    def __getitem__(self, idx):
        working_coords = self.coordlist[idx]
        x_slice = slice(*working_coords[0])
        y_slice = slice(*working_coords[1])
        z_slice = slice(*working_coords[2])

        tile = self.volume_tensor[x_slice, y_slice, z_slice].to(self.volume_dtype).detach()
        prediction = self.trainee(tile.unsqueeze(0).unsqueeze(0).to('cuda'))
        prediction = torch.sigmoid(prediction).squeeze().squeeze().detach()
        #prediction[prediction >= 0.8] = 1
        #prediction[prediction < 0.8] = 0
        
        #prediction = prediction.to(torch.int)
        self.imprint_tensor[x_slice, y_slice, z_slice] += prediction

        return tile, prediction


    def predict(self):
        '''Predict on all patches within 3d volume via getitem function. Normalize resultant imprint and strip padding.'''
        # Normalizing
        #self.volume_tensor = cc.QuantileTransform(pmin=0, pmax=1, vmin=0.05, vmax=0.95, clamp=False)(self.volume_tensor + 0.000001)
        length = self.__len__()
        print("Predicting on", length, 'patches')
        for i in range(length):
            self.__getitem__(i)
            sys.stdout.write(f"\rPrediction {i + 1}/{length}")
            sys.stdout.flush()

        s = slice(self.tile_size, -self.tile_size)

        self.volume_tensor = self.volume_tensor[s, s, s]
        self.imprint_tensor = self.imprint_tensor[s, s, s]
        #self.imprint_tensor[self.imprint_tensor >= threshold] = 1
        #self.imprint_tensor[self.imprint_tensor <= threshold] = 0
        #self.imprint_tensor = self.imprint_tensor.to(torch.int)


    def pad_dimension(self):
        with torch.no_grad():
            self.volume_tensor = self.volume_tensor.clone().detach().unsqueeze(0)
            if len(self.volume_tensor.shape) == 4:
                padding = torch.ones(1, 6, dtype=torch.int) * self.tile_size
                padding = tuple(*padding)
                self.volume_tensor = torch.nn.functional.pad(self.volume_tensor, padding, 'reflect').squeeze()
                #self.volume_tensor = torch.nn.functional.pad(self.volume_tensor, padding, 'constant').squeeze()
                #self.volume_tensor = self.volume_tensor.squeeze()
            else:
                print('Input tensor has shape', self.volume_tensor.shape)


    def get_frame_coords(self, step_size):
        coords = []
        for dim in range(3):
            dim_start_frame = list(np.arange(0, self.volume_tensor.shape[dim] - self.tile_size, step_size))
            # Remove all elements from starting frame list if all they're going to get is padding
            dim_start_frame.remove(0)
            # Remove all elements from starting frame list if all they're going to get is padding
            dim_end_frame = [d + self.tile_size for d in dim_start_frame]
            coords.append(zip(dim_start_frame, dim_end_frame))
            
        for dim in range(len(coords)):
            if dim == 0:
                self.x_coords = [i for i in coords[dim]]
            if dim == 1:
                self.y_coords = [i for i in coords[dim]]
            if dim == 2:
                self.z_coords = [i for i in coords[dim]]
        
        self.coordlist = []
        try:
            for x in self.x_coords:
                for y in self.y_coords:
                    for z in self.z_coords:
                        self.coordlist.append([x, y, z])
        except:
            for x in self.x_coords:
                for y in self.y_coords:
                    self.coordlist.append([x, y])


def findthresh(prediction, ground_truth):
    threshold_lst = np.arange(0.05, 1, 0.05)
    lst = []
    for thresh in threshold_lst:
        prediction_temp = prediction.clone()
        prediction_temp[prediction_temp >= thresh] = 1
        prediction_temp[prediction_temp < thresh] = 0
        dice_coeff = dice(prediction_temp, ground_truth, multiclass=False)
        lst.append(dice_coeff.tolist())

    mx = max(lst)
    mx_idx = lst.index(mx)
    return threshold_lst[mx_idx], lst[mx_idx]


if __name__ == "__main__":
    t1 = time.time()
    #volume_path = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_0/predictions/real_data/oct_volumes/I_mosaic_0_0_0.mgz"
    #volume_path = "/autofs/cluster/octdata2/users/epc28/veritas/sandbox/tiles/volume-0001.nii"
    volume_path = "/cluster/octdata/users/cmagnain/190312_I46_SomatoSensory/I46_Somatosensory_20um_crop.nii"
    model_path = "/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_0"
    out_file = "prediction_16x-avg.nii"

    unet = models.UNet(model_path)
    oct = OctVolume(volume_path, unet.trainee, tile_size=unet.model_params['data']['shape'], step_size=64)

    with torch.no_grad():

        oct.predict()
        x, y = oct.volume_tensor.cpu().numpy(), oct.imprint_tensor.cpu().numpy()

        nifti = nib.nifti1.Nifti1Image(y, affine=oct.volume_affine, header=oct.volume_header)
        savedir = f"{model_path}/predictions/caroline_data/"
        os.makedirs(savedir, exist_ok=True)
        
        ground_truth = f"{savedir}/ground_truth.nii"
        ground_truth_nifti = nib.load(ground_truth)
        ground_truth_tensor = torch.tensor(ground_truth_nifti.get_fdata(), dtype=torch.int).to('cuda')
        ground_truth_tensor[ground_truth_tensor >= 1] = 1

        best_threshold, best_dice_score = findthresh(oct.imprint_tensor, ground_truth_tensor)
        best_dice_score = round(best_dice_score, 3)
        best_threshold = round(best_threshold, 2)

        #dice_coeff = round(dice(oct.imprint_tensor, ground_truth_tensor, multiclass=False).tolist(), 3)
        print("\nThreshold =", best_threshold)
        print('\nDICE =', best_dice_score)

        y = oct.imprint_tensor.cpu().numpy()
        y[y >= best_threshold] = 1
        y[y < best_threshold] = 0

        out_file = f"prediction_stepsz-{step_size}_thresh-{best_threshold}_dice-{best_dice_score}.nii"
        print('\n', f"Saving to {out_file}")
        nib.save(nifti , f"{model_path}/predictions/caroline_data/{out_file}")

    t2 = time.time()
    print(f"\nProcess took {round(t2 - t1, 2)} [sec]")

    #plt.figure()
    ##subplot(r,c) provide the no. of rows and columns
    #f, axarr = plt.subplots(1, 3, figsize=(20, 20), constrained_layout=True)
    #axarr = axarr.flatten()

    #f.suptitle(f'Samples from /autofs/cluster/octdata2/users/epc28/veritas/output/real_data/nonlinearly-augmented', fontsize=15)
    #axarr[0].imshow(np.max(y, axis=0), cmap='magma')
    #axarr[1].imshow(np.max(y, axis=1), cmap='magma')
    #axarr[2].imshow(np.max(y, axis=2), cmap='magma')
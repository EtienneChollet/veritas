__all__ = [
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict'
]
# Standard imports
import os
import sys
import time
import torch
import random
import numpy as np
import nibabel as nib
from veritas.utils import PathTools, volume_info
from torch.utils.data import Dataset

# Custom imports
from veritas.utils import Options

class RealOct(object):
    """
    Base class for real OCT volumetric data.
    """
    def __init__(self,
                 input:{torch.Tensor, str},
                 mask:{torch.Tensor, str}=None,
                 patch_size:int=256,
                 step_size:int=256,
                 binarize:bool=False,
                 binary_threshold:int=0.5,
                 normalize:bool=False,
                 pad_it:bool=False,
                 padding_method:str='replicate', # change to "reflect"
                 device:str='cuda',
                 dtype:torch.dtype=torch.float32,
                 patch_coords_:bool=False,
                 trainee=None
                 ):

        """
        Parameters
        ----------
        input : {torch.Tensor, 'path'}
            Tensor of entire tensor or path to nifti.
        mask : {None, torch.Tensor, 'path'}
        patch_size : int
            Size of patch with which to partition tensor into.
        step_size : int {256, 128, 64, 32, 16}
            Size of step between adjacent patch origin.
        binarize : bool
            Whether to binarize tensor.
        binary_threshold : float
            Threshold at which to binarize (must be used with binarized=True)
        normalize: bool
            Whether to normalize tensor.
        pad_ : bool
            If tensor should be padded.
        padding_method: {'replicate', 'reflect', 'constant'}
            How to pad tensor.
        device: {'cuda', 'cpu'}
            Device to load tensor onto.
        dtype: torch.dtype
            Data type to load tensor as.

        Attributes
        ----------
        volume_nifti
            Nifti represnetation of volumetric data.

        Notes
        -----
        1. Normalize
        2. Binarize
        3. Convert to dtype
        """
        self.input=input
        self.dtype=dtype
        self.patch_size=patch_size
        self.step_size=step_size
        self.binarize=binarize
        self.binary_threshold=binary_threshold
        self.normalize=normalize
        self.device=device
        self.pad_it=pad_it
        self.padding_method=padding_method
        self.tensor, self.nifti, self.affine = self.load_tensor(
            self.input,
            normalize=self.normalize,
            pad_it=self.pad_it
            )
        #self.mask_tensor, self.mask_nifti, self.mask_affine = self.load_tensor(mask)
        
    def load_tensor(
            self,
            input,
            name:str='tensor',
            normalize:bool=False,
            pad_it:bool=False,
            binarize:bool=False
            ):
        """
        Prepare volume.

        Steps
        -----
        1. Load input volume if given path
        2. Convert to float32
        3. Detach from graph
        """
        if isinstance(input, str):
            # Getting name of volume (will be used later for saving prediction)
            self.tensor_name = input.split('/')[-1].strip('.nii')
            # Get directory location of volume (will also use later for saving)
            self.volume_dir = self.input.strip('.nii').strip(self.tensor_name).strip('/')
            nifti = nib.load(input)
            # Load tensor on device with dtype. Detach from graph.
            tensor = torch.as_tensor(
                nifti.get_fdata(), device=self.device, dtype=self.dtype
                ).detach()
            affine = nifti.affine
        elif isinstance(input, torch.Tensor):
            tensor = input.to(self.device).to(self.dtype).detach()
            nifti = None
            # needs identity matrix for affine
        volume_info(tensor, 'Raw')
        if normalize == True:
            tensor = self.normalize_volume(tensor)
        if pad_it == True:
            tensor = self.pad_volume(tensor)
        if self.binarize == True:
            tensor[tensor <= self.binary_threshold] = 0
            tensor[tensor > self.binary_threshold] = 1
        
        return tensor, nifti, affine


    def normalize_volume(self, input):
        input -= input.min()
        input /= input.max()
        return input
        #volume_info(self.tensor, 'Normalized')


    def pad_volume(self, tensor):
        """
        Pad all dimensions of 3 dimensional tensor and update volume.
        """
        # Input tensor must be 4 dimensional [1, n, n, n] to do padding
        padding = [self.patch_size] * 6 # Create 6 ele list of patch size
        tensor = torch.nn.functional.pad(
            input=tensor.unsqueeze(0),
            pad=padding,
            mode=self.padding_method
        )[0]
        volume_info(tensor, 'Padded')
        return tensor
        


class RealOctPatchLoader(RealOct, Dataset):
    """Stuff"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_coords()            

    def __len__(self):
        return len(self.complete_patch_coords)
    
    def __getitem__(self, idx, tensor=None):
        working_patch_coords = self.complete_patch_coords[idx]
        # Generating slices for easy handling
        x_slice = slice(*working_patch_coords[0])
        y_slice = slice(*working_patch_coords[1])
        z_slice = slice(*working_patch_coords[2])
        # Loading patch via coords and detaching from tracking
        if tensor == None:
            patch = self.tensor[x_slice, y_slice, z_slice].detach()
        else:
            tensor[x_slice, y_slice, z_slice].detach()
        coords = [x_slice, y_slice, z_slice]
        return patch, coords
        
    
    def patch_coords(self):
        self.complete_patch_coords = []
        tensor_shape = self.tensor.shape
        # used to be x_coords = [[x, x+self.patch_size] for x in range(0, tensor_shape[0] - self.step_size + 1, self.step_size)]
        x_coords = [[x, x+self.patch_size] for x in range(self.step_size, tensor_shape[0] - self.patch_size, self.step_size)]
        y_coords = [[y, y+self.patch_size] for y in range(self.step_size, tensor_shape[1] - self.patch_size, self.step_size)]
        z_coords = [[z, z+self.patch_size] for z in range(self.step_size, tensor_shape[2] - self.patch_size, self.step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    self.complete_patch_coords.append([x, y, z])


    def random_patch_sampler(self,
                             n_patches:int=10,
                             out_dir:str=None,
                             name_prefix:str='patch',
                             seed:int=None,
                             threshold:float=None,
                             mask:{torch.Tensor, str}=None,
                             mask_id:int=1,
                             mask_threshold:float=0.5,
                             output=None):
        """
        Extract random patches from parent volume and save as nii.

        Parameters
        ----------
        n_patches : int
            Number of patches to extract and save.
        out_dir : str
            Directory to save patches to. Defaults to directory of volume.
        name_prefix : str {'patch', 'prediction'}
            Prefix for naming patches in patch directory. 
        seed : int
            Random seed for selecting patch ID's. If none, optimal seed will be found.
        threshold : float
            Minimum mean value that all patches must have if seed is None. 
        output : {None, 'coords'}
            Determine what to return (by default returns seed)
        """
        if mask is None:
            threshold_tensor=self.tensor
        else:
            threshold_tensor, mask_nifti, mask_affine = self.load_tensor(mask)

        output = output
        coords_list = []

        if out_dir is None:
            out_dir = self.input.split('/')[:-1]
            out_dir.append('patches')
            out_dir = '/'.join(out_dir)
        
        if seed is None:
            print('Finding first best seed according to threshold')
            if threshold is None:
                print('You need to define a threshold to find the best seed :)')
                exit(0)
            threshold = float(threshold)
            keep_going = True
            seed = 0
            while keep_going:
                patch_means = []
                torch.random.manual_seed(seed)
                # Gathering patch indicies for mean analysis
                random_patch_indicies = torch.randint(len(self), [n_patches]).tolist()
                for i in range(len(random_patch_indicies)):
                    patch, coords = self[random_patch_indicies[i]](threshold_tensor)
                    patch_means.append(patch.mean().item())
                least = min(patch_means)
                if least < threshold:
                    keep_going = True
                    seed += 1
                    sys.stdout.write(f"\rTrying seed {seed}")
                    sys.stdout.flush()
                if least >= threshold:
                    keep_going = False
                    print(f'\nGot it! seed {seed} is good!!!')
                    print(least)
                    print(coords)
                    break

        elif isinstance(seed, int):
            print(f'Using user defined seed (seed {seed})')
            torch.random.manual_seed(seed)
            random_patch_indicies = torch.randint(len(self), [n_patches]).tolist()
        else:
            print('What do I do with that seed! Gimme an int!')

        PathTools(out_dir).makeDir()
        for i in range(len(random_patch_indicies)):
            patch, coords = self[random_patch_indicies[i]]
            coords = [coords[0].start, coords[1].start, coords[2].start]
            coords_list.append(coords)

            aff = np.copy(self.affine)
            M = aff[:3, :3]
            abc = aff[:3, 3]
            abc += np.diag(M * coords)

            #patch_affine = np.copy(self.affine)
            #patch_affine[0][3] += (coords[0].start * 0.02)
            #patch_affine[1][3] += (coords[1].start * 0.02)
            #patch_affine[2][3] += (coords[2].start * 0.02)

            patch = nib.nifti1.Nifti1Image(patch.cpu().numpy(), affine=aff)
            nib.save(patch, f'{out_dir}/{name_prefix}_{i}.nii')
        print(f'Saved patches to {out_dir}')

        if output == None:
            return seed
        elif output == 'coords':
            for set in coords_list:
                print(set)


class RealOctPredict(RealOctPatchLoader, Dataset):

    def __init__(self, trainee=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trainee = trainee
        self.imprint_tensor = torch.zeros(
            self.tensor.shape, device=self.device, dtype=self.dtype
            )

    def __getitem__(self, idx:int):
        """
        Predict on a single patch.

        Parameters
        ----------
        idx : int
            Patch ID number to predict on. Updates self.imprint_tensor.
        """     
        patch, coords = super().__getitem__(idx)
        ## Needs to go on cuda for prediction
        prediction = self.trainee(patch.unsqueeze(0).unsqueeze(0))
        prediction = torch.sigmoid(prediction).squeeze()
        if self.device == 'cpu':
            prediction = prediction.to('cpu')

        backend = dict(dtype=prediction.dtype, device=prediction.device)
        #weight = torch.linspace(0, 3.14, self.patch_size, **backend).sin()
        #weight = weight[None, None, :] * weight[None, :, None] * weight[:, None, None]

        scale = 1/4
        min = torch.pi * scale
        max = ((1/scale) - 1) * (torch.pi * scale)
        
        weight_x = torch.linspace(min, max, prediction.shape[0], **backend).sin()
        weight_y = torch.linspace(min, max, prediction.shape[1], **backend).sin()
        weight_z = torch.linspace(min, max, prediction.shape[2], **backend).sin()
        weight = weight_x[:, None, None] * weight_y[None, :, None] * weight_z[None, None, :]
        self.imprint_tensor[coords[0], coords[1], coords[2]] += (prediction * weight)
        # some issues with inconsistent sizes. Using broadcasting
        #self.imprint_weight[coords[0], coords[1], coords[2]] += weight 
        #self.imprint_tensor[coords[0], coords[1], coords[2]] += prediction

    def predict_on_all(self):
        if self.tensor.dtype != torch.float32:
            self.tensor = self.tensor.to(torch.float32)
        if self.device != 'cuda':
            self.device == 'cuda'
            self.tensor = self.tensor.to('cuda') 
        n_patches = len(self)
        t0 = time.time()
        for i in range(n_patches):
            self[i]
            if (i+1) % 10 == 0:
                total_elapsed_time = time.time() - t0
                average_time_per_pred = round(total_elapsed_time / (i+1), 2)
                sys.stdout.write(f"\rPrediction {i + 1}/{n_patches} | {average_time_per_pred} sec/pred | {round(average_time_per_pred * n_patches / 60, 2)} min total pred time")
                sys.stdout.flush()

        # Step size, then number to divide by
        #avg_factors = {256:1, 128:8, 64:64, 32:512, 16:4096}
        patchsize_to_stepsize = self.patch_size // self.step_size
        #print('\n', patchsize_to_stepsize)
        # for patchsze=256: avg_factor(stepsize=[256,128,64,32]) = [1,8,64,512]
        if self.patch_size == 256:
            avg_factor = 8 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 128:
            avg_factor = 8 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 64:
            #avg_factor = 8 ** (patchsize_to_stepsize - 1)
            factors = {1:1, 2:8, 4:64, 8:512}
            avg_factor = factors[patchsize_to_stepsize]
        else:
            avg_factor = 1

        # Remove padding
        s = slice(self.patch_size, -self.patch_size)
        self.imprint_tensor = self.imprint_tensor[s, s, s]
        #self.imprint_weight = self.imprint_weight[s, s, s]
        #self.imprint_tensor *= self.imprint_weight

        print(f"\n\n{avg_factor}x Averaging...")
        self.imprint_tensor /= avg_factor
        self.imprint_tensor /= self.imprint_tensor.max()
        print(f"Prediction Minimum: {self.imprint_tensor.min()}")
        print(f"Prediction Maximum: {self.imprint_tensor.max()}")

    def save_prediction(self, dir=None):
        """
        Save prediction volume.

        Parameters
        ----------
        dir : str
            Directory to save volume. If None, it will save volume to same path.
        """
        self.out_dir, self.full_path = Options(self).out_filepath(dir)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"\nSaving prediction to {self.full_path}...")
        self.imprint_tensor = self.imprint_tensor.cpu().numpy()        
        out_nifti = nib.nifti1.Nifti1Image(dataobj=self.imprint_tensor, affine=self.affine)
        nib.save(out_nifti, self.full_path)
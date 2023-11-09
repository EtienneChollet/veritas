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
import nibabel as nib
from torch.utils.data import Dataset

# Custom imports
from veritas.utils import Options

class RealOct(object):
    """
    Base class for real OCT volumetric data.
    """
    def __init__(self,
                 input:torch.Tensor | str,
                 device:str='cuda',
                 dtype:torch.dtype=torch.float32,
                 patch_size:int=256,
                 step_size:int=256,
                 normalize:bool=False,
                 binarize:bool=False,
                 pad:bool=False,
                 padding_method:str='replicate',
                 kill_nifti:bool=True,
                 verbose:bool=False
                 ):

        """
        Parameters
        ----------
        input : tensor or 'path'
            Tensor of volume or path to nifti.
        device : {'cuda', 'cpu'}
            Device to load and hold data.
        dtype : dtype
            Data type to load volume as.
        patch_size : int
            Size of patch with which to partition volume into.
        step_size : int
            Size of step between adjacent patch origin.
        normalize : bool
            Whether to normalize volume.
        binarize : bool
            Whether to binarize volume. If float, sets threshold to value.
        p_bounds : list[float]
            Bounds for normalization percentile (only if normalize=True).
        v_bounds : list[float]
            Bounds for histogram after normalization (only if normalize=True).
        device : {'cuda', 'cpu'}
            Device to load volume onto.
        padding_method : {'replicate', 'reflect', 'constant'}
            How to pad volume.
        kill_nifti : bool
            Remove nifti from memory after loading.
        verbose : bool
            Print tensor details at each step.

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
        self.device=device
        self.dtype=dtype
        self.patch_size=patch_size
        self.step_size=step_size
        self.normalize=normalize
        self.binarize=binarize
        self.padding_method=padding_method
        self.load_volume()
        self.affine = self.nifti.affine
        self.verbose = verbose
        if kill_nifti == True:
            self.nifti = None
        with torch.no_grad():
            if self.normalize == True:
                self.normalize_volume()
            if self.binarize != False:
                self.binarize_volume()
            if pad == True:
                self.pad_volume()


    def load_volume(self):
        """
        Prepare volume.

        Steps
        -----
        1. Load
        2. Move to device
        3. Change dtype
        4. Detach from graph
        """
        if isinstance(self.input, str):
            # Getting name of volume (will be used later for saving prediction)
            self.volume_name = self.input.split('/')[-1].strip('.nii')
            # Get directory location of volume (will also use later for saving)
            self.volume_dir = self.input.strip('.nii').strip(self.volume_name).strip('/')
            self.nifti = nib.load(self.input)
            # Load tensor on device with dtype. Detach from graph.
            self.tensor = torch.as_tensor(
                self.nifti.get_fdata(), device=self.device, dtype=self.dtype
                ).detach()
        elif isinstance(self.input, torch.Tensor):
            self.tensor = self.input.to(self.device).to(self.dtype).detach()
            self.nifti = None
        #volume_info(self.tensor, 'Raw')


    def normalize_volume(self):
        self.tensor -= self.tensor.min()
        self.tensor /= self.tensor.max()
        #volume_info(self.tensor, 'Normalized')

    def binarize_volume(self):
        if isinstance(self.binarize, float):
            self.tensor[self.tensor >= self.binarize] = 1
            self.tensor[self.tensor <= self.binarize] = 0
        elif self.binarize == True:
            self.tensor[self.tensor >= 0.5] = 1
            self.tensor[self.tensor <= 0.5] = 0


    def pad_volume(self):
        """
        Pad all dimensions of 3 dimensional tensor and update volume.
        """
        # Input tensor must be 4 dimensional [1, n, n, n] to do padding
        padding = [self.patch_size] * 6 # Create 6 ele list of patch size
        self.tensor = torch.nn.functional.pad(
            input=self.tensor.unsqueeze(0),
            pad=padding,
            mode='replicate'
        )[0]
        self.padded_shape = self.tensor.shape
        #volume_info(self.tensor, 'Padded')


class RealOctPatchLoader(RealOct, Dataset):
    """Stuff"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_coords()            

    def __len__(self):
        return len(self.complete_patch_coords)
    
    def __getitem__(self, idx):
        working_patch_coords = self.complete_patch_coords[idx]
        # Generating slices for easy handling
        x_slice = slice(*working_patch_coords[0])
        y_slice = slice(*working_patch_coords[1])
        z_slice = slice(*working_patch_coords[2])
        # Loading patch via coords and detaching from tracking
        patch = self.tensor[x_slice, y_slice, z_slice].detach()
        coords = [x_slice, y_slice, z_slice]
        return patch, coords
        
    
    def patch_coords(self):
        self.complete_patch_coords = []
        x_coords = [[x, x+self.patch_size] for x in range(0, self.padded_shape[0] - self.step_size + 1, self.step_size)]
        y_coords = [[y, y+self.patch_size] for y in range(0, self.padded_shape[1] - self.step_size + 1, self.step_size)]
        z_coords = [[z, z+self.patch_size] for z in range(0, self.padded_shape[2] - self.step_size + 1, self.step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    self.complete_patch_coords.append([x, y, z])


class RealOctPredict(RealOctPatchLoader, Dataset):

    def __init__(self, trainee=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with torch.no_grad():
            self.trainee = trainee
            self.imprint_tensor = torch.zeros(
                self.padded_shape, device=self.device, dtype=self.dtype
                ).detach()


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
        prediction = self.trainee(patch.unsqueeze(0).unsqueeze(0)).detach()
        prediction = torch.sigmoid(prediction).squeeze()
        if self.device == 'cpu':
            prediction = prediction.to('cpu')
        self.imprint_tensor[coords[0], coords[1], coords[2]] += prediction 
    
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
        # for patchsze=256: avg_factor(stepsize=[256,128,64,32]) = [1,8,64,512]
        if self.patch_size == 256:
            avg_factor = 8 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 128:
            avg_factor = 4 ** (patchsize_to_stepsize - 1)
        elif self.patch_size == 64:
            avg_factor = 2 ** (patchsize_to_stepsize - 1)
        else:
            avg_factor =1
        print(f"\n\n{avg_factor}x Averaging...")
        self.imprint_tensor = self.imprint_tensor / avg_factor
        s = slice(self.patch_size, -self.patch_size)
        self.imprint_tensor = self.imprint_tensor[s, s, s]


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
        #print(self.imprint_tensor.shape)
        #print(self.imprint_tensor.max())

        out_nifti = nib.nifti1.Nifti1Image(dataobj=self.imprint_tensor, affine=self.affine)
        nib.save(out_nifti, self.full_path)
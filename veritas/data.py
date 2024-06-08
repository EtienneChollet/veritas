__all__ = [
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict',
    'RealOctDataset'
]

# Standard library imports
from glob import glob
import os
import gc
import random
import sys
import time

# Third-party imports
import nibabel as nib
import numpy as np
import skimage
import sklearn
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset
import torchvision
from scipy import ndimage
from typing import Union, Optional, Tuple, List


# Local application/library specific imports
from cornucopia.cornucopia.intensity import QuantileTransform
from veritas.utils import Options, PathTools, volume_info
from veritas.models import PatchPredict



class RealOct(object):
    """
    Base class for real OCT volumetric data.
    """
    def __init__(
        self,
        input: Union[torch.Tensor, str],
        patch_size: int = 128,
        redundancy: int = 3,
        binarize: bool = False,
        binary_threshold: float = 0.5,
        normalize: bool = False,
        pad_it: bool = False,
        padding_method: str = 'reflect',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32
    ):
        """
        Parameters
        ----------
        input : Union[torch.Tensor, str]
            A tensor containing the entire dataset or a string path to a NIfTI
            file.
        patch_size : int
            Size of the patches into which the tensor is divided.
        step_size : int, optional
            Distance between the origins of adjacent patches. Typical values
            might include 256, 128, 64, 32, or 16. Default is 256.
        binarize : bool, optional
            Indicates whether to binarize the tensor. If True,
            `binary_threshold` must be specified. Default is False.
        binary_threshold : float, optional
            The threshold value for binarization. Only used if `binarize` is
            True.
        normalize : bool, optional
            Specifies whether to normalize the tensor. Default is False.
        pad_it : bool, optional
            If True, the tensor will be padded using the method specified by
            `padding_method`. Default is False.
        padding_method : {'replicate', 'reflect', 'constant'}, optional
            Specifies the method to use for padding. Default is 'reflect'.
        device : {'cuda', 'cpu'}, optional
            The device on which the tensor is loaded. Default is 'cuda'.
        dtype : torch.dtype, optional
            The data type of the tensor when loaded into a PyTorch tensor.
            Default is `torch.float32`.

        Attributes
        ----------
        volume_nifti : nib.Nifti1Image or None
            Represents the NIfTI image of the volumetric data if loaded from a
            file, otherwise None.

        Notes
        -----
        - The tensor is normalized if `normalize` is set to True.
        - The tensor is binarized using `binary_threshold` if `binarize` is set
            to True.
        - The tensor data type is converted according to the `dtype` parameter.
        """

        self.input = input
        self.patch_size = patch_size
        self.redundancy = redundancy - 1
        self.step_size = int(patch_size * (1 / (2**self.redundancy)))
        self.binarize = binarize
        self.binary_threshold = binary_threshold
        self.normalize = normalize
        self.pad_it = pad_it
        self.padding_method = padding_method
        self.device = device
        self.dtype = dtype
        self.tensor, self.nifti, self.affine = self.load_tensor()

    def load_tensor(self) -> Tuple[
        torch.Tensor, Optional[nib.Nifti1Image], Optional[np.ndarray]
    ]:
        """
        Loads and processes the input volume, applying normalization, padding, 
        and binarization as specified.

        Returns
        -------
        torch.Tensor
            The processed tensor data, either loaded from a NIfTI file or
            received directly, and transformed to the specified device and
            dtype.
        Optional[nib.Nifti1Image]
            The original NIfTI volume if the input is a file path, otherwise
            None.
        Optional[np.ndarray]
            The affine transformation matrix of the NIfTI image if the input is
            a file path, otherwise None.

        Notes
        -----
        - If `input` is a path, the NIfTI file is loaded and the data is
        converted to the specified dtype and moved to the specified device.
        - Normalization rescales tensor values to the 0-1 range if `normalize`
        is True.
        - Padding adds specified borders around the data if `pad_it` is True.
        - Binarization converts data to 0 or 1 based on `binary_threshold` if
        `binarize` is True.
        """
        tensor, nifti, affine = None, None, None
        if isinstance(self.input, str):
            self.tensor_name = self.input.split('/')[-1].strip('.nii')
            base_name = self.input.strip('.nii')
            clean_name = base_name.strip(self.tensor_name).strip('/')
            self.volume_dir = f"/{clean_name}"
            nifti = nib.load(self.input)
            tensor = torch.from_numpy(nifti.get_fdata()).to(
                self.device, dtype=torch.float32)
            affine = nifti.affine
        else:
            tensor = self.input.to(self.device, self.dtype)

        if self.normalize:
            tensor = self.normalize_volume(tensor)
        if self.pad_it:
            tensor = self.pad_volume(tensor)
        if self.binarize:
            tensor = torch.where(
                tensor > self.binary_threshold, 
                torch.tensor(1.0, device=self.device),
                torch.tensor(0.0, device=self.device)
            )
        return tensor, nifti, affine

    def normalize_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize the tensor to the range 0 to 1.
        """
        tensor -= tensor.min()
        tensor /= tensor.max()
        return tensor
    
    def pad_volume(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies symmetric padding to a tensor to increase its dimensions,
        ensuring that its size is compatible with the specified `patch_size`.

        Parameters
        ----------
        tensor : torch.Tensor
            The tensor to be padded.

        Returns
        -------
        torch.Tensor
            The tensor after applying symmetric padding.

        Notes
        -----
        Padding is added symmetrically to all dimensions of the input tensor 
        based on half of the `patch_size`. The padding mode used is determined 
        by the `padding_method` attribute, which can be 'replicate', 'reflect',
        or 'constant'.
        """
        # Ensures padding does not exceed tensor dimensions
        padded_tensor = torch.nn.functional.pad(
            input=tensor.unsqueeze(0),
            pad=[self.patch_size] * 6,
            mode=self.padding_method
        ).squeeze()
        return padded_tensor
    
    
    def make_mask(self, n_clusters=2):
        """
        Make tissue masks

        n_clusters : int
            Number of unique intensity values to extract
        mode : {'seperate', 'unified'}
            Which masks to return. 'seperate' returns tissue mask, wm mask,
            and gm mask. Unified returns all 3.
        """
        def make_tissue_mask(n_clusters):
            print('Making whole-tissue mask...')
            # Smoothing data with gaussian filter
            preprocessed_vol = torch.clone(self.tensor).to('cuda')
            preprocessed_vol = torchvision.transforms.functional.gaussian_blur(
                preprocessed_vol, 15, sigma=1e2
                )
            # Sampling volume and computing quantile
            selection_tensor = torch.randint(
                0, 100, preprocessed_vol.shape).to('cuda')
            selection = preprocessed_vol[selection_tensor == 5]
            selection = selection[selection != 0]
            q = torch.quantile(selection, 0.3)
            preprocessed_vol[preprocessed_vol < q] = 0
            # Freeing up some gpu memory!
            del selection_tensor
            del selection
            torch.cuda.empty_cache()

            # Normalizing and simplifying data
            preprocessed_vol -= preprocessed_vol.min()
            preprocessed_vol /= preprocessed_vol.max()
            #preprocessed_vol = preprocessed_vol.to(torch.float64)
            preprocessed_vol *= (256 ** 2)
            preprocessed_vol -= (256 ** 2) / 2
            preprocessed_vol = preprocessed_vol.to(torch.int16)

            X = torch.clone(preprocessed_vol).reshape(-1, 1)
            a = torch.arange(start=0, end=X.shape[0], step=1e6).to(torch.int32)
            c = [slice(a[i], a[i+1]) for i in range(len(a) - 1)]
            x = X[c[0]]
            for i in range(len(c)):
                if i % 20 == 0:
                    x = torch.concatenate((x, X[c[i]]))

            kmeans = sklearn.cluster.KMeans(n_clusters=2, n_init=3)
            y_pred = kmeans.fit(x.cpu())
            complete_pred = kmeans.predict(
                preprocessed_vol.cpu().reshape(-1, 1)
                )
            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
            lut = np.zeros_like(idx)
            lut[idx] = np.arange(n_clusters)
            labels = lut[complete_pred]
            labels = labels.reshape(preprocessed_vol.shape)
            labels = labels.astype(np.int8)

            eroded = ndimage.binary_erosion(labels, iterations=10)

            del preprocessed_vol
            del labels
            torch.cuda.empty_cache()

            eroded = torch.from_numpy(eroded).to('cuda')
            eroded_blurred = torchvision.transforms.functional.gaussian_blur(
                eroded, 9, sigma=1e2
                )
            final_pred = eroded_blurred.clip(0)

            del eroded
            del eroded_blurred
            torch.cuda.empty_cache()

            return final_pred
        

        def make_matter_masks(tissue_mask, n_clusters=3):
            print('Making WM & GM masks...')
            tissue_masked = tissue_mask.cuda() * self.tensor.cuda()
            tissue = torchvision.transforms.functional.gaussian_blur(
                tissue_masked, 15, sigma=1e2
                )
            del tissue_mask
            del tissue_masked
            torch.cuda.empty_cache()

            tissue -= tissue.min()
            tissue /= tissue.max()
            tissue *= (255 ** 2)
            tissue -= (255 ** 2) / 2
            tissue = tissue.to(torch.int16)

            X = torch.clone(tissue).reshape(-1, 1)
            a = torch.arange(start=0, end=X.shape[0], step=1e6).to(torch.int32)
            c = [slice(a[i], a[i+1]) for i in range(len(a) - 1)]

            x = X[c[0]]
            for i in range(len(c)):
                if i % 20 == 0:
                    x = torch.concatenate((x, X[c[i]]))

            kmeans = sklearn.cluster.KMeans(n_clusters=3, n_init=3)
            y_pred = kmeans.fit(x.cpu().numpy())
            complete_pred = kmeans.predict(tissue.cpu().reshape(-1, 1))

            idx = np.argsort(kmeans.cluster_centers_.sum(axis=1))
            lut = np.zeros_like(idx)
            lut[idx] = np.arange(n_clusters)
            labels = lut[complete_pred]
            labels = labels.reshape(tissue.shape)
            labels = torch.from_numpy(labels).to(torch.uint8)

            gm_mask = torch.clone(labels)
            wm_mask = torch.clone(labels)

            gm_mask[gm_mask != 1] = 0
            wm_mask[wm_mask != 2] = 0
            wm_mask[wm_mask == 2] = 1

            return gm_mask, wm_mask
        
        tissue_mask = make_tissue_mask(n_clusters)
        gm_mask, wm_mask = make_matter_masks(tissue_mask)

        return tissue_mask, gm_mask, wm_mask


    def get_mask(self, mask_type='tissue-mask'):
        """"
        Load mask. Makes mask if does not exist in filesystem.

        Parameters
        ----------
        mask_type : {'tissue-mask', 'gm-mask', 'wm-mask'}
            Type of mask to load.
        """
        mask_types = ['tissue-mask', 'gm-mask', 'wm-mask']
        if mask_type not in mask_types:
            print('Invalid mask type!')
            exit(0)
        mask_path = f"{self.volume_dir}/kmeans-{mask_type}.nii"

        if os.path.exists(mask_path):
            mask = nib.load(mask_path)
            return mask
        else:
            print("Looks like I'll make the masks myself then!")
            masks = self.make_mask()
            for i in range(len(mask_types)):
                nifti = nib.nifti1.Nifti1Image(
                    masks[i].cpu().numpy().astype(np.uint8), affine=self.affine
                    )
                nib.save(nifti,
                         f'{self.volume_dir}/kmeans-{mask_types[i]}.nii'
                         )
            return masks[mask_types.index(mask_type)]
        


class RealOctPatchLoader(RealOct, Dataset):
    """
    A subclass for loading 3D volume patches efficiently using PyTorch, 
    optimized to work with GPU. It inherits from RealOct and Dataset, and it
    extracts specific patches defined by spatial coordinates.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the loader, setting up the internal structure and 
        computing the coordinates for all patches to be loaded.
        """
        super().__init__(*args, **kwargs)
        self.patch_coords()

    def __len__(self):
        """
        Return the total number of patches.
        """
        return len(self.complete_patch_coords)

    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve a patch by index.
        
        Parameters:
            idx (int): The index of the patch to retrieve.

        Returns:
            tuple: A tuple containing the patch tensor and its slice indices.
        """
        x_slice, y_slice, z_slice = self.complete_patch_coords[idx]
        patch = self.tensor[x_slice, y_slice, z_slice].detach().cuda()
        return patch, (x_slice, y_slice, z_slice)
    
    def patch_coords(self):
        """
        Computes the coordinates for slicing the tensor into patches based on 
        the defined patch size and step size. This method populates the 
        complete_patch_coords list with slice objects.
        """
        self.complete_patch_coords = []
        tensor_shape = self.tensor.shape
        x_coords = [slice(x, x + self.patch_size) for x in range(
            self.step_size, tensor_shape[0] - self.patch_size, self.step_size)]
        y_coords = [slice(y, y + self.patch_size) for y in range(
            self.step_size, tensor_shape[1] - self.patch_size, self.step_size)]
        z_coords = [slice(z, z + self.patch_size) for z in range(
            self.step_size, tensor_shape[2] - self.patch_size, self.step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    self.complete_patch_coords.append((x, y, z))


    def random_patch_sampler(self,
                             n_patches:int=10,
                             out_dir:str=None,
                             name_prefix:str='patch',
                             seed:int=None,
                             threshold:float=None,
                             save_patches:bool=True,
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
            Random seed for selecting patch ID's. If none, optimal seed will be
            found.
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
                print("""You need to define a threshold to find the best seed
                       :)""")
                exit(0)
            threshold = float(threshold)
            keep_going = True
            seed = 0
            while keep_going:
                patch_means = []
                torch.random.manual_seed(seed)
                # Gathering patch indicies for mean analysis
                random_patch_indicies = torch.randint(
                    len(self), [n_patches]
                    ).tolist()
                for i in range(len(random_patch_indicies)):
                    patch, coords = self[random_patch_indicies[i]]
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
            random_patch_indicies = torch.randint(
                len(self), [n_patches]
                ).tolist()
        else:
            print('What do I do with that seed! Gimme an int!')
        # Saving patches
        if save_patches == True:
            PathTools(out_dir).makeDir()        
            for i in range(len(random_patch_indicies)):
                patch, coords = self[random_patch_indicies[i]]
                coords = [coords[0].start, coords[1].start, coords[2].start]
                coords_list.append(coords)

                aff = np.copy(self.affine)
                M = aff[:3, :3]
                abc = aff[:3, 3]
                abc += np.diag(M * coords)

                patch = patch.to(self.dtype).cpu().numpy()
                patch = nib.nifti1.Nifti1Image(patch, affine=aff)
                nib.save(patch, f'{out_dir}/{name_prefix}_{i}.nii')
            print(f'Saved patches to {out_dir}')
        elif save_patches == False:
            print("Fine, I won't save them")

        if output == None:
            return seed
        elif output == 'coords':
            for set in coords_list:
                print(set)


class RealOctPredict(RealOctPatchLoader, Dataset):
    """
    This class extends RealOctPatchLoader for prediction purposes, applying
    normalization and custom weighting to patches for use in a machine learning
    prediction script.

    Parameters
    ----------
    trainee : Optional[torch.nn.Module], optional
        The machine learning model that will be testing on the patches,
        by default None.
    normalize_patches : bool, optional
        Whether to normalize patches before processing, by default True.

    Attributes
    ----------
    backend : dict
        Configuration for tensor data types and device.
    imprint_tensor : torch.Tensor
        A tensor initialized to zero, with the same shape as the dataset's
        main tensor, used for storing patch imprints.
    patch_weight : torch.Tensor
        A 3D weight matrix created by the outer product of a sine wave filter,
        designed to attenuate edge prediction signal.
    """
    def __init__(self, trainee: torch.nn.Module = None,
                 normalize_patches: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = {'dtype': self.dtype, 'device': self.device}
        self.trainee = trainee.eval()
        self.imprint_tensor = torch.zeros(self.tensor.shape, **self.backend)
        self.normalize_patches = normalize_patches
        self.patch_weight = self._prepare_patch_weights()


    def _prepare_patch_weights(self, min_scale: float = 0.5
                                ) -> torch.Tensor:
        """
        Create a 3D patch weight by the outer product of the 1D filter.

        Parameters
        ----------
        size : int
            Size of the patch.

        Returns
        -------
        torch.Tensor
            3D weights tensor for patches.
        """
        half_filter_1d = torch.linspace(
            min_scale, torch.pi/2, self.patch_size // 2,
            device=self.device).sin()
        filter_1d = torch.concat([half_filter_1d, half_filter_1d.flip(0)])
        patch_weight = (
            filter_1d[:, None, None]
            * filter_1d[None, :, None]
            * filter_1d[None, None, :]).cuda()
        return patch_weight
        
    def __getitem__(self, idx:int):
        """
        Predict on a single patch, optimized fort GPU.

        Parameters
        ----------
        idx : int
            Patch ID number to predict on. Updates self.imprint_tensor.
        """
        with torch.no_grad():
            patch, coords = super().__getitem__(idx)
            ## Needs to go on cuda for prediction. Useful when predicting on 
            # large volumes that are on CPU.
            if self.device != 'cuda':
                patch = patch.to('cuda')
            patch = patch.unsqueeze(0).unsqueeze(0)
            if self.normalize_patches == True:
                try:
                    #0.2, 0.8
                    patch = QuantileTransform(vmin=0.2, vmax=0.8)(patch.float())
                except:
                    patch -= patch.min()
                    patch /= patch.max()
            prediction = self.trainee(patch)
            prediction = torch.sigmoid(prediction).squeeze()
            weighted_prediction = (
                prediction * self.patch_weight
                )
            self.imprint_tensor[
                coords[0], coords[1], coords[2]
                ] += weighted_prediction

    def predict_on_all(self):
        if self.tensor.dtype != torch.float32:
            self.tensor = self.tensor.to(torch.float32)
            print('Input tensor needs to be float32!!')
        n_patches = len(self)
        t0 = time.time()
        print('Starting predictions!!')
        for i in range(n_patches):
            self[i]
            if (i+1) % 10 == 0:
                total_elapsed_time = time.time() - t0
                avg_pred_time = round(total_elapsed_time / (i+1), 3)
                total_pred_time = round(avg_pred_time * n_patches / 60, 2)
                # Construct the status message
                status_message = (
                    f"\rPrediction {i + 1}/{n_patches} | "
                    f"{avg_pred_time} sec/pred | "
                    f"{total_pred_time} min total pred time"
                    )
                sys.stdout.write(status_message)
                sys.stdout.flush()

        # Remove padding
        s = slice(self.patch_size, -self.patch_size)
        self.imprint_tensor = self.imprint_tensor[s, s, s]
        redundancy = ((self.patch_size ** 3) // (self.step_size ** 3))

        print(f"\n\n{redundancy}x Averaging...")
        self.imprint_tensor /= redundancy
        self.imprint_tensor = self.imprint_tensor.cpu().numpy()


    def save_prediction(self, dir=None):
        """
        Save prediction volume.

        Parameters
        ----------
        dir : str
            Directory to save volume. If None, it will save volume to same
            path.
        """
        self.out_dir, self.full_path = Options(self).out_filepath(dir)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"\nSaving prediction to {self.full_path}...")
        out_nifti = nib.nifti1.Nifti1Image(
            dataobj=self.imprint_tensor,
            affine=self.affine)
        nib.save(out_nifti, self.full_path)


class RealOctPredictLightweight(RealOct, Dataset):
        """
        A PyTorch Dataset that extracts patches from a 3D volume with 
        overlapping and reflection padding, suitable for training a neural
        network model.

        Parameters
        ----------
        trainee : torch.nn.Module, optional
            The neural network model that will be used to predict on patches.
        patch_size : int, default 128
            The edge length of the cube used for patches (D, H, W are equal).
        redundancy : int, default 3
            Overlap redundancy factor to determine step size.
        normalize_patches : bool, default True
            Whether to normalize the patches.
        """
        def __init__(self, trainee: torch.nn.Module = None,
                     patch_size: int = 128, redundancy: int = 3,
                     normalize_patches: bool = True, 
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.trainee = trainee
            self.patch_size = [patch_size] * 3
            self.redundancy = redundancy-1
            self.normalize_patches = normalize_patches
            self.step_size = [int(patch_size * (1 / (2**self.redundancy)))] * 3
            self.imprint_tensor = torch.zeros(
                self.tensor.shape, device=self.device
                )
            self.patch_weight = self._prepare_patch_weights()
            self.num_patches = [
                (self.tensor.shape[i] + (2*self.patch_size[i])) 
                // self.step_size[i]
                for i in range(3)
            ]

        def _prepare_patch_weights(self, min_scale: float = 0.5
                                   ) -> torch.Tensor:
            """
            Create a 3D patch weight by the outer product of the 1D filter.

            Parameters
            ----------
            size : int
                Size of the patch.

            Returns
            -------
            torch.Tensor
                3D weights tensor for patches.
            """
            half_filter_1d = torch.linspace(
                min_scale, torch.pi/2, self.patch_size[0] // 2,
                device=self.device).sin()
            filter_1d = torch.concat([half_filter_1d, half_filter_1d.flip(0)])
            patch_weight = (
                filter_1d[:, None, None]
                * filter_1d[None, :, None]
                * filter_1d[None, None, :]).cuda()
            return patch_weight

        def __len__(self):
            # Total number of patches
            return (self.num_patches[0] 
                    * self.num_patches[1] 
                    * self.num_patches[2])

        def __getitem__(self, idx):
            # Calculate the patch indices in 3D grid
            z = ((idx // (self.num_patches[1] 
                          * self.num_patches[2])) 
                          % self.num_patches[0]) - self.redundancy
            y = ((idx // self.num_patches[2]) 
                 % self.num_patches[1]) - self.redundancy
            x = (idx % self.num_patches[2]) - self.redundancy
            step_size = self.step_size
            patch_size = self.patch_size
            tensor_size = self.tensor.size()

            # Calculate the start and end indices for each dimension
            start_z = z * step_size[0] - step_size[0]
            start_y = y * step_size[1] - step_size[1]
            start_x = x * step_size[2] - step_size[2]
            end_z = start_z + patch_size[0]
            end_y = start_y + patch_size[1]
            end_x = start_x + patch_size[2]
            # Calculate padding needs for start and end of each dimension
            pad_before_z = max(0, -start_z)
            pad_after_z = max(0, end_z - self.tensor.size(0))
            pad_before_y = max(0, -start_y)
            pad_after_y = max(0, end_y - self.tensor.size(1))
            pad_before_x = max(0, -start_x)
            pad_after_x = max(0, end_x - self.tensor.size(2))

            patch = self.tensor[
                pad_before_z+start_z:end_z-pad_after_z,
                pad_before_y+start_y:end_y-pad_after_y,
                pad_before_x+start_x:end_x-pad_after_x
                ]
    
            if torch.count_nonzero(torch.tensor(patch.shape)) > 2:
                pad = (pad_before_x, pad_after_x, pad_before_y,
                       pad_after_y, pad_before_z, pad_after_z)
                patch = torch.nn.functional.pad(
                    +patch.unsqueeze(0), pad=pad, mode=self.padding_method)
                if self.normalize_patches == True:
                    try:
                        patch = QuantileTransform()(patch.float())
                    except:
                        pass
                prediction = self.trainee(patch.unsqueeze(0))
                prediction = torch.sigmoid(prediction).squeeze()
                weighted_prediction = (
                    prediction * self.patch_weight
                )
                # Trimming prediction to fit inside imprint
                weighted_prediction = weighted_prediction[
                    pad_before_z:self.patch_size[0] - pad_after_z,
                    pad_before_y:self.patch_size[1] - pad_after_y,
                    pad_before_x:self.patch_size[2] - pad_after_x
                ]
                self.imprint_tensor[
                    pad_before_z+start_z : end_z-pad_after_z,
                    pad_before_y+start_y:end_y-pad_after_y,
                    pad_before_x+start_x:end_x-pad_after_x
                    ] += weighted_prediction
                
        def predict_on_all(self):
            if self.tensor.dtype != torch.float32:
                self.tensor = self.tensor.to(torch.float32)
            n_patches = len(self)
            t0 = time.time()
            print('Starting predictions!!')
            for i in range(n_patches):
                self[i]
                if (i+1) % 10 == 0:
                    total_elapsed_time = time.time() - t0
                    avg_pred_time = round(total_elapsed_time / (i+1), 3)
                    total_pred_time = round(avg_pred_time * n_patches / 60, 2)
                    # Construct the status message
                    status_message = (
                        f"\rPrediction {i + 1}/{n_patches} | "
                        f"{avg_pred_time} sec/pred | "
                        f"{total_pred_time} min total pred time"
                        )
                    sys.stdout.write(status_message)
                    sys.stdout.flush()

            redundancy = ((self.patch_size[0] ** 3) 
                          // (self.step_size[0] ** 3))
            print(f"\n\n{redundancy}x Averaging...")
            self.imprint_tensor /= redundancy
            self.imprint_tensor = self.imprint_tensor.cpu().numpy()

        def save_prediction(self, dir=None):
            """
            Save prediction volume.

            Parameters
            ----------
            dir : str
                Directory to save volume. If None, it will save volume to same
                path.
            """
            self.step_size = self.step_size[0]
            self.out_dir, self.full_path = Options(self).out_filepath(dir)
            os.makedirs(self.out_dir, exist_ok=True)
            print(f"\nSaving prediction to {self.full_path}...")
            out_nifti = nib.nifti1.Nifti1Image(
                dataobj=self.imprint_tensor, affine=self.affine)
            nib.save(out_nifti, self.full_path)
                            


class RealOctDataset(Dataset):
    """
    Dataset for loading and processing 3D vascular networks.
    """
    def __init__(self,
                 path,
                 subset=None,
                 ):
        """
        Initialize the dataset with the given inputs and subset size.

        Parameters
        ----------
        path : str
            Path to parent directory containing x and y data in dirs
            "x" and "y"
        subset : int, optional
            Number of examples to consider for the dataset, if not all.

        Returns
        -------
        label as shape as torch.Tensor of shape (1, n, n, n)
        """
        #self.subset=slice(subset)
        self.subset=subset
        self.xpaths = np.asarray(sorted(glob(f'{path}/x/*')))[self.subset]
        self.ypaths = np.asarray(sorted(glob(f'{path}/y/*')))[self.subset]

    def __len__(self):
        return len(self.xpaths)

    def __getitem__(self, idx):
        """
        Get a patch and corresponding label map.

        Parameters
        ----------
        idx : int
            Index of the label to retrieve.
        """
        x_tensor = torch.from_numpy(
            nib.load(self.xpaths[idx]).get_fdata()).to('cuda').float()
        y_tensor = torch.from_numpy(
            nib.load(self.ypaths[idx]).get_fdata()).to('cuda').float()
        return x_tensor.unsqueeze(0), y_tensor.unsqueeze(0)


class SubpatchExtractor:
    """
    A class for extracting subpatches from a parent NIfTI volume
    using affine transformations for coordinates and PyTorch for
    computations.

    Attributes
    ----------
    parent_path : str
        The file path to the parent volume.
    parent_nift : nib.Nifti1Image
        The parent volume loaded as a NIfTI image.
    parent_tensor : torch.Tensor
        The parent volume data loaded into a GPU tensor.

    Methods
    -------
    load_parent():
        Loads the parent volume data into a GPU tensor.

    find_subpatch_coordinates_using_affine(subpatch_path: str) -> Tuple[int, int, int]:
        Calculates the voxel coordinates of the subpatch origin within
        the parent volume using affine transformations.

    extract_subpatch(subpatch_origin: Tuple[int, int, int], size: Tuple[int, int, int]) -> nib.Nifti1Image:
        Extracts and returns a subpatch from the parent volume based on
        the specified origin and size.
    """

    def __init__(self, parent_path=None, parent_tensor=None, parent_affine=None):
        """
        Parameters
        ----------
        parent_path : str
            The file path to the parent volume.
        """
        if (parent_tensor is not None) and (parent_affine is not None):
            self.parent_tensor = parent_tensor
            self.parent_affine = parent_affine
        elif (parent_tensor is None) and (parent_affine is None):
            self.parent_path = parent_path
            self.parent_nift, self.parent_tensor = self.load_parent()
            self.parent_affine = self.parent_nift.affine
        else:
            print('Make sure you have an affine!')
            

    def load_parent(self):
        parent_nift = nib.load(self.parent_path)
        parent_tensor = torch.from_numpy(
            parent_nift.get_fdata()
            ).to('cuda')
        return parent_nift, parent_tensor

    def find_subpatch_coordinates_using_affine(self, subpatch_affine) -> Tuple[int, int, int]:
        """
        Calculates the origin coordinates of a subpatch within the parent
        volume using affine transformations from both the parent and subpatch.

        Parameters
        ----------
        subpatch_path : str
            The file path to the subpatch volume.

        Returns
        -------
        Tuple[int, int, int]
            The voxel coordinates (x, y, z) of the subpatch origin within
            the parent volume.
        """
        subpatch_origin_in_world = subpatch_affine @ np.array([0, 0, 0, 1])
        parent_affine_inv = np.linalg.inv(self.parent_affine)
        subpatch_origin_in_parent = parent_affine_inv @ subpatch_origin_in_world
        return tuple(np.round(subpatch_origin_in_parent[:3]).astype(int))
    
    def find_subpatch_coordinates_using_nifti(self, subpatch_path: str) -> Tuple[int, int, int]:
        """
        Calculates the origin coordinates of a subpatch within the parent
        volume using affine transformations from both the parent and subpatch.

        Parameters
        ----------
        subpatch_path : str
            The file path to the subpatch volume.

        Returns
        -------
        Tuple[int, int, int]
            The voxel coordinates (x, y, z) of the subpatch origin within
            the parent volume.
        """
        subpatch_nift = nib.load(subpatch_path)
        parent_affine = self.parent_affine
        subpatch_affine = subpatch_nift.affine
        subpatch_origin_in_world = subpatch_affine @ np.array([0, 0, 0, 1])
        parent_affine_inv = np.linalg.inv(parent_affine)
        subpatch_origin_in_parent = parent_affine_inv @ subpatch_origin_in_world
        return tuple(np.round(subpatch_origin_in_parent[:3]).astype(int))


    def extract_subpatch(self, subpatch_origin: Tuple[int, int, int], 
                         size: Tuple[int, int, int],
                         return_: str='tensor') -> nib.Nifti1Image:
        """
        Extracts a subpatch from the parent volume given the origin and
        size.

        Parameters
        ----------
        subpatch_origin : Tuple[int, int, int]
            The voxel coordinates (x, y, z) marking the starting point
            of the subpatch.
        size : Tuple[int, int, int]
            The size of the subpatch (width, height, depth) in voxels.
        return_ : str {'tensor', 'nifti'}
            What this function will return.

        Returns
        -------
        nib.Nifti1Image
            The extracted subpatch as a NIfTI image.
        """
        start_x, start_y, start_z = subpatch_origin
        end_x = start_x + size[0]
        end_y = start_y + size[1]
        end_z = start_z + size[2]
        new_affine = self.parent_affine.copy()
        for i in range(3):
            new_affine[i, 3] += subpatch_origin[i] * self.parent_affine[i, i]

        subpatch_tensor = self.parent_tensor[
            start_x:end_x, start_y:end_y, start_z:end_z]
        if return_ == 'tensor':
            return subpatch_tensor
        elif return_ == 'nifti':
            nift = nib.Nifti1Image(
                    subpatch_tensor.cpu().numpy(),
                    new_affine)
            return nift


class CAAPrediction:
    """Manage efficient patch-based predictions on NIfTI data.

    Parameters
    ----------
    case : str
        Identifier of the medical case.
    roi : str
        Region of interest within the medical case.
    model_n : int
        Model version number.
    patch_size : int
        Size of the patches to extract for prediction (default 128).
    redundancy : int
        Redundancy factor for overlap of patches (default 2).
    model_exp_name : str
        Base directory name for model outputs (default 'models').
    subpatch_ids : Optional[List[str]]
        Subpatch identifiers for detailed analysis.

    Attributes
    ----------
    imprint_tensor : torch.Tensor
        Tensor to store the predictions.
    complete_patch_coords : List[tuple]
        Coordinates for patches used in prediction.
    num_patches : int
        Number of patches.

    Methods
    -------
    predict()
        Perform all predictions using the loaded model and update
        imprint tensor.
    save_niftis()
        Save outputs and subpatches as NIfTI files.
    """
    def __init__(
            self,
            case: str,
            roi: str,
            model_n: int,
            patch_size: int = 128,
            redundancy: int = 2,
            model_exp_name: str = 'models',
            subpatch_ids: Optional[List[str]] = None
            ) -> None:
        self.case = case
        self.roi = roi
        self.model_n = model_n
        self.patch_size = patch_size
        self.model_exp_name = model_exp_name
        self.redundancy = redundancy-1
        self.step_size = int(patch_size * (1 / (2**self.redundancy)))
        self.subpatch_ids = subpatch_ids
        self.case_roi_basedir = f'/autofs/cluster/octdata2/users/epc28/data/CAA/{case}/{roi}'
        self.model_prediction_dir = f'/autofs/cluster/octdata2/users/epc28/veritas/output/{model_exp_name}/version_{model_n}/predictions'
        self.parent_path = f'{self.case_roi_basedir}/{case}_{roi}.nii'
        self.parent_tensor, self.parent_affine= self._load_parent()
        print('Parent Loaded')
        print('Parent Padded')
        self.imprint_tensor = torch.zeros(self.parent_tensor.shape, device='cuda').to(torch.float32)
        self.complete_patch_coords = self._get_patch_coords()
        self.num_patches = len(self.complete_patch_coords)

    def predict(self):
        sys.stdout.write(f'\rNow Predicting.')
        sys.stdout.flush()
        predictor_ = PatchPredict(model_n=self.model_n, model_exp_name=self.model_exp_name)
        for i in self.complete_patch_coords:
            try:
                self.imprint_tensor[i] += predictor_.predict(self.parent_tensor[i])
            except:
                pass
        del self.parent_tensor, predictor_
        torch.cuda.empty_cache()
        gc.collect()
        self._reformat_imprint_tensor()

    def save_niftis(self):
        # Start by extracting and saving subpatches
        if self.subpatch_ids is not None:
            for id in self.subpatch_ids:
                extractor = SubpatchExtractor(
                    parent_tensor=self.imprint_tensor,
                    parent_affine=self.parent_affine
                    )
                # subpatch refers to the thing that will be referenced to get the good coordinates
                subpatch_inpath = f'{self.case_roi_basedir}/ground_truth/etienne/gt_{id}.nii'
                subpatch_nifti = nib.load(subpatch_inpath)
                subpatch_aff = subpatch_nifti.affine
                coords = extractor.find_subpatch_coordinates_using_affine(subpatch_aff)
                subpatch_nift = extractor.extract_subpatch(coords, [64, 64, 64], return_='nifti')
                subpatch_outpath = f'{self.model_prediction_dir}/{self.case}-{self.roi}_patch-{id}.nii'
                nib.save(subpatch_nift, subpatch_outpath)
                status_message = f"\rSaved subpatch {id}: {subpatch_outpath}"
                sys.stdout.write(status_message)
                sys.stdout.flush()

        # Save raw probability map
        imprint_tensor_outpath = f'{self.model_prediction_dir}/{self.case}-{self.roi}_prediction-r{self.redundancy+1}.nii'
        imprint_nifti = nib.nifti1.Nifti1Image(self.imprint_tensor.to(torch.float32).cpu().numpy(), self.parent_affine)
        nib.save(imprint_nifti, imprint_tensor_outpath)
        status_message = f"\rSaved probability map: {imprint_tensor_outpath}"
        sys.stdout.write(status_message)
        sys.stdout.flush()

        # Save binary map
        binary_imprint_tensor_outpath = f'{self.model_prediction_dir}/{self.case}-{self.roi}_prediction-r{self.redundancy+1}-BINARY.nii'
        self.imprint_tensor[self.imprint_tensor >= 0.5] = 1
        self.imprint_tensor[self.imprint_tensor < 0.5] = 0
        self.imprint_tensor = self.imprint_tensor.to(torch.uint8).cpu().numpy()
        imprint_nifti = nib.nifti1.Nifti1Image(self.imprint_tensor, self.parent_affine)
        nib.save(imprint_nifti, binary_imprint_tensor_outpath)
        status_message = f"\rSaved binarized probability map: {binary_imprint_tensor_outpath}"
        sys.stdout.write(status_message)
        sys.stdout.flush()


    def _reformat_imprint_tensor(self):
        if self._got_padded:
            self.imprint_tensor = self.imprint_tensor[
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size
                ]
        redundancy = (self.patch_size ** 3) // (self.step_size ** 3)
        print(f"\n\n{redundancy}x Averaging...")
        self.imprint_tensor /= redundancy


    def _get_patch_coords(self):
        patch_size = self.patch_size
        step_size = self.step_size
        parent_shape = self.parent_tensor.shape
        complete_patch_coords = []
        x_coords = [slice(x, x + patch_size) for x in range(
            step_size, parent_shape[0] - patch_size, step_size)]
        y_coords = [slice(y, y + patch_size) for y in range(
            step_size, parent_shape[1] - patch_size, step_size)]
        z_coords = [slice(z, z + patch_size) for z in range(
            step_size, parent_shape[2] - patch_size, step_size)]
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    complete_patch_coords.append((x, y, z))
        complete_patch_coords = np.array(complete_patch_coords)
        return complete_patch_coords

    def _load_parent(self):
        nift = nib.load(self.parent_path)
        aff = nift.affine
        tensor = torch.from_numpy(nift.get_fdata())#.to('cuda')
        tensor = self._padit(tensor).to('cuda')
        return tensor, aff

    def _padit(self, tensor):
        pad = [self.patch_size] * 6
        tensor = torch.nn.functional.pad(tensor.unsqueeze(0).unsqueeze(0), pad, mode='reflect')
        tensor = tensor.squeeze()
        self._got_padded = True
        return tensor



import torch
import torch.nn.functional as F

class UNetPredictor:
    def __init__(self, model, model_path, device='cuda'):
        """
        Initialize the predictor with the model checkpoint and device.

        Parameters
        ----------
        model_path : str
            Path to the saved model checkpoint.
        device : torch.device, optional
            Device to run the model on. If None, uses CUDA if available.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to('cuda')
        self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the model from a checkpoint.

        Parameters
        ----------
        model_path : str
            Path to the model checkpoint to load.
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, image_path):
        """
        Make a prediction using an image file path.

        Parameters
        ----------
        image_path : str
            Path to the image to predict.

        Returns
        -------
        torch.Tensor
            The model's prediction as a tensor.
        """
        image = self.preprocess_input(image_path)
        return self.predict_tensor(image)


    def preprocess_input(self, image_path):
        """
        Load and preprocess an input image from a file path.

        Parameters
        ----------
        image_path : str
            Path to the input image.

        Returns
        -------
        torch.Tensor
            Preprocessed image tensor.
        """
        image = nib.load(image_path).get_fdata()
        image = torch.from_numpy(image)  # Add channel and batch dimensions
        image = image.to(torch.float32).to('cuda')
        return image

    def predict_tensor(self, tensor):
        """
        Make a prediction using a pre-loaded tensor.

        Parameters
        ----------
        tensor : torch.Tensor
            A pre-loaded and preprocessed tensor ready for prediction.

        Returns
        -------
        torch.Tensor
            The model's prediction as a tensor.
        """
        with torch.no_grad():
            prediction = self.model(tensor.to('cuda'))
        return F.softmax(prediction, dim=1)
        #return prediction.sigmoid()  # Assuming the output needs to be sigmoid-activated

def main(case, roi, model_n, redundancy, subpatch_ids, model_exp_name='models'):
    t1 = time.time()
    predictor = CAAPrediction(
        case=case,
        roi=roi,
        model_n=model_n,
        redundancy=redundancy,
        subpatch_ids=subpatch_ids,
        model_exp_name=model_exp_name
        )
    print('Starting predictions:', time.time() - t1)
    predictor.predict()
    print('Done predicting:', time.time() - t1)
    predictor.save_niftis()
    print('Done. Total time [s]:', time.time() - t1)


if __name__ == '__main__':
    case_roi = {
        'caa6-frontal' : ['caa6', 'frontal'],
        'caa6-occipital' : ['caa6', 'occipital'],
        'caa26-frontal' : ['caa26', 'frontal'],
        'caa26-occipital' : ['caa26', 'occipital'],
    }
    patches_ = {
        'caa6-frontal' : ['3', '4'],
        'caa6-occipital' : ['4', '5'],
        'caa26-frontal' : ['0', '1'],
        'caa26-occipital' : ['3', '8'],
    }
    versions = [22222]
    print('Starting Tests!!!')
    for version in versions:
        for key in case_roi.keys():
            case, roi = case_roi[key]
            patches = patches_[key]
            main(case, roi, version, redundancy=2, subpatch_ids=patches, model_exp_name='cco_models')
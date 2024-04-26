__all__ = [
    'RealOct',
    'RealOctPatchLoader',
    'RealOctPredict',
    'RealOctDataset'
]

# Standard imports
import os
import sys
import time
import torch
import skimage
import sklearn
from glob import glob
import torchvision
from scipy import ndimage
from sklearn.cluster import KMeans
import random
import numpy as np
import nibabel as nib
from veritas.utils import PathTools, volume_info
from torch.utils.data import Dataset
from cornucopia.cornucopia.intensity import QuantileTransform

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
                 padding_method:str='reflect', # change to "reflect"
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
        pad_it : bool
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
        self.shape = self.tensor.shape
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
            self.volume_dir = f"/{self.input.strip('.nii').strip(self.tensor_name).strip('/')}"
            nifti = nib.load(input)
            # Load tensor on device with dtype. Detach from graph.
            affine = nifti.affine
            tensor = torch.from_numpy(nifti.get_fdata()).to(device=self.device, dtype=torch.float).detach()
        elif isinstance(input, torch.Tensor):
            tensor = input.to(self.device).to(self.dtype).detach()
            nifti = None
        #volume_info(tensor, 'Raw')
        if normalize == True:
            tensor = self.normalize_volume(tensor)
        if pad_it == True:
            tensor = self.pad_volume(tensor)
        if self.binarize == True:
            tensor[tensor <= self.binary_threshold] = 0
            tensor[tensor > self.binary_threshold] = 1
        tensor = tensor.to(self.dtype)
        return tensor, nifti, affine


    def normalize_volume(self, input):
        print('\nNormalizing volume...')
        #input = QuantileTransform(pmin=0.02, pmax=0.98)(input)
        input -= input.min()
        input /= input.max()
        return input
        #volume_info(self.tensor, 'Normalized')


    def pad_volume(self, tensor):
        """
        Pad all dimensions of 3 dimensional tensor and update volume.
        """
        print('\nPadding volume...')
        # Input tensor must be 4 dimensional [1, n, n, n] to do padding
        padding = [self.patch_size] * 6 # Create 6 ele list of patch size
        tensor = torch.nn.functional.pad(
            input=tensor.unsqueeze(0),
            pad=padding,
            mode=self.padding_method
        )[0]
        #volume_info(tensor, 'Padded')
        return tensor
    
    
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
            preprocessed_vol = torchvision.transforms.functional.gaussian_blur(preprocessed_vol, 15, sigma=1e2)
            # Sampling volume and computing quantile
            selection_tensor = torch.randint(0, 100, preprocessed_vol.shape).to('cuda')
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
            complete_pred = kmeans.predict(preprocessed_vol.cpu().reshape(-1, 1))

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
            eroded_blurred = torchvision.transforms.functional.gaussian_blur(eroded, 9, sigma=1e2)
            final_pred = eroded_blurred.clip(0)

            del eroded
            del eroded_blurred
            torch.cuda.empty_cache()

            return final_pred
        

        def make_matter_masks(tissue_mask, n_clusters=3):
            print('Making WM & GM masks...')
            tissue_masked = tissue_mask.cuda() * self.tensor.cuda()
            tissue = torchvision.transforms.functional.gaussian_blur(tissue_masked, 15, sigma=1e2)

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
                nifti = nib.nifti1.Nifti1Image(masks[i].cpu().numpy().astype(np.uint8), affine=self.affine)
                nib.save(nifti, f'{self.volume_dir}/kmeans-{mask_types[i]}.nii')
            return masks[mask_types.index(mask_type)]
        


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
        patch = self.tensor[x_slice, y_slice, z_slice].detach()
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
                    patch, coords = self[random_patch_indicies[i]]#(threshold_tensor)
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

    def __init__(self, trainee=None, normalize_patches=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = dict(dtype=self.dtype, device=self.device)
        self.trainee = trainee
        self.imprint_tensor = torch.zeros(self.tensor.shape, **self.backend)
        self.normalize_patches=normalize_patches
        # The edges of the kernel will be, at most, multiplied by min_scale
        # No weight = 3/2, zeros = 0
        min_scale = 1/2 #1/2
        half_filter_1d = torch.linspace(min_scale, torch.pi/2, self.patch_size//2, **self.backend).sin()
        filter_1d = torch.concat([half_filter_1d, half_filter_1d.flip(0)])
        self.patch_weight = filter_1d[:, None, None] * filter_1d[None, :, None] * filter_1d[None, None, :]
        self.patch_weight = self.patch_weight.to('cuda')

    def __getitem__(self, idx:int):
        """
        Predict on a single patch.

        Parameters
        ----------
        idx : int
            Patch ID number to predict on. Updates self.imprint_tensor.
        """
        patch, coords = super().__getitem__(idx)
        ## Needs to go on cuda for prediction. Useful when predicting on large
        ## volumes that are on CPU.
        if self.device != 'cuda':
            patch = patch.to('cuda')
        patch = patch.unsqueeze(0).unsqueeze(0)
        if self.normalize_patches == True:
            #patch -= patch.min()
            #patch /= patch.max()
            #try:
            patch = QuantileTransform()(patch.float())
            #except:
            #    pass
        prediction = self.trainee(patch)
        prediction = torch.sigmoid(prediction).squeeze()
        weighted_prediction = (prediction * self.patch_weight).to(**self.backend)
        self.imprint_tensor[coords[0], coords[1], coords[2]] += weighted_prediction

    def predict_on_all(self):
        if self.tensor.dtype != torch.float32:
            self.tensor = self.tensor.to(torch.float32)
            print('Input tensor needs to be float32!!')
        n_patches = len(self)
        t0 = time.time()
        print('Starting predictions!!')
        with torch.no_grad():            
            for i in range(n_patches):
                self[i]
                if (i+1) % 10 == 0:
                    total_elapsed_time = time.time() - t0
                    average_time_per_pred = round(total_elapsed_time / (i+1), 3)
                    sys.stdout.write(f"\rPrediction {i + 1}/{n_patches} | {average_time_per_pred} sec/pred | {round(average_time_per_pred * n_patches / 60, 2)} min total pred time")
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
            Directory to save volume. If None, it will save volume to same path.
        """
        self.out_dir, self.full_path = Options(self).out_filepath(dir)
        os.makedirs(self.out_dir, exist_ok=True)
        print(f"\nSaving prediction to {self.full_path}...")
        out_nifti = nib.nifti1.Nifti1Image(dataobj=self.imprint_tensor, affine=self.affine)
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
            Path to parent directory containing x and y data in dirs "x" and "y"
        subset : int, optional
            Number of examples to consider for the dataset, if not all.

        Returns
        -------
        label as shape as torch.Tensor of shape (1, n, n, n)
        """
        self.subset=slice(subset)
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
        x_tensor = torch.from_numpy(nib.load(self.xpaths[idx]).get_fdata()).to('cuda').float()
        y_tensor = torch.from_numpy(nib.load(self.ypaths[idx]).get_fdata()).to('cuda').float()
        return x_tensor.unsqueeze(0), y_tensor.unsqueeze(0)
__all__ = [
    'VesselSynth',
    'OctVolSynth',
    'OctVolSynthDataset',
    'RealAug'
]

# Standard imports
import os
import json
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import random
import glob
from typing import Tuple, Union, List
from torch import nn
from torch.cuda.amp import autocast  # Utilizing automatic mixed precision
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur
import torchvision.transforms as transforms


# Custom Imports
from veritas.utils import PathTools, JsonTools, volume_info
from vesselsynth.vesselsynth.utils import backend
from vesselsynth.vesselsynth.io import default_affine
from vesselsynth.vesselsynth.synth import SynthVesselOCT
from cornucopia.cornucopia.labels import RandomSmoothLabelMap, BernoulliDiskTransform
from cornucopia.cornucopia.noise import RandomGammaNoiseTransform
from cornucopia.cornucopia.geometric import ElasticTransform
from cornucopia.cornucopia import (
    RandomSlicewiseMulFieldTransform, RandomGammaTransform,
    RandomMulFieldTransform, RandomGaussianMixtureTransform
)
from cornucopia.cornucopia.noise import RandomChiNoiseTransform, RandomGaussianNoiseTransform, RandomGammaNoiseTransform
from cornucopia.cornucopia.intensity import QuantileTransform
from cornucopia.cornucopia import fov, geometric, intensity, labels, noise
from cornucopia.cornucopia.random import Uniform, Fixed, RandInt, Normal


class VesselSynth(object):
    """
    Synthesize 3D vascular network and save as nifti.
    """
    def __init__(self,
                 device:str='cuda',
                 json_param_path:str='scripts/1_vesselsynth/vesselsynth_params.json',
                 experiment_dir='synthetic_data',
                 experiment_number=1
                 ):
        """
        Initialize the VesselSynth class to synthesize 3D vascular networks.

        Parameters
        ----------
        device : str, optional
            Which device to run computations on, default is 'cuda'.
        json_param_path : str, optional
            Path to JSON file containing parameters.
        experiment_dir : str, optional
            Directory for output of synthetic experiments.
        experiment_number : int, optional
            Identifier for the experiment.
        """
        # All JIT things need to be handled here. Do not put them outside this class.
        os.environ['PYTORCH_JIT_USE_NNC_NOT_NVFUSER'] = '1'
        backend.jitfields = True
        self.device = device
        self.json_params = json.load(open(json_param_path))   # This is the json file that should be one directory above this one. Defines all variables
        self.shape = self.json_params['shape']                           
        self.n_volumes = self.json_params['n_volumes']
        self.begin_at_volume_n = self.json_params['begin_at_volume_n']
        self.experiment_path = f"output/{experiment_dir}/exp{experiment_number:04d}"
        PathTools(self.experiment_path).makeDir()
        self.header = nib.Nifti1Header()
        self.prepOutput(f'{self.experiment_path}/#_vesselsynth_params.json')
        self.backend()
        self.outputShape()


    def synth(self):
        """
        Synthesize a vascular network.
        """
        file = open(f'{self.experiment_path}/#_notes.txt', 'x')
        file.close()

        for n in range(self.begin_at_volume_n, self.begin_at_volume_n + self.n_volumes):
            print(f"Making volume {n:04d}")
            synth_names = ['prob', 'label', "level", "nb_levels",
                         "branch", "skeleton"]
            # Synthesize volumes
            synth_vols = SynthVesselOCT(shape=self.shape, device=self.device)()
            # Save each volume individually
            for i in range(len(synth_names)):
                self.saveVolume(n, synth_names[i], synth_vols[i])   


    def backend(self):
        """
        Check and set the computation device.
        """
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'


    def outputShape(self):
        """
        Ensure shape is a list of three dimensions.
        """
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]


    def saveVolume(self, volume_n:int, volume_name:str, volume:torch.Tensor):
        """
        Save the synthesized volume as a NIfTI file.

        Parameters
        ----------
        volume_n : int
            Identifier for the volume.
        volume_name : str
            Name of the synthesized volume type.
        volume : torch.Tensor
            Synthesized volume tensor.
        """
        affine = default_affine(volume.shape[-3:])
        nib.save(nib.Nifti1Image(
            volume.squeeze().cpu().numpy(), affine, self.header),
            f'{self.experiment_path}/{volume_n:04d}_vessels_{volume_name}.nii.gz')
        
        
    def prepOutput(self, abspath:str):
        """
        Clear files in output dir and log synth parameters to json file.
        
        Parameters
        ---------
        abspath: str
            JSON abspath to log parameters
        """
        json_object = json.dumps(self.json_params, indent=4)
        with open(abspath, 'w') as file:
            file.write(json_object)


class OctVolSynthDataset(Dataset):
    """
    Dataset class for synthesizing OCT intensity volumes from vascular networks.
    """
    def __init__(self,
                 exp_path:str=None,
                 label_type:str='label',
                 device:str="cuda",
                 synth_params='complex'
                 ):
        """
        Initialize the dataset for synthesizing OCT volumes.

        Parameters
        ----------
        exp_path : str
            Path to the experiment directory.
        label_type : str
            Type of label to use for synthesis.
        device : str
            Computation device to use.
        synth_params : str
            Parameters for synthesis complexity.
        """
        self.device = device
        self.backend = dict(dtype=torch.float32, device=device)
        self.label_type = label_type
        self.exp_path = exp_path
        self.synth_params=synth_params
        self.label_paths = sorted(glob.glob(f"{exp_path}/*label*"))
        self.y_paths = sorted(glob.glob(f"{self.exp_path}/*{self.label_type}*"))
        self.sample_fig_dir = f"{exp_path}/sample_vols/figures"
        self.sample_nifti_dir = f"{exp_path}/sample_vols/niftis"
        PathTools(self.sample_nifti_dir).makeDir()
        PathTools(self.sample_fig_dir).makeDir()


    def __len__(self) -> int:
        return len(self.label_paths)


    def __getitem__(self, idx:int, save_nifti=False, make_fig=False,
                    save_fig=False) -> tuple:
        """
        Retrieve a synthesized OCT volume and corresponding probability map.

        Parameters
        ----------
        idx : int
            Index of the sample.
        save_nifti : bool, optional
            Whether to save the synthesized volume as a NIfTI file.
        make_fig : bool, optional
            Whether to generate a figure of the synthesized volume.
        save_fig : bool, optional
            Whether to save the generated figure.
        """
        # Loading nifti and affine
        label_nifti = nib.load(self.label_paths[idx])
        label_affine = label_nifti.affine
        # Loading tensor into torch
        self.label_tensor_backend = dict(device='cuda', dtype=torch.int32)
        label_tensor = torch.from_numpy(label_nifti.get_fdata()).to(**self.label_tensor_backend)
        label_tensor = torch.clip(label_tensor, 0, 32767)[None]

        # Synthesizing volume
        im, prob = OctVolSynth(synth_params=self.synth_params)(label_tensor)
        # Converting image and prob map to numpy. Reshaping
        im = im.detach().cpu().numpy().squeeze()

        if self.label_type == 'label':
            prob = prob.to(torch.int32).cpu().numpy().squeeze()

        if save_nifti == True:
            volume_name = f"volume-{idx:04d}"
            out_path_volume = f'{self.sample_nifti_dir}/{volume_name}.nii'
            out_path_prob = f'{self.sample_nifti_dir}/{volume_name}_MASK.nii'
            print(f"Saving Nifti to: {out_path_volume}")
            nib.save(nib.Nifti1Image(im, affine=label_affine), out_path_volume)
            nib.save(nib.Nifti1Image(prob, affine=label_affine), out_path_prob)
        if save_fig == True:
            make_fig = True
        if make_fig == True:
            self.make_fig(im, prob)
        if save_fig == True:
            plt.savefig(f"{self.sample_fig_dir}/{volume_name}.png")
        return im, prob
    

    def make_fig(self, im:np.ndarray, prob:np.ndarray) -> None:
        """
        Make 2D figure (GT, prediction, gt-pred superimposed).
        Print to console.

        Parameters
        ----------
        im : arr[float]
            Volume of x data
        prob: arr[bool] 
            Volume of y data
        """
        plt.figure()
        f, axarr = plt.subplots(1, 3, figsize=(15, 15), constrained_layout=True)
        axarr = axarr.flatten()
        frame = np.random.randint(0, im.shape[0])
        axarr[0].imshow(im[frame], cmap='gray')
        axarr[1].imshow(prob[frame], cmap='gray')
        axarr[2].imshow(im[frame], cmap='gray')
        axarr[2].contour(prob[frame], cmap='magma', alpha=1)



class VascularNetworkDataset(Dataset):
    """
    Dataset for loading and processing 3D vascular networks.
    """
    def __init__(self,
                 inputs,
                 subset=-1,
                 ):
        """
        Initialize the dataset with the given inputs and subset size.

        Parameters
        ----------
        inputs : list or str
            List of file paths or a directory/pattern representing the input labels.
        subset : int, optional
            Number of examples to consider for the dataset, if not all.

        Returns
        -------
        label as shape as torch.Tensor of shape (1, n, n, n)
        """
        self.subset=slice(subset)
        self.inputs=np.asarray(inputs[self.subset]) # uses less RAM in multithreads

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Get a single label from the dataset.

        Parameters
        ----------
        idx : int
            Index of the label to retrieve.
        """
        label = torch.from_numpy(nib.load(self.inputs[idx]).get_fdata()).to('cuda')
        label = torch.clip(label, 0, 32767).to(torch.int16)[None]
        return label



class OctVolSynth(nn.Module):
    """
    Module to synthesize OCT-like volumes from vascular networks.
    """
    def __init__(self,
                 synth_params:str='complex',
                 dtype=torch.float32,
                 device:str='cuda',
                 ):
        super().__init__()
        """
        Initialize the module for OCT volume synthesis with parameters optimized for GPU execution.

        Parameters
        ----------
        synth_params : str, optional
            Parameter set defining the complexity of the synthesis.
        dtype : torch.dtype, optional
            Data type for internal computations, adjusted for GPU compatibility.
        device : str, optional
            Computation device to use, default is 'cuda'.
        """
        self.synth_params=synth_params
        self.backend = dict(device=device, dtype=dtype)
        self.dtype = dtype
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.setup_parameters()
        
    def setup_parameters(self):
        """
        Setup parameters and data-dependent tensors directly on the GPU to minimize transfers.
        """
        # Parameters should be read and initialized here, for now using placeholders
        self.json_path = f'/autofs/cluster/octdata2/users/epc28/veritas/scripts/2_imagesynth/imagesynth_params-{self.synth_params}.json'
        self.json_dict = JsonTools(self.json_path).read()
        self.speckle_a = float(self.json_dict['speckle'][0])
        self.speckle_b = float(self.json_dict['speckle'][1])
        self.gamma_a = float(self.json_dict['gamma'][0])
        self.gamma_b = float(self.json_dict['gamma'][1])
        self.thickness_ = int(self.json_dict['z_decay'][0])
        self.nb_classes_ = int(self.json_dict['parenchyma']['nb_classes'])
        self.shape_ = int(self.json_dict['parenchyma']['shape'])
        self.i_max = float(self.json_dict['imax'])
        self.i_min = float(self.json_dict['imin'])
        self.zband_ = True
        self.tex_ = True
        self.balls_ = True
        self.dc_off_ = False

    @autocast()  # Enables mixed precision for faster computation
    def forward(self, vessel_labels_tensor:torch.Tensor) -> tuple:
        """
        Forward pass for generating the OCT-like volumes.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor of vessels with unique ID integer labels.
        """
        # synthesize the main parenchyma (background tissue)
        # Get sorted list of all vessel labels for later
        vessel_labels_tensor = vessel_labels_tensor.to(self.device)
        vessel_labels = torch.unique(
            vessel_labels_tensor).sort().values.nonzero().squeeze()
        n_unique_ids=1

        # Randomly make a negative control
        if random.randint(1, 10) == 7:
            vessel_labels_tensor[vessel_labels_tensor > 0] = 0
            n_unique_ids = 0
        else:
            # Hide some vessels randomly
            n_unique_ids = len(vessel_labels)
            number_vessels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
                )
            vessel_ids_to_hide = vessel_labels[
                torch.randperm(n_unique_ids)[:number_vessels_to_hide]]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels_tensor[vessel_labels_tensor == vessel_id] = 0

        # Apply DC offset
        parenchyma = self.parenchyma_(vessel_labels_tensor)
        if self.dc_off_ == True:
            dc_offset = random.uniform(0, 0.25)
            parenchyma += dc_offset

        final_volume = parenchyma.clone()
        # Determine if there are any vessels to deal with
        if n_unique_ids > 0:
            # synthesize vessels (grouped by intensity)
            vessels = self.vessels_(vessel_labels_tensor) 
            if self.synth_params == 'complex':
                if self.tex_ == True:
                    # texturize those vessels!!!
                    vessel_texture = self.vessel_texture_(vessel_labels_tensor)
                    vessels[vessel_labels_tensor > 0] *= vessel_texture[
                        vessel_labels_tensor > 0
                        ]
            final_volume[vessel_labels_tensor > 0] *= vessels[
                    vessel_labels_tensor > 0
                    ]
        # Normalizing
        final_volume = QuantileTransform()(final_volume)
        # final output needs to be in float32 or else torch throws mismatch
        # error between this and weights tensor.
        final_volume = final_volume.to(torch.float32)
        return final_volume, vessel_labels_tensor#.clip_(0, 1)
        

    def parenchyma_(self, vessel_labels_tensor:torch.Tensor):
        """
        Generate parenchyma based on vessel labels using GPU optimized
        operations.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor of vessels with unique ID integer labels.
        """
        # Create the label map of parenchyma but convert to float32 for further
        #computations. Add 1 so that we can work with every single pixel 
        # (no zeros)
        parenchyma = RandomSmoothLabelMap(
            nb_classes=random.randint(2, self.nb_classes_),
            shape=random.randint(2, self.shape_),
            )(vessel_labels_tensor) + 1
        # Randomly assigning intensities to parenchyma
        parenchyma = parenchyma.to(torch.float32)
        for i in torch.unique(parenchyma):
            parenchyma.masked_fill_(parenchyma==i, random.gauss(i,0.2))
        parenchyma /= parenchyma.max()
        # Applying speckle noise model
        parenchyma = RandomGammaNoiseTransform(
            sigma=Uniform(self.speckle_a, self.speckle_b)
            )(parenchyma)
        if self.synth_params == 'complex':
            if self.balls_ == True:
                if random.randint(0, 2) == 1:
                    balls = BernoulliDiskTransform(
                        prob=1e-2,
                        radius=random.randint(1, 4),
                        value=random.uniform(0, 2)
                        )(parenchyma)[0]
                    if random.randint(0, 2) == 1:
                        balls = ElasticTransform(shape=5)(balls).detach()
                    parenchyma *= balls
                    # Applying z-stitch artifact
            if self.zband_ == True:
                parenchyma = RandomSlicewiseMulFieldTransform(
                    thickness=self.thickness_
                    )(parenchyma)
            elif self.zband_ == False:
                parenchyma = RandomMulFieldTransform(5)(parenchyma)
        elif self.synth_params == 'simple':
            # Give bias field in lieu of slicewise transform
            parenchyma = RandomMulFieldTransform(5)(parenchyma)
        parenchyma = RandomGammaTransform((
            self.gamma_a, self.gamma_b))(parenchyma)
        parenchyma = QuantileTransform()(parenchyma)
        #volume_info(parenchyma)
        return parenchyma
    

    def vessels_(self, vessel_labels_tensor:torch.Tensor, n_groups:int=10):
        """
        Parameters
        ----------
        vessel_labels_tensor : tensor[int]
            Tensor of vessels with unique ID integer labels
        n_groups : int
            Number of vessel groups differentiated by intensity
        min_i : float
            Minimum intensity of vessels compared to background
        max_i : float
            Maximum intensity of vessels compared to background
        """
        # Generate an empty tensor that we will fill with vessels and their
        # scaling factors to imprint or "stamp" onto parenchymal volume
        scaling_tensor = torch.zeros(
            vessel_labels_tensor.shape,
            dtype=self.dtype,
            device=vessel_labels_tensor.device)
        vessel_texture_fix_factor = random.uniform(0.5, 1)
        # Iterate through each vessel group based on their unique intensity
        vessel_labels_left = torch.unique(vessel_labels_tensor)
        for int_n in vessel_labels_left:
            intensity = Uniform(self.i_min, self.i_max)()
            scaling_tensor.masked_fill_(vessel_labels_tensor == int_n,
                                        intensity * vessel_texture_fix_factor)
        return scaling_tensor
    

    def vessel_texture_(self, vessel_labels_tensor:torch.Tensor, nb_classes:int=4,
                shape:int=5):
        """
        Parameters
        ----------
        vessel_labels_tensor : tensor[int]
            Tensor of vessels with unique ID integer labels
        nb_classes : int
            Number of unique parenchymal "blobs"
        shape : int
            Number of spline control points
        """
        
        # Create the label map of parenchyma but convert to float32 for further computations
        # Add 1 so that we can work with every single pixel (no zeros)
        #nb_classes = RandInt(2, self.nb_classes_)()
        vessel_texture = RandomSmoothLabelMap(
            nb_classes=Fixed(2),
            shape=self.shape_,
            )(vessel_labels_tensor) + 1
        # Applying gaussian mixture
        vessel_texture = RandomGaussianMixtureTransform(
            mu=random.uniform(0.7, 1),
            sigma=0.8,
            dtype=self.dtype
            )(vessel_texture)
        # Normalizing and clamping min
        vessel_texture -= vessel_texture.min()
        vessel_texture /= (vessel_texture.max()*2)
        vessel_texture += 0.5
        vessel_texture.clamp_min_(0)
        return vessel_texture




###################################

class RealAug(nn.Module):
    """
    Module to synthesize OCT-like volumes from vascular networks.
    """
    def __init__(self):
        super().__init__()
        """
        Initialize the module for OCT volume synthesis with parameters optimized for GPU execution.

        Parameters
        ----------
        synth_params : str, optional
            Parameter set defining the complexity of the synthesis.
        dtype : torch.dtype, optional
            Data type for internal computations, adjusted for GPU compatibility.
        device : str, optional
            Computation device to use, default is 'cuda'.
        """

    @autocast()  # Enables mixed precision for faster computation on supported GPUs
    def forward(self, batch) -> tuple:
        """
        Forward pass for generating the OCT-like volumes.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor of vessels with unique ID integer labels.
        """
        x, y = batch
        x, y = fov.RandomFlipTransform()(x, y)
        x = QuantileTransform(
            vmin=Uniform(0, 0.5)(),
            vmax=Uniform(0.5, 1)()
            )(x)
        return x, y
    

class YibeiSynth(nn.Module):
    """
    Module to synthesize data for Yibei.
    """
    def __init__(self):
        """
        Initialize module.
        """
        super().__init__()
        self.setup_parameters()

    def setup_parameters(self):
        self.banding_ = True
        self.balls_ = True
        self.i_min = 0.1
        self.i_max = 2

    @autocast() # Enable mixed precision
    def forward(self, vessel_labels: torch.Tensor) -> tuple:
        """
        Synthesize X and y data from vessel label maps.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels with unique ID's.
        """
        self.shape = list(vessel_labels.shape)
        vessel_labels = vessel_labels.to('cuda')
        vessel_intensities, vessel_mask = self.sample_vessels(vessel_labels)
        parenchyma = self.sample_parenchyma()
        parenchyma[vessel_mask > 0] *= vessel_intensities[vessel_mask > 0]
        return parenchyma, vessel_mask

    def sample_vessels(self, vessel_labels):
        vessel_labels = self.flip(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
            vessel_masks = vessel_labels.clip_(0, 1).to(torch.int16)
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int16, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int16, device='cuda')
        return vessel_intensities, vessel_masks


    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    

    def sample_label_intensity(self, labels, unique_ids):
        scaling_tensor = torch.ones(self.shape).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if random.randint(0, 1) == 1:
                intensity = Uniform(0, 0.7)()
            else:
                intensity = Uniform(1.3, 2)()
            scaling_tensor.masked_fill_(labels == id, intensity)
        return scaling_tensor
    def flip(self, vessel_labels):
        return fov.RandomFlipTransform()(vessel_labels)

    def sample_parenchyma(self):
        p0 = torch.ones(self.shape, device='cuda')
        if RandInt(0, 1)() == 1:
            p0 = labels.RandomSmoothLabelMap(4)(p0) + 1
            p0 = p0.to('cuda').to(torch.float32)
        p1 = self.get_base_noise()(p0)
        p2 = self.slicewise()(p1)
        return intensity.QuantileTransform()(p2)

    def get_base_noise(self):
        p1s = [
            noise.RandomGaussianNoiseTransform(0.5),
            noise.RandomChiNoiseTransform(0.5),
            noise.RandomGammaNoiseTransform(0.5),
        ]
        #0, p1s.shape, 1 ele
        random_idx = random.randint(0, 2)
        return p1s[random_idx].to('cuda')
    
    def slicewise(self):
        if random.randint(0, 1) == 1:
            transform = intensity.RandomSlicewiseMulFieldTransform()
        else:
            transform = intensity.RandomMulFieldTransform()
        return transform.to('cuda')
    
    def get_p2(self):
        pass


class YibeiSynthSemantic(nn.Module):
    """
    Module to synthesize data for Yibei.
    """
    def __init__(self):
        """
        Initialize module.
        """
        super().__init__()
        self.setup_parameters()

    def setup_parameters(self):
        self.banding_ = True
        self.balls_ = True
        self.i_min = 0.1
        self.i_max = 2

    @autocast() # Enable mixed precision
    def forward(self, vessel_labels: torch.Tensor) -> tuple:
        """
        Synthesize X and y data from vessel label maps.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels with unique ID's.
        """
        self.shape = list(vessel_labels.shape)
        vessel_labels = vessel_labels.to('cuda')
        vessel_intensities, onehot, vessel_mask = self.sample_vessels(vessel_labels)
        parenchyma = self.sample_parenchyma()
        parenchyma[vessel_mask > 0] *= vessel_intensities[vessel_mask > 0]
        #onehot[0, ...].masked_fill_(parenchyma == 0, 1)
        #onehot[1, ...].masked_fill_(parenchyma == 0, 0)
        #onehot[2, ...].masked_fill_(parenchyma == 0, 0)
        return parenchyma, onehot 
    
    def normalize_(self, tensor, verbose=False):
        style = RandInt(0, 3)()
        if RandInt(0, 1)() == 1:
            tensor += Uniform(0 ,0.5)()
        if style == 0:
            tensor = intensity.QuantileTransform(
                vmin=Uniform(0, 0.5)(),
                pmin=Uniform(0, 0.5)(),
                vmax=Uniform(0.5, 1)(),
                pmax=Uniform(0.5, 1)()
                )(tensor)
        elif style == 1:
            tensor -= tensor.min()
            tensor /= tensor.max()
        elif style == 2:
            tensor = transforms.Normalize(tensor.std(), tensor.mean())(tensor)
        else:
            pass
        if verbose == True:
            print(f'Using normalization style {style}')
        return tensor


    def sample_vessels(self, vessel_labels):
        vessel_labels = self.flip(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int16, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int16, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks


    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    

    def sample_label_intensity(self, labels, unique_ids):
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if random.randint(0, 1) == 1:
                intensity = Uniform(0, 0.8)()
                onehot[1, ...].masked_fill_(labels == id, 1)
            else:
                intensity = Uniform(1.2, 2)()
                onehot[2, ...].masked_fill_(labels == id, 1)
            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)
        return scaling_tensor, onehot, mask
    

    def flip(self, vessel_labels):
        return fov.RandomFlipTransform()(vessel_labels)


    def sample_parenchyma(self):
        p0 = torch.ones(self.shape, device='cuda')
        if RandInt(0, 1)() == 1:
            p0 = labels.RandomSmoothLabelMap(3)(p0) + 1
            p0 = p0.to(torch.float32)
        p1 = self.get_base_noise()(p0)
        p2 = self.slicewise()(p1)
        p3 = intensity.RandomGammaTransform()(p2)
        pout = p3
        pout = self.normalize_(pout, verbose=False)
        return pout


    def get_base_noise(self):
        p1s = [
            noise.RandomGaussianNoiseTransform(0.2),
            noise.RandomChiNoiseTransform(0.2),
            noise.RandomGammaNoiseTransform(0.2),
        ]
        #0, p1s.shape, 1 ele
        random_idx = RandInt(0, 2)()
        return p1s[random_idx].to('cuda')
    
    def slicewise(self):
        if random.randint(0, 1) == 1:
            transform = intensity.RandomSlicewiseMulFieldTransform()
        else:
            transform = intensity.RandomMulFieldTransform()
        return transform.to('cuda')
    
    def get_p2(self):
        pass


###############################################################################

class ParenchymaSynth(nn.Module):

    def __init__(self, shape=[1, 128, 128, 128], light_vessels=False, blur=False, banding=False, balls=False):
        super(ParenchymaSynth, self).__init__()
        self.shape = shape
        self.light_vessels = light_vessels
        self.blur = blur
        self.balls = balls
        self.banding = banding
        self.setup_parameters()
        self.parenchyma = torch.ones(size=self.shape).cuda()

    def setup_parameters(self):
        self.intensity_a = 3
        self.intensity_b = 20

    @autocast()
    def forward(self):
        self.add_base_intensities()
        self.add_base_noise()
        if self.banding == True:
            self.slicewise()
        if self.balls == True:
            self.add_balls()
        return self.parenchyma.squeeze()

    def add_base_intensities(self):
        self.parenchyma = labels.RandomSmoothLabelMap(
            RandInt(3, 10)(), RandInt(3, 10)()
            )(self.parenchyma).to('cuda') + 1
        
        self.parenchyma = self.parenchyma.to(torch.float32)
        if self.blur == True:
            if Uniform(0, 1)() <= 0.8:
                kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
                idx = RandInt(0, len(kernel_sizes)-1)() 
                self.parenchyma = transforms.GaussianBlur(
                    kernel_sizes[idx], Uniform(0.1, 20)()
                    )(self.parenchyma)

    def add_base_noise(self):
        noise_models = [
            #noise.RandomGaussianNoiseTransform(Uniform(0, 0.5)),
            #noise.RandomChiNoiseTransform(Uniform(0, 0.5)),
            noise.RandomGammaNoiseTransform(0.1, 1)
        ]
        idx = RandInt(0, len(noise_models)-1)()
        noise_model = noise_models[idx]
        self.parenchyma = noise_model(
            self.parenchyma
            ).to('cuda').to(torch.float32)

    def slicewise(self):
        if random.randint(0, 1) == 1:
            self.parenchyma = intensity.RandomSlicewiseMulFieldTransform()(self.parenchyma)
        else:
            self.parenchyma = intensity.RandomMulFieldTransform()(self.parenchyma)

    def add_balls(self):
        dupe = torch.ones(self.shape, device=self.parenchyma.device)
        if self.light_vessels:
            intensity_sampler = Uniform(1, 10)()
        else:
            intensity_sampler = Uniform(0.1, 1.4)()

        # MAKING SURE BALLS DON'T LOOK LIKE VESSELS
        radius = RandInt(1, 20)()
        if radius >= 4:
            prob = Uniform(0.005, 0.01)()
        elif radius < 4:
            prob = Uniform(0.01, 0.05)()
        
        balls = labels.SmoothBernoulliDiskTransform(
            prob=prob,
            radius=RandInt(1, 20)(),
            shape=RandInt(5, 40)(),
            value=intensity_sampler
            )(dupe)
        balls = transforms.GaussianBlur(3, Uniform(0.1, 5)())(balls)
        self.parenchyma *= balls


class InsaneSynthSemantic(nn.Module):
    """
    Module to synthesize data for Yibei.
    """
    def __init__(self, light_vessels=True, blur=True, banding=True, balls=True, verbose=True):
        """
        Initialize module.
        """
        super().__init__()
        self.light_vessels = light_vessels
        self.blur = blur
        self.banding = banding
        self.balls = balls
        self.verbose = verbose
        #self.setup_parameters(light_vessels, blur, banding, balls, verbose)

    @autocast() # Enable mixed precision
    def forward(self, vessel_labels: torch.Tensor) -> tuple:
        """
        Synthesize X and y data from vessel label maps.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels with unique ID's.
        """
        
        #if RandInt(0, 1)() == 1:
        #    self.light_vessels = True
        #else:
        #    self.light_vessels = False
        with torch.no_grad():
            self.shape = list(vessel_labels.shape)
            vessel_labels = vessel_labels.to('cuda')
            vessel_intensities, onehot, vessel_mask = self.sample_vessels(vessel_labels)
            parenchyma = ParenchymaSynth(
                light_vessels=self.light_vessels,
                blur=self.blur,
                banding=self.banding,
                balls=self.balls
                )()
            parenchyma[vessel_mask > 0] *= vessel_intensities[vessel_mask > 0]
            #parenchyma = self.final_transforms_(parenchyma)
            parenchyma, onehot = self.sanitycheck(parenchyma, onehot)
            return parenchyma, onehot
    
    def sanitycheck(self, parenchyma, onehot):
        if torch.isnan(parenchyma).sum():
            parenchyma = torch.zeros(parenchyma.shape, dtype=torch.float32, device='cuda')
            onehot = torch.zeros(onehot.shape, dtype=torch.float32, device='cuda')
        else:
            pass
        if torch.isinf(parenchyma).sum():
            parenchyma = torch.zeros(parenchyma.shape, dtype=torch.float32, device='cuda')
            onehot = torch.zeros(onehot.shape, dtype=torch.float32, device='cuda')
        else:
            pass
        return parenchyma, onehot
    

    def final_transforms_(self, tensor):
        tensor = QuantileTransform(vmin=0.02, vmax=0.98)(tensor)
        #a = torch.quantile(tensor, 0.02)
        #print(a)
        #tensor -= tensor.min()
        #tensor /= tensor.max()
        return tensor

    def sample_vessels(self, vessel_labels):
        vessel_labels = self.flip(vessel_labels)
        vessel_labels = self.flip(vessel_labels)
        vessel_labels = self.flip(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int16, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int16, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks


    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    
    def sample_label_intensity(self, labels, unique_ids):
        if self.light_vessels == True:
            intensity_ = Uniform(2, 10)
        elif self.light_vessels == False: 
            intensity_a = Uniform(0, 0.5)
            intensity_b = Uniform(1.5, 2)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if self.light_vessels == True:
                intensity = intensity_()
                onehot[2, ...].masked_fill_(labels == id, 1)
            elif self.light_vessels == False:
                if random.randint(0, 1) == 1:
                    # Make darker vessels
                    intensity = intensity_a()
                    onehot[1, ...].masked_fill_(labels == id, 1)
                else:
                    # Make brighter vessels
                    intensity = intensity_b()
                    onehot[2, ...].masked_fill_(labels == id, 1)

            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)

        if self.blur == True:
            if RandInt(0, 1)() == 1:
                if self.light_vessels == True:
                    scaling_tensor = transforms.GaussianBlur(3, Uniform(0.5, 2)())(scaling_tensor)
                if self.light_vessels == False:
                    scaling_tensor = transforms.GaussianBlur(3, 0.5)(scaling_tensor)
        return scaling_tensor, onehot, mask
    
    def flip(self, vessel_labels):
        return fov.RandomFlipTransform()(vessel_labels)

class YibeiVessels(nn.Module):
    
    def __init__(self, allow_light_vessels=False, blur=False, banding=False, balls=False):
        super(YibeiVessels, self).__init__()
        self.allow_light_vessels = allow_light_vessels
        self.blur = blur
        self.balls = balls
        self.banding = banding

    @autocast()
    def forward(self, vessel_labels):
        self.vessel_labels = vessel_labels.to('cuda')
        self.shape = list(vessel_labels.shape)
        self.parenchyma = torch.ones(size=self.shape).to('cuda')
        if self.allow_light_vessels == True:
            if RandInt(0, 1)() == 0:
                self.light_vessels = True
            else:
                self.light_vessels = False
        with torch.no_grad():
            self.add_base_intensities()
            vessel_intensities, onehot, vessel_mask = self.sample_vessels(self.vessel_labels)
            out_vol = self.parenchyma * vessel_intensities
            out_vol = noise.RandomGammaNoiseTransform(0.1, 0.8)(out_vol)
            if self.banding == True:
                out_vol = self.slicewise(out_vol)
        return out_vol, onehot

    def sample_vessels(self, vessel_labels):
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int16, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int16, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks

    def add_base_intensities(self):
        self.parenchyma = labels.RandomSmoothLabelMap(
            RandInt(2, 5)(), RandInt(3, 10)()
            )(self.parenchyma).to('cuda') + 1
        
        self.parenchyma = self.parenchyma.to(torch.float32)
        if self.blur == True:
            if Uniform(0, 1)() <= 0.8:
                kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
                idx = RandInt(0, len(kernel_sizes)-1)() 
                self.parenchyma = transforms.GaussianBlur(
                    kernel_sizes[idx], Uniform(0.1, 20)()
                    )(self.parenchyma)

    def sample_label_intensity(self, labels, unique_ids):
        if self.light_vessels == True:
            intensity_ = Uniform(2, 10)
        elif self.light_vessels == False: 
            intensity_a = Uniform(0, 0.5)
            intensity_b = Uniform(1.5, 2)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if self.light_vessels == True:
                intensity = intensity_()
                onehot[2, ...].masked_fill_(labels == id, 1)
            elif self.light_vessels == False:
                if random.randint(0, 1) == 1:
                    # Make darker vessels
                    intensity = intensity_a()
                    onehot[1, ...].masked_fill_(labels == id, 1)
                else:
                    # Make brighter vessels
                    intensity = intensity_b()
                    onehot[2, ...].masked_fill_(labels == id, 1)

            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)

        if self.blur == True:
            if RandInt(0, 1)() == 1:
                if self.light_vessels == True:
                    scaling_tensor = transforms.GaussianBlur(3, Uniform(0.5, 2)())(scaling_tensor)
                if self.light_vessels == False:
                    scaling_tensor = transforms.GaussianBlur(3, 0.5)(scaling_tensor)
        return scaling_tensor, onehot, mask
                
    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def slicewise(self, tensor):
        if random.randint(0, 1) == 1:
            tensor = intensity.RandomSlicewiseMulFieldTransform()(tensor)
        else:
            tensor = intensity.RandomMulFieldTransform()(tensor)
        return tensor
    
class YibeiVesselsAndBalls(nn.Module):
    
    def __init__(self, allow_light_vessels=True, blur=True, banding=True, balls=True):
        super(YibeiVesselsAndBalls, self).__init__()
        self.allow_light_vessels = allow_light_vessels
        self.blur = blur
        self.balls = balls
        self.banding = banding

    @autocast()
    def forward(self, vessel_labels):
        self.vessel_labels = vessel_labels.to('cuda')
        self.shape = list(vessel_labels.shape)
        self.parenchyma = torch.ones(size=self.shape).to('cuda')
        if self.allow_light_vessels == True:
            if RandInt(0, 1)() == 0:
                self.light_vessels = True
            else:
                self.light_vessels = False
        with torch.no_grad():
            self.add_base_intensities()
            vessel_intensities, onehot, vessel_mask = self.sample_vessels(self.vessel_labels)
            out_vol = self.parenchyma * vessel_intensities
            if self.balls == True:
                out_vol = self.add_balls(out_vol)
            out_vol = noise.RandomGammaNoiseTransform(0.1, 0.8)(out_vol)
            if self.banding == True:
                out_vol = self.slicewise(out_vol)

        return out_vol, onehot

    def sample_vessels(self, vessel_labels):
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int16, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int16, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks

    def add_base_intensities(self):
        self.parenchyma = labels.RandomSmoothLabelMap(
            RandInt(2, 5)(), RandInt(3, 10)()
            )(self.parenchyma).to('cuda') + 1
        
        self.parenchyma = self.parenchyma.to(torch.float32)
        if self.blur == True:
            if Uniform(0, 1)() <= 0.8:
                kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
                idx = RandInt(0, len(kernel_sizes)-1)() 
                self.parenchyma = transforms.GaussianBlur(
                    kernel_sizes[idx], Uniform(0.1, 20)()
                    )(self.parenchyma)

    def sample_label_intensity(self, labels, unique_ids):
        if self.light_vessels == True:
            intensity_ = Uniform(1.5, 10)
        elif self.light_vessels == False: 
            intensity_a = Uniform(0, 0.5)
            intensity_b = Uniform(1.5, 2)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if self.light_vessels == True:
                intensity = intensity_()
                onehot[2, ...].masked_fill_(labels == id, 1)
            elif self.light_vessels == False:
                if random.randint(0, 1) == 1:
                    # Make darker vessels
                    intensity = intensity_a()
                    onehot[1, ...].masked_fill_(labels == id, 1)
                else:
                    # Make brighter vessels
                    intensity = intensity_b()
                    onehot[2, ...].masked_fill_(labels == id, 1)

            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)

        if self.blur == True:
            if RandInt(0, 1)() == 1:
                if self.light_vessels == True:
                    scaling_tensor = transforms.GaussianBlur(3, Uniform(0.5, 2)())(scaling_tensor)
                if self.light_vessels == False:
                    scaling_tensor = transforms.GaussianBlur(3, 0.5)(scaling_tensor)
        return scaling_tensor, onehot, mask
                
    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def slicewise(self, tensor):
        if random.randint(0, 1) == 1:
            tensor = intensity.RandomSlicewiseMulFieldTransform()(tensor)
        else:
            tensor = intensity.RandomMulFieldTransform()(tensor)
        return tensor

    def add_balls(self, tensor):
        dupe = torch.ones(self.shape, device=self.parenchyma.device)
        if self.light_vessels:
            intensity_sampler = Uniform(1, 10)()
        else:
            intensity_sampler = Uniform(0.1, 1.4)()

        # MAKING SURE BALLS DON'T LOOK LIKE VESSELS
        radius = RandInt(1, 20)()
        if radius >= 4:
            prob = Uniform(0.005, 0.01)()
        elif radius < 4:
            prob = Uniform(0.01, 0.1)()
        
        balls = labels.SmoothBernoulliDiskTransform(
            prob=prob,
            radius=RandInt(1, 20)(),
            shape=RandInt(5, 40)(),
            value=intensity_sampler
            )(dupe)
        balls = transforms.GaussianBlur(3, Uniform(0.1, 5)())(balls)
        tensor[balls > 0] *= balls[balls> 0]
        return tensor


class MultiContrastSemantic(nn.Module):
    
    def __init__(self, allow_light_vessels=False, blur=False, banding=False, balls=False, gamma_noise=False, gamma_shift=False, closest_i=0.5):
        super(MultiContrastSemantic, self).__init__()
        self.allow_light_vessels = allow_light_vessels
        self.blur = blur
        self.balls = balls
        self.banding = banding
        self.closest_i = closest_i
        self.gamma_noise = gamma_noise
        self.gamma_shift = gamma_shift

    #@autocast()
    def forward(self, vessel_labels):
        self.vessel_labels = vessel_labels.to('cuda')
        self.shape = list(vessel_labels.shape)
        parenchyma = torch.ones(size=self.shape, dtype=torch.float32).to('cuda').unsqueeze(0)
        if self.allow_light_vessels == True:
            if RandInt(0, 1)() == 0:
                self.light_vessels = True
            else:
                self.light_vessels = False
        else:
            self.light_vessels = False
        with torch.no_grad():
            parenchyma = self.add_base_intensities(parenchyma)
            vessel_intensities, onehot, vessel_mask = self.sample_vessels(self.vessel_labels)
            out_vol = parenchyma * vessel_intensities
            if self.balls == True:
                if Uniform(0, 1)() < 0.5:
                    out_vol = self.add_balls(out_vol)
            if self.gamma_noise == True:
                out_vol = noise.RandomGammaNoiseTransform(0.01, 0.8)(out_vol)
            if self.banding == True:
                out_vol = self.slicewise(out_vol)
            out_vol -= out_vol.min()
            out_vol /= out_vol.max()
            if self.gamma_shift == True:
                out_vol = self.gamma_transform(out_vol)
        return out_vol.squeeze(), onehot

    def gamma_transform(self, tensor):
        if Uniform(0, 1)() <= 0.9:
            if RandInt(0, 1)() == 0:
                # Make it brighter
                exp = Normal(0.75, 0.2)()
            else:
                exp = Normal(1.25, 0.2)()
            if exp > 2:
                exp = 2
            elif exp < 0.2:
                exp=0.2
            return tensor ** exp
        else:
            return tensor

    def sample_vessels(self, vessel_labels):
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks

    def add_base_intensities(self, parenchyma):
        parenchyma = labels.RandomSmoothLabelMap(
            RandInt(3, 100)(), RandInt(3, 10)()
            )(parenchyma).to('cuda') + 1
        
        parenchyma = parenchyma.to(torch.float32)
        if self.blur == True:
            if Uniform(0, 1)() <= 0.5:
                kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17]
                idx = RandInt(0, len(kernel_sizes)-1)() 
                parenchyma = transforms.GaussianBlur(
                    kernel_sizes[idx], Uniform(0.1, 20)()
                    )(parenchyma)
        return parenchyma
                
    def sample_label_intensity(self, labels, unique_ids):
        if self.light_vessels == True:
            intensity_ = Uniform(1 + (1-self.closest_i), 10)
        elif self.light_vessels == False: 
            intensity_a = Uniform(0, self.closest_i)
            intensity_b = Uniform(1 + (1-self.closest_i), 2)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if self.light_vessels == True:
                intensity = intensity_()
                onehot[2, ...].masked_fill_(labels == id, 1)
            elif self.light_vessels == False:
                if random.randint(0, 1) == 1:
                    # Make darker vessels
                    intensity = intensity_a()
                    onehot[1, ...].masked_fill_(labels == id, 1)
                else:
                    # Make brighter vessels
                    intensity = intensity_b()
                    onehot[2, ...].masked_fill_(labels == id, 1)

            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)

        if self.blur == True:
            if Uniform(0, 1)() > 0.5:
                if self.light_vessels == True:
                    scaling_tensor = transforms.GaussianBlur(3, Uniform(0.5, 2)())(scaling_tensor)
                if self.light_vessels == False:
                    scaling_tensor = transforms.GaussianBlur(3, 0.5)(scaling_tensor)
        return scaling_tensor, onehot, mask
                
    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def slicewise(self, tensor):
        if random.randint(0, 1) == 1:
            tensor = intensity.RandomSlicewiseMulFieldTransform()(tensor)
        else:
            tensor = intensity.RandomMulFieldTransform()(tensor)
        return tensor

    def add_balls(self, tensor):
        dupe = torch.ones(tensor.shape, device='cuda')
        if self.light_vessels:
            intensity_sampler = Uniform(1, 10)()
        else:
            intensity_sampler = Uniform(0.1, 1.4)()

        # MAKING SURE BALLS DON'T LOOK LIKE VESSELS
        radius = RandInt(1, 20)()
        if radius >= 4:
            prob = Uniform(0.005, 0.01)()
        elif radius < 4:
            prob = Uniform(0.01, 0.1)()
        
        balls = labels.SmoothBernoulliDiskTransform(
            prob=prob,
            radius=RandInt(1, 20)(),
            shape=RandInt(5, 40)(),
            value=intensity_sampler
            )(dupe)
        balls = transforms.GaussianBlur(3, Uniform(0.1, 5)())(balls)
        tensor[balls > 0] *= balls[balls> 0]
        return tensor



class SynthSanityCheck(nn.Module):
    
    def __init__(self, parenchyma_classes=3):
        super(SynthSanityCheck, self).__init__()
        self.parenchyma_classes = parenchyma_classes

    #@autocast()
    def forward(self, vessel_labels):
        self.vessel_labels = vessel_labels.to('cuda')
        self.shape = list(vessel_labels.shape)
        parenchyma = torch.ones(size=self.shape, dtype=torch.float32).to('cuda').unsqueeze(0)

        with torch.no_grad():
            parenchyma = self.add_base_intensities(parenchyma)
            vessel_intensities, onehot, vessel_mask = self.sample_vessels(self.vessel_labels)
            out_vol = parenchyma * vessel_intensities
            #out_vol = noise.RandomGammaNoiseTransform(0.01, 0.8)(out_vol)
            out_vol -= out_vol.min()
            out_vol /= out_vol.max()
            out_vol **= Uniform(0.1, 3)()

            # This SHOULD be uncommented in the future. Output should be (1, 1, 128, 128, 128)
            out_vol = out_vol.squeeze()#.unsqueeze(0)
            #out_vol = self.gamma_transform(out_vol)
        return out_vol, onehot

    def sample_vessels(self, vessel_labels):
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        if isinstance(unique_ids, torch.Tensor):
            vessel_intensities, onehot, vessel_masks = self.sample_label_intensity(
                vessel_labels, unique_ids
                )
        elif unique_ids == 0:
            vessel_intensities = torch.ones(self.shape, dtype=torch.int, device='cuda')
            vessel_masks = torch.zeros(self.shape, dtype=torch.int, device='cuda')
            onehot = torch.zeros((3, self.shape[0], self.shape[1], self.shape[2]), device='cuda')
            onehot[0, ...] = 1
        return vessel_intensities, onehot, vessel_masks

    def add_base_intensities(self, parenchyma):
        parenchyma = labels.RandomSmoothLabelMap(
            RandInt(3, self.parenchyma_classes)(), RandInt(3, 10)()
            )(parenchyma).to('cuda') + 1
        parenchyma = parenchyma.to(torch.float32)
        return parenchyma
                
    def sample_label_intensity(self, labels, unique_ids):
        intensity_a = Uniform(0, 0.5)
        intensity_b = Uniform(1.5, 2)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to('cuda')
        mask = torch.zeros(self.shape).to('cuda')
        onehot = torch.zeros(3, self.shape[0], self.shape[1], self.shape[2]).to('cuda')
        for id in unique_ids:
            # Get darker vessels
            if Uniform(0, 1)() < 0.5:
                onehot[2, ...].masked_fill_(labels == id, 1)
                intensity = intensity_b()
            else:
                if random.randint(0, 1) == 1:
                    # Make darker vessels
                    intensity = intensity_a()
                    onehot[1, ...].masked_fill_(labels == id, 1)
                else:
                    # Make brighter vessels
                    intensity = intensity_b()
                    onehot[2, ...].masked_fill_(labels == id, 1)

            scaling_tensor.masked_fill_(labels == id, intensity)
            mask.masked_fill_(labels == id, 1)
        onehot[0, ...].masked_fill_(mask == 0, 1)
        return scaling_tensor, onehot, mask
                
    def prune_labels(self, vessel_labels):
        if random.randint(0, 5) == 3:
            vessel_labels = torch.zeros(vessel_labels.shape)
            unique_ids = 0
        else:
            unique_ids = torch.unique(vessel_labels)
            n_unique_ids = len(unique_ids)
            n_labels_to_hide = torch.randint(
                n_unique_ids//10,
                n_unique_ids-1, [1]
            )
            vessel_ids_to_hide = unique_ids[
                torch.randperm(n_unique_ids)[:n_labels_to_hide]
            ]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels[vessel_labels == vessel_id] = 0
            # Removing all ID's from unique_ids that have been hidden
            unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids


class ParenchymaSynth(nn.Module):
    """
    A module for synthesizing parenchymal tissue structures with optional noise.

    This class generates synthetic parenchymal tissue structures and applies
    a sequence of transformations including base tissue generation, noise application,
    and intensity normalization. Noise types and parameters can be dynamically specified.

    Parameters
    ----------
    shape : list, optional
        Shape of the output tensor. Default is [1, 128, 128, 128].
    high_frequency_noise : dict, optional {'RandomGammaNoiseTransform',
        'RandomChiNoiseTransform', 'RandomGaussianNoiseTransform}
        A dictionary mapping noise transformation names to their parameters. If `None`,
        no high-frequency noise is applied. Default is None. 
    min_classes : int, optional
        The minimum number of different classes (tissue types) to generate in the base map.
        Default is 3.

    Methods
    -------
    forward()
        Generates the synthetic tissue image and applies transformations.
    normalize(p)
        Normalizes the tensor `p` by scaling its values to the range [0, 1].
    get_base_(p)
        Generates a base tensor representing different tissue types.
    high_frequency_noise_(p)
        Applies randomly selected noise transformations from the specified options in `high_frequency_noise`.
    
    Examples
    --------
    >>> synth = ParenchymaSynth()
    >>> output = synth()
    >>> print(output.shape)
    torch.Size([1, 128, 128, 128])
    """

    def __init__(self, shape=[1, 128, 128, 128], high_frequency_noise=None, min_classes=3):
        super(ParenchymaSynth, self).__init__()
        self.shape = shape
        self.device = 'cuda'
        self.dtype = torch.float32
        self.min_classes = min_classes
        self.high_frequency_noise = high_frequency_noise

    def forward(self):
        p = torch.ones(self.shape).to(self.device).to(self.dtype)
        p = self.get_base_(p).to(self.dtype)
        #p = self.high_frequency_noise_(p).to(self.dtype)
        p = inject_noise(p, 0.1)
        if Uniform(0, 1)() < 0.2:
            b = labels.SmoothBernoulliDiskTransform(prob=Uniform(0.001, 0.1)(), radius=10, shape=7, value=Uniform(0, 2)())(torch.zeros(p.shape)).to(torch.float32).to(self.device)
            p[b > 0] *= b[b > 0]
        p -= (p.min() + 1)
        return p

    def get_base_(self, p):
        p = labels.RandomSmoothLabelMap(RandInt(2, self.min_classes)(), RandInt(3, 8)())(p).to(torch.float32) + 1
        p = intensity.RandomSlicewiseMulFieldTransform(order=5, thickness=5)(p)
        return p.to(self.dtype)
    
    def high_frequency_noise_(self, p):
        noise_mapping = {
            'RandomGammaNoiseTransform': RandomGammaNoiseTransform,
            'RandomChiNoiseTransform': RandomChiNoiseTransform,
            'RandomGaussianNoiseTransform': RandomGaussianNoiseTransform,
        }
        if self.high_frequency_noise is not None:
            noises = []
            for i, (noise_name, noise_params) in enumerate(self.high_frequency_noise.items()):
                if noise_name in noise_mapping:
                    noise_instance = noise_mapping[noise_name]
                    if noise_params is not None:
                        noise_instance = noise_instance(*noise_params)
                    else:
                        noise_instance = noise_instance()
                    noises.append(noise_instance)
                else:
                    print("Noise model not found")
            idx = RandInt(0, len(noises)-1)()
            noise_model = noises[idx]
            return noise_model(p)
        else:
            return p

class VesselIntensitySynth(nn.Module):
    
    def __init__(self, min_difference=0.1):
        super(VesselIntensitySynth, self).__init__()
        self.dtype = torch.float32
        self.device = 'cuda'
        self.min_difference = min_difference

    def forward(self, vessel_labels):
        vessel_labels = vessel_labels.to(self.device)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = vessel_labels.unsqueeze(0)
        self.shape = vessel_labels.shape
        vessel_labels = vessel_labels.to(self.dtype)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        scaling_tensor, onehot, mask = self.sample_label_intensity(vessel_labels, unique_ids)
        return scaling_tensor, onehot

    def prune_labels(self, vessel_labels):
        unique_ids = torch.unique(vessel_labels)
        n_unique_ids = len(unique_ids)
        n_labels_to_hide = torch.randint(
            n_unique_ids//2,
            n_unique_ids-1, [1]
        )
        vessel_ids_to_hide = unique_ids[
            torch.randperm(n_unique_ids)[:n_labels_to_hide]
        ]
        for vessel_id in vessel_ids_to_hide:
            vessel_labels[vessel_labels == vessel_id] = 0
        # Removing all ID's from unique_ids that have been hidden
        unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def sample_label_intensity(self, labels, unique_ids):
        intensity_a = Uniform(0, 0.5)
        intensity_b = Uniform(1.5, 2.1)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to(self.device).to(self.dtype)
        mask = torch.zeros(self.shape).to(self.device).to(self.dtype)
        onehot = torch.zeros(3, self.shape[1], self.shape[2], self.shape[3]).to(self.device).to(self.dtype)
        for id in unique_ids:
            if random.randint(0, 1) == 1:
                # Make darker vessels
                intensity = intensity_a()
                onehot[1, ...].masked_fill_(labels[0] == id, 1)
            else:
                # Make brighter vessels
                intensity = intensity_b()
                onehot[2, ...].masked_fill_(labels[0] == id, 1)

            scaling_tensor.masked_fill_(labels[0] == id, intensity)
            mask.masked_fill_(labels[0] == id, 1)
        onehot[0, ...].masked_fill_(mask[0] == 0, 1)
        return scaling_tensor, onehot, mask
    
def inject_noise(tensor, noise_factor=0.1):
    """
    Injects random noise into a 3D tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to which noise will be added.
    noise_factor (float): The factor by which the noise will be scaled.
    
    Returns:
    torch.Tensor: The tensor with added noise.
    """
    # Ensure the input tensor is a 3D tensor
    if tensor.dim() != 4:
        raise ValueError("Input tensor must be 3-dimensional")
    
    # Generate random noise
    noise = torch.randn_like(tensor)
    
    # Scale the noise and add it to the input tensor
    noisy_tensor = tensor + noise_factor * noise
    
    return noisy_tensor

class FuseSynth(nn.Module):
    
    def __init__(self, synth_config=None):
        super(FuseSynth, self).__init__()
        self.synth_config = synth_config

    def normalize(self, t):
        min_val = torch.min(t)
        max_val = torch.max(t)
        normalized_tensor = 2 * (t - min_val) / (max_val - min_val) - 1
        return normalized_tensor
    
    def forward(self, vessel_labels):
        if self.synth_config is not None:
            x, y = VesselIntensitySynth(**self.synth_config['vessels'])(vessel_labels)
            p = ParenchymaSynth(shape=x.shape, **self.synth_config['parenchyma'])()
        else:
            x, y = VesselIntensitySynth()(vessel_labels)
            p = ParenchymaSynth(shape=x.shape)()
        p += 1
        final_out = torch.clone(p)
        p[y[0].unsqueeze(0) == 0] *= x[y[0].unsqueeze(0) == 0]
        final_out[y[0].unsqueeze(0) == 0] = p[y[0].unsqueeze(0) == 0]
        #final_out = inject_noise(final_out.squeeze(), 0.25)
        final_out = transforms.GaussianBlur(3,Uniform(0.01, 1)())(final_out) 
        final_out = self.normalize(final_out)
        return final_out.squeeze(), y

class SanityCheckDataset(Dataset):
    
    def __init__(self):
        path = '/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/sanity_check_synthvols'
        self.x_paths = sorted(glob.glob(f'{path}/x/*'))
        self.y_paths = sorted(glob.glob(f'{path}/y/*'))

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = torch.load(self.x_paths[idx])
        y = torch.load(self.y_paths[idx])
        return x, y
    

class ParenchymaSynth1(nn.Module):
    """
    A module for synthesizing parenchymal tissue structures with optional noise.

    This class generates synthetic parenchymal tissue structures and applies
    a sequence of transformations including base tissue generation, noise application,
    and intensity normalization. Noise types and parameters can be dynamically specified.

    Parameters
    ----------
    shape : list, optional
        Shape of the output tensor. Default is [1, 128, 128, 128].
    high_frequency_noise : dict, optional {'RandomGammaNoiseTransform',
        'RandomChiNoiseTransform', 'RandomGaussianNoiseTransform}
        A dictionary mapping noise transformation names to their parameters. If `None`,
        no high-frequency noise is applied. Default is None. 
    min_classes : int, optional
        The minimum number of different classes (tissue types) to generate in the base map.
        Default is 3.

    Methods
    -------
    forward()
        Generates the synthetic tissue image and applies transformations.
    normalize(p)
        Normalizes the tensor `p` by scaling its values to the range [0, 1].
    get_base_(p)
        Generates a base tensor representing different tissue types.
    high_frequency_noise_(p)
        Applies randomly selected noise transformations from the specified options in `high_frequency_noise`.
    
    Examples
    --------
    >>> synth = ParenchymaSynth()
    >>> output = synth()
    >>> print(output.shape)
    torch.Size([1, 128, 128, 128])
    """

    def __init__(self, shape=[1, 128, 128, 128], high_frequency_noise=None, min_classes=3):
        super(ParenchymaSynth1, self).__init__()
        self.shape = shape
        self.device = 'cuda'
        self.dtype = torch.float32
        self.min_classes = min_classes
        self.high_frequency_noise = high_frequency_noise

    def forward(self):
        p = torch.ones(self.shape).to(self.device).to(self.dtype)
        p = self.get_base_(p).to(self.dtype)
        p = inject_noise(p, 0.1)
        if Uniform(0, 1)() < 0.2:
            b = labels.SmoothBernoulliDiskTransform(prob=Uniform(0.001, 0.1)(), radius=10, shape=7, value=Uniform(0, 2)())(torch.zeros(p.shape)).to(torch.float32).to(self.device)
            p[b > 0] *= b[b > 0]
        p -= (p.min() + 1)
        return p

    def get_base_(self, p):
        p = labels.RandomSmoothLabelMap(RandInt(2, self.min_classes)(), RandInt(3, 8)())(p).to(torch.float32) + 1
        p = intensity.RandomSlicewiseMulFieldTransform(order=5, thickness=5)(p)
        return p.to(self.dtype)
    

class VesselIntensitySynth1(nn.Module):
    
    def __init__(self, min_difference=0.1):
        super(VesselIntensitySynth1, self).__init__()
        self.dtype = torch.float32
        self.device = 'cuda'
        self.min_difference = min_difference

    def forward(self, vessel_labels):
        vessel_labels = vessel_labels.to(self.device)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        vessel_labels = vessel_labels.unsqueeze(0)
        self.shape = vessel_labels.shape
        vessel_labels = vessel_labels.to(self.dtype)
        vessel_labels, unique_ids = self.prune_labels(vessel_labels)
        scaling_tensor, onehot, mask = self.sample_label_intensity(vessel_labels, unique_ids)
        return scaling_tensor, onehot

    def prune_labels(self, vessel_labels):
        unique_ids = torch.unique(vessel_labels)
        n_unique_ids = len(unique_ids)
        n_labels_to_hide = torch.randint(
            n_unique_ids//2,
            n_unique_ids-1, [1]
        )
        vessel_ids_to_hide = unique_ids[
            torch.randperm(n_unique_ids)[:n_labels_to_hide]
        ]
        for vessel_id in vessel_ids_to_hide:
            vessel_labels[vessel_labels == vessel_id] = 0
        # Removing all ID's from unique_ids that have been hidden
        unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def sample_label_intensity(self, labels, unique_ids):
        intensity_a = Uniform(0, 0.5)
        intensity_b = Uniform(1.5, 2.1)
            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(self.shape).to(self.device).to(self.dtype)
        mask = torch.zeros(self.shape).to(self.device).to(self.dtype)
        #print(self.shape)
        onehot = torch.zeros(3, self.shape[1], self.shape[2], self.shape[3]).to(self.device).to(self.dtype)
        for id in unique_ids:
            if random.randint(0, 1) == 1:
                # Make darker vessels
                intensity = intensity_a()
                onehot[1, ...].masked_fill_(labels[0] == id, 1)
            else:
                # Make brighter vessels
                intensity = intensity_b()
                onehot[2, ...].masked_fill_(labels[0] == id, 1)

            scaling_tensor.masked_fill_(labels[0] == id, intensity)
            mask.masked_fill_(labels[0] == id, 1)
        onehot[0, ...].masked_fill_(mask[0] == 0, 1)
        return scaling_tensor, onehot, mask
    
def inject_noise(tensor, noise_factor=0.1):
    """
    Injects random noise into a 3D tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor to which noise will be added.
    noise_factor (float): The factor by which the noise will be scaled.
    
    Returns:
    torch.Tensor: The tensor with added noise.
    """
    # Ensure the input tensor is a 3D tensor
    if tensor.dim() != 4:
        raise ValueError("Input tensor must be 3-dimensional")
    
    # Generate random noise
    noise = torch.randn_like(tensor)
    
    # Scale the noise and add it to the input tensor
    noisy_tensor = tensor + noise_factor * noise
    
    return noisy_tensor

class FuseSynth1(nn.Module):
    
    def __init__(self, synth_config=None):
        super(FuseSynth1, self).__init__()
        self.synth_config = synth_config
    
    def forward(self, vessel_labels):
        if self.synth_config is not None:
            x, y = VesselIntensitySynth1(**self.synth_config['vessels'])(vessel_labels)
            p = ParenchymaSynth1(shape=x.shape, **self.synth_config['parenchyma'])()
        else:
            x, y = VesselIntensitySynth1()(vessel_labels)
            p = ParenchymaSynth1(shape=x.shape)()
        p += 1
        final_out = torch.clone(p)
        p[y[0].unsqueeze(0) == 0] *= x[y[0].unsqueeze(0) == 0]
        final_out[y[0].unsqueeze(0) == 0] = p[y[0].unsqueeze(0) == 0]
        #final_out = inject_noise(final_out.squeeze(), 0.25)
        final_out = transforms.GaussianBlur(3,Uniform(0.1, 1)())(final_out) 
        final_out = transforms.Normalize(0.7, 0.22)(final_out)
        return final_out.squeeze(), y



####
class VesselDataset(Dataset):
    
    def __init__(self, exp_n=1, transform=None):
        self.path = f"/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp{f'{exp_n}'.zfill(4)}"
        self.x_paths = glob.glob(f'{self.path}/*label*')
        self.transform = transform

    def __len__(self):
        return len(self.x_paths)

    def __getitem__(self, idx):
        x = torch.from_numpy(nib.load(self.x_paths[idx]).get_fdata())
        x = x.to('cuda').to(torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        return x


class AddVessels(nn.Module):
    """
    Module to add vessels to images with specific intensity and masking.

    Parameters
    ----------
    label_type : str, optional
        Type of label to add ('dark', 'light', or 'both').
    light_range : list, optional
        Range for light vessel intensity (default is [1.9, 2]).
    dark_range : list, optional
        Range for dark vessel intensity (default is [0, 0.1]).

    Attributes
    ----------
    device : str
        Device to run the model on ('cuda' for GPU).
    shape : torch.Size
        Shape of the vessel labels.
    """
    
    def __init__(self,
                 label_type: str = 'both',
                 light_range: list = [1.9, 2],
                 dark_range: list = [0, 0.1]
                 ) -> None:
        """
        Initialize AddVessels.
        """
        super(AddVessels, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.light_range = light_range
        self.dark_range = dark_range
        self.label_type = self._handle_label_type(label_type)


    def forward(self, vessel_labels: torch.Tensor, 
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for adding vessels.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the scaling tensor and one-hot encoded tensor.
        """
        self.shape = vessel_labels.shape
        vessel_labels = self._apply_geometric_transofrms(vessel_labels)
        vessel_labels = vessel_labels.unsqueeze(0).to(torch.float32)
        vessel_labels, unique_ids = self._prune_labels(vessel_labels)
        scaling_tensor, onehot, mask = self._sample_label_intensity(vessel_labels, unique_ids)
        return scaling_tensor, onehot
    
    def _handle_label_type(self, label_type: str) -> str:
        if label_type not in {'dark', 'light', 'both'}:
            raise ValueError("label_type must be 'dark', 'light', or 'both'")
        self.n_classes = 3 if label_type == 'both' else 1
        return label_type

    def _apply_geometric_transofrms(self, vessel_labels: torch.Tensor) -> torch.Tensor:
        """
        Apply random flip transforms to the vessel labels.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels.

        Returns
        -------
        torch.Tensor
            Transformed vessel labels.
        """
        for _ in range(3):
            vessel_labels = fov.RandomFlipTransform()(vessel_labels)
        return vessel_labels
    

    def _prune_labels(self, vessel_labels: torch.Tensor,
                     prune_factor: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prune vessel labels to hide some of them.

        Parameters
        ----------
        vessel_labels : torch.Tensor
            Tensor of vessel labels.
        prune_factor : int
            How aggressively to prune labels. Default is 2.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Pruned vessel labels and list of unique IDs.
        """
        self.prune_factor = prune_factor
        unique_ids = torch.unique(vessel_labels)
        n_unique_ids = len(unique_ids)
        n_labels_to_hide = torch.randint(
            n_unique_ids//self.prune_factor,
            n_unique_ids-1, [1]
        )
        vessel_ids_to_hide = unique_ids[
            torch.randperm(n_unique_ids)[:n_labels_to_hide]
        ]
        for vessel_id in vessel_ids_to_hide:
            vessel_labels[vessel_labels == vessel_id] = 0
        # Removing all ID's from unique_ids that have been hidden
        #unique_ids = unique_ids[~torch.isin(unique_ids, vessel_ids_to_hide)]
        unique_ids = unique_ids[~torch.any(unique_ids[:, None] == vessel_ids_to_hide, dim=-1)]
        return vessel_labels, unique_ids
    
    def _sample_label_intensity(self,
                               labels: torch.Tensor, 
                               unique_ids: torch.Tensor,
                               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample label intensity and create scaling tensors.

        Parameters
        ----------
        labels : torch.Tensor
            Tensor of labels.
        unique_ids : torch.Tensor
            Unique IDs of the labels.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Scaling tensor, one-hot encoded tensor, and mask tensor.
        """            
        # making sure unique_ids doesn't have any zeros
        unique_ids = unique_ids[unique_ids != 0]
        scaling_tensor = torch.ones(
            self.shape, device=self.device, dtype=torch.float32)
        mask = torch.zeros(
            self.shape, device=self.device, dtype=torch.float32)
        onehot = torch.zeros(
            self.n_classes, *self.shape, device=self.device, dtype=torch.float32)
        for id in unique_ids:
            sampler, channel = self._sample_intensity()
            onehot[channel, ...].masked_fill_(labels[0] == id, 1)
            scaling_tensor.masked_fill_(labels[0] == id, sampler())
            mask.masked_fill_(labels[0] == id, 1)
        onehot[0, ...].masked_fill_(mask[0] == 0, 1)
        return scaling_tensor, onehot, mask
    
    
    def _sample_intensity(self) -> Union[Uniform, Tuple[Uniform, int]]:
        """
        Return the appropriate intensity sampler based on the label type.

        Returns
        -------
        Union[Uniform, Tuple[Uniform, int]]
            Intensity sampler(s) and channel based on the label type.
        """
        if self.label_type == 'dark':
            return (Uniform(*self.dark_range), 0)
        elif self.label_type == 'light':
            return (Uniform(*self.light_range), 0)
        elif self.label_type == 'both':
            return random.choice([
                (Uniform(*self.dark_range), 1), 
                (Uniform(*self.light_range), 2)
                ])
        else:
            raise ValueError("label_type must be 'dark', 'light', or 'both'")
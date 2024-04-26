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
from torch import nn
from torch.cuda.amp import autocast  # Utilizing automatic mixed precision
from torch.utils.data import Dataset
from torchvision.transforms import GaussianBlur

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
from cornucopia.cornucopia.random import Uniform, Fixed, RandInt
from cornucopia.cornucopia.intensity import QuantileTransform
from cornucopia.cornucopia import fov


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
        self.label_tensor_backend = dict(device='cuda', dtype=torch.int64)
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
                 subset=None,
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
        vessel_labels = torch.unique(vessel_labels_tensor).sort().values.nonzero().squeeze()

        # Randomly make a negative control
        if random.randint(1, 10) == 7:
            vessel_labels_tensor[vessel_labels_tensor > 0] = 0
            n_unique_ids = 0
        else:
            # Hide some vessels randomly
            n_unique_ids = len(vessel_labels)
            number_vessels_to_hide = torch.randint(n_unique_ids//10, n_unique_ids-1, [1])
            vessel_ids_to_hide = vessel_labels[torch.randperm(n_unique_ids)[:number_vessels_to_hide]]
            for vessel_id in vessel_ids_to_hide:
                vessel_labels_tensor[vessel_labels_tensor == vessel_id] = 0

        # Apply DC offset
        parenchyma = self.parenchyma_(vessel_labels_tensor)
        #dc_offset = random.uniform(0, 0.25)
        #parenchyma += dc_offset
        final_volume = parenchyma.clone()

        # Determine if there are any vessels to deal with
        if n_unique_ids > 0:
            # synthesize vessels (grouped by intensity)
            vessels = self.vessels_(vessel_labels_tensor) 
            if self.synth_params == 'complex':
                pass
                # texturize those vessels!!!
                vessel_texture = self.vessel_texture_(vessel_labels_tensor)
                vessels[vessel_labels_tensor > 0] *= vessel_texture[vessel_labels_tensor > 0]
            final_volume[vessel_labels_tensor > 0] *= vessels[vessel_labels_tensor > 0]        
        # Normalizing
        final_volume = QuantileTransform()(final_volume)
        # final output needs to be in float32 or else torch throws mismatch error between this and weights tensor.
        final_volume = final_volume.to(torch.float32)
        return final_volume, vessel_labels_tensor.clip_(0, 1)
        

    def parenchyma_(self, vessel_labels_tensor:torch.Tensor):
        """
        Generate parenchyma based on vessel labels using GPU optimized operations.

        Parameters
        ----------
        vessel_labels_tensor : torch.Tensor
            Tensor of vessels with unique ID integer labels.
        """
        # Create the label map of parenchyma but convert to float32 for further computations
        # Add 1 so that we can work with every single pixel (no zeros)
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
            if random.randint(0, 2) == 90:
                balls = BernoulliDiskTransform(
                    prob=1e-2,
                    radius=random.randint(1, 4),
                    value=random.uniform(0, 2)
                    )(parenchyma)[0]
                if random.randint(0, 2) == 1:
                    balls = ElasticTransform(shape=5)(balls).detach()
                parenchyma *= balls
                # Applying z-stitch artifact
            parenchyma = RandomSlicewiseMulFieldTransform(
                thickness=self.thickness_
                )(parenchyma)
        elif self.synth_params == 'simple':
            # Give bias field in lieu of slicewise transform
            pass
        #parenchyma = RandomMulFieldTransform(5)(parenchyma)
        parenchyma = RandomGammaTransform((self.gamma_a, self.gamma_b))(parenchyma)
        parenchyma = QuantileTransform()(parenchyma)
        #parenchyma -= parenchyma.min()
        #parenchyma /= parenchyma.max()
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
            scaling_tensor.masked_fill_(vessel_labels_tensor == int_n, intensity * vessel_texture_fix_factor)
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
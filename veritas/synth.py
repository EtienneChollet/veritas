__all__ = [
    'VesselSynth',
    'OctVolSynth',
    'OctVolSynthDataset'
]

# Standard imports
import os
import json
import torch
from torch import nn
import math as pymath
import nibabel as nib
import random

# Custom Imports
from veritas.utils import PathTools, JsonTools
from vesselsynth.vesselsynth.utils import backend
from vesselsynth.vesselsynth.io import default_affine
from vesselsynth.vesselsynth.synth import SynthVesselOCT
from cornucopia.cornucopia.labels import RandomSmoothLabelMap
from cornucopia.cornucopia.noise import RandomGammaNoiseTransform
from cornucopia.cornucopia import RandomSlicewiseMulFieldTransform, RandomGammaTransform
from cornucopia.cornucopia.random import Uniform, Fixed, RandInt


from veritas.utils import PathTools
import matplotlib.pyplot as plt
import glob
from torch.utils.data import Dataset
import numpy as np



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
        Parameters
        ----------
        device : 'cuda' or 'cpu' str
            Which device to run computations on
        json_param_path : str
            Location of json file containing parameters
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

        begin_at : int
            Volume number to begin at.
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
        Check backend for CUDA.
        """
        self.device = torch.device(self.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available, using CPU.')
            self.device = 'cpu'


    def outputShape(self):
        if not isinstance(self.shape, list):
            self.shape = [self.shape]
        while len(self.shape) < 3:
            self.shape += self.shape[-1:]


    def saveVolume(self, volume_n:int, volume_name:str, volume:torch.Tensor):
        """
        Save volume as nii.gz.

        Parameters
        ----------
        volume_n : int
            Volume "ID" number
        volume_name : str
            Volume name ['prob', 'label', "level", "nb_levels",\
            "branch", "skeleton"]
        volume : tensor
            Vascular network tensor corresponding with volume_name
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
        file = open(abspath, 'w')
        file.write(json_object)
        file.close()


class OctVolSynthDataset(Dataset):
    """
    Synthesize OCT intensity volume from vascular network.
    """
    def __init__(self,
                 exp_path:str=None,
                 label_type:str='label',
                 device:str="cuda",
                 synth_params='complex'
                 ):
        """
        Parameters
        ----------
        exp_path : str
            Path to synthetic experiment dir.
        """
        self.device = device
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
        Parameters
        ----------
        idx : int
            Volume ID number.
        save_nifti : bool
            Save volume as nifti to sample dir.
        make_fig : bool
            Make figure and print it to ipynb output.
        save_fig : bool
            Generate and save figure to sample dir.
        """
        # Loading nifti and affine
        label_nifti = nib.load(self.label_paths[idx])
        label_tensor = label_nifti.get_fdata()
        label_affine = label_nifti.affine
        # Reshaping
        print(np.unique(label_tensor))
        label_tensor = np.clip(label_tensor, 0, 255).astype(np.uint8)[None]
        label_tensor = torch.from_numpy(label_tensor)
        # Synthesizing volume
        im, prob = OctVolSynth(
            synth_params=self.synth_params
        )(label_tensor)
        # Converting image and prob map to numpy. Reshaping
        im = im.detach().cpu().numpy().squeeze()
        if self.label_type == 'label':
            prob = prob.to(torch.uint8).detach().cpu().numpy().squeeze()
            #prob[prob >= 1] = 1
            # Should be [0, 1]
            #print(prob.min(), prob.max())
        #elif self.label_type != 'label':
        #    prob = nib.load(self.y_paths[idx]).get_fdata()
        #    prob[prob > 0] = 1
        #    prob[prob < 0] = 0
        else:
            pass
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
    ### FROM YAEL
    
    def __init__(self,
                 inputs,
                 subset=None,
                 ):
        """
        Parameters
        ----------
        inputs : list[file] or directory or pattern
            Input vessel labels.
        subset : int
            Number of examples for combined training and validation.

        Returns
        -------
        label as shape as torch.Tensor of shape (1, n, n, n)
        """
        self.subset=slice(subset)
        self.inputs=np.asarray(inputs[self.subset]) # uses less RAM in multithreads

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        label = nib.load(self.inputs[idx]).get_fdata()
        label = np.clip(label, 0, 127).astype(np.int8)[None]
        return label


class OctVolSynth(nn.Module):
    """
    Synthesize OCT-like volumes from vascular network.
    """
    def __init__(self,
                 synth_params:str='complex',
                 dtype=torch.float32,
                 device:str='cuda',
                 ):
        super().__init__()
        """
        Parameters
        ----------
        dtype : torch.dtype
            Type of data that will be used in synthesis.
        device : {'cuda', 'cpu'}
            Device that will be used for syntesis.
        synth_params : {'complex', 'simple'}
            How to synthesize intensity volumes.
        """
        self.synth_params=synth_params
        self.dtype = dtype
        self.device = device
        self.json_dict = JsonTools(f'scripts/2_imagesynth/imagesynth_params-{self.synth_params}.json').read()
        self.speckle_a = float(self.json_dict['speckle'][0])
        self.speckle_b = float(self.json_dict['speckle'][1])
        self.gamma_a = float(self.json_dict['gamma'][0])
        self.gamma_b = float(self.json_dict['gamma'][1])
        self.thickness_ = int(self.json_dict['z_decay'][0])
        self.nb_classes_ = int(self.json_dict['parenchyma']['nb_classes'])
        self.shape_ = int(self.json_dict['parenchyma']['shape'])
        self.i_max = float(self.json_dict['imax'])
        self.i_min = float(self.json_dict['imin'])
    
    def forward(self, vessel_labels_tensor:torch.Tensor) -> tuple:
        """
        Parameters
        ----------
        vessel_labels_tensor : tensor
            Tensor of vessels with unique ID integer labels
        """
        # synthesize the main parenchyma (background tissue)
        # Get sorted list of all vessel labels for later
        self.vessel_labels = sorted(vessel_labels_tensor.unique().tolist())
        self.vessel_labels = [i for i in self.vessel_labels if i != 0]
        parenchyma = self.parenchyma_(vessel_labels_tensor)
        self.n_unique_ids = len(self.vessel_labels)
        # Determine if there are any vessels to deal with
        if self.n_unique_ids == 0:
            final_volume = parenchyma
        elif self.n_unique_ids >= 1:
            # synthesize vessels (grouped by intensity)
            vessels = self.vessels_(vessel_labels_tensor)
            # Create a parenchyma-like mask to texturize vessels 
            #if self.synth_params == 'complex':
            #    vessel_texture = self.vessel_texture_(vessel_labels_tensor)
                # Texturize those vessels!!
            vessels[vessels == 0] = 1
            #vessels = torch.mul(vessels, vessel_texture)
            final_volume = torch.mul(parenchyma, vessels)
        # Normalizing
        final_volume -= final_volume.min()
        final_volume /= final_volume.max()
        # final output needs to be in float32 or else torch throws mismatch error between this and weights tensor.
        final_volume = final_volume.to(torch.float32)
        vessel_labels_tensor[vessel_labels_tensor >= 1] = 1
        return final_volume, vessel_labels_tensor
        

    def parenchyma_(self, vessel_labels_tensor:torch.Tensor):
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
        parenchyma = RandomSmoothLabelMap(
            nb_classes=random.randint(2, self.nb_classes_),
            shape=random.randint(2, self.shape_)
            )(vessel_labels_tensor).to(self.dtype) + 1
        # Applying speckle noise model
        parenchyma = RandomGammaNoiseTransform(
            sigma=Uniform(self.speckle_a, self.speckle_b)
            )(parenchyma)[0]
        # Applying z-stitch artifact
        if self.synth_params == 'complex':
            parenchyma = RandomSlicewiseMulFieldTransform(
                thickness=self.thickness_
                )(parenchyma)
        parenchyma = RandomGammaTransform((self.gamma_a, self.gamma_b))(parenchyma)
        parenchyma -= parenchyma.min()
        parenchyma /= parenchyma.max()
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
        # Need to put this guy on CPU before mask fill operation.
        # Generate an empty tensor that we will fill with vessels and their
        # scaling factors to imprint or "stamp" onto parenchymal volume
        scaling_tensor = torch.zeros(
            vessel_labels_tensor.shape,
            dtype=self.dtype,
            device=vessel_labels_tensor.device)
        # Calculate the number of elements (vessels) in each intensity group
        # Iterate through each vessel group based on their unique intensity
        for int_n in range(self.n_unique_ids):
            # Assign intensity for this group from uniform distro
            intensity = Uniform(0.001, 0.75)()
            # Get label ID's of all vessels that will be assigned to this intensity
            vessel_labels_at_i = self.vessel_labels[int_n : int_n + 1]
            # Fill the empty tensor with the vessel scaling factors
            for ves_n in vessel_labels_at_i:
                scaling_tensor.masked_fill_(vessel_labels_tensor == ves_n, intensity)    
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
        nb_classes = RandInt(2, self.nb_classes_)()
        parenchyma = RandomSmoothLabelMap(
            nb_classes=Fixed(nb_classes),
            shape=self.shape_
            )(vessel_labels_tensor).to(self.dtype) + 1
        # Applying speckle noise model
        #parenchyma = RandomGammaTransform((self.gamma_a, self.gamma_b))(parenchyma)
        parenchyma -= parenchyma.min()
        parenchyma /= parenchyma.max()
        parenchyma[parenchyma <= 0] = 1e-1
        parenchyma /= 4
        parenchyma -= 1
        parenchyma = abs(parenchyma)
        return parenchyma

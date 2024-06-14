__all__ = [
    'MatchHistogram'
    'Options',
    'Thresholding',
    'PathTools',
    'JsonTools',
    'Checkpoint'
]
# Standard Imports
import os
import sys
import glob
import json
import torch
import shutil
import numpy as np
import torch.nn.functional as F
import scipy.ndimage as ndimage
from torch import nn


class MatchHistogram(nn.Module):
    def __init__(self, mean=0.0, std=0.2, num_bins=256):
        """
        Histogram Matching Module to map the intensity values of an image to follow a normal distribution.

        Parameters
        ----------
        mean : float, optional
            Mean of the normal distribution, by default 0.0.
        std : float, optional
            Standard deviation of the normal distribution, by default 0.2.
        num_bins : int, optional
            Number of bins for histogram, by default 256.
        """
        super(MatchHistogram, self).__init__()
        self.mean = mean
        self.std = std
        self.num_bins = num_bins

    def calculate_cdf(self, hist):
        """Calculate the cumulative distribution function (CDF) for a histogram."""
        cdf = hist.cumsum(0)
        cdf_normalized = cdf / cdf[-1]
        return cdf_normalized

    def forward(self, source):
        """
        Forward pass to perform histogram matching.

        Parameters
        ----------
        source : torch.Tensor
            Source image (HxW), normalized between -1 and 1.

        Returns
        -------
        matched : torch.Tensor
            The transformed source image with histogram matching a normal distribution.
        """
        device = source.device

        # Normalize the source image to the range [0, 255] for histogram computation
        source_normalized = ((source + 1) / 2 * 255).clamp(0, 255).long()

        # Compute the histogram and CDF of the source image
        src_hist = torch.histc(source_normalized.float(), bins=self.num_bins, min=0, max=255).to(device)
        src_cdf = self.calculate_cdf(src_hist)

        # Create the normal distribution CDF
        normal_values = torch.linspace(-1, 1, self.num_bins, device=device)
        normal_cdf = torch.distributions.Normal(self.mean, self.std).cdf(normal_values)
        normal_cdf = normal_cdf / normal_cdf[-1]  # Normalize to range [0, 1]

        # Create a lookup table to map the pixel values
        lookup_table = torch.zeros(self.num_bins, device=device)
        for src_pixel in range(self.num_bins):
            normal_pixel = torch.searchsorted(normal_cdf, src_cdf[src_pixel])
            lookup_table[src_pixel] = normal_pixel

        # Apply the lookup table to the source image
        source_flat = source_normalized.flatten().long()
        matched_flat = lookup_table[source_flat]
        matched = matched_flat.view(source.shape).float()

        # Convert matched image back to the range [-1, 1]
        matched = matched / (self.num_bins - 1) * 2 - 1

        return matched


class Options(object):
    """
    Base class for options.
    """
    def __init__(self, cls):
        self.cls = cls
        self.attribute_dict = self.cls.__dict__

    def out_filepath(self, dir=None):
        """
        Determine out filename. Same dir as volume.
        """
        stem = f"{self.attribute_dict['tensor_name']}-prediction"
        stem += f"_stepsz-{self.attribute_dict['step_size']}"
        try:
            stem += f"_{self.attribute_dict['accuracy_name']}-{self.attribute_dict['accuracy_val']}"
        except:
            pass
        stem += '.nii.gz'
        if dir is None:
            self.out_dir = f"/{self.attribute_dict['volume_dir']}/predictions"
        else:
            self.out_dir = dir
        self.full_path = f"{self.out_dir}/{stem}"
        return self.out_dir, self.full_path



class Thresholding(object):
    """
    Decide if and how to threshold. Perform thresholding.
    """
    def __init__(self, prediction_tensor:torch.Tensor,
                 ground_truth_tensor:torch.Tensor,
                 threshold:{float, False, 'auto'}=0.5,
                 compute_accuracy:bool=False
                 ):
        """
        Parameters
        ----------
        prediction_tensor : tensor[float]
            Tensor of prediction volume.
        ground_truth_tensor : tensor[bool]
            Ground truth tensor.
        threshold : {float, False, 'auto'}
            Intensity value at which to threshold. If False, return prob map.
        compute_accuracy : bool
            If true, compute accuracy and print to console.
        """
        self.prediction_tensor = prediction_tensor
        self.ground_truth_tensor = ground_truth_tensor
        self.threshold = threshold
        self.compute_accuracy = compute_accuracy

    def apply(self):
        """
        Run thresholding.
        """
        if self.threshold == False:
            # Return the unaltered probability map
            print("\nNot thresholding...")
            return self.prediction_tensor, None, None
        elif isinstance(self.threshold, float):
            print('\nApplying fixed threshold...')
            return self.fixedThreshold()
        elif self.threshold == 'auto' and isinstance(self.ground_truth_tensor,
                                                   torch.Tensor):
            return self.autoThreshold()
        else:
            print("\nCan't do the thresholding. Check your settings")
            exit(0)
        

    def autoThreshold(self, start:float=0.05, stop:float=0.95, step:float=0.05):
        """
        Auto threshold volume.

        Parameters
        ----------
        start : float
            Intensity to begin thresholding
        stop : float
            Intensity to stop thresholding
        step : float
            Increase from start to stop with this step size
        """
        threshold_lst = np.arange(start, stop, step)
        accuracy_lst = []

        for thresh in threshold_lst:
            temp = self.prediction_tensor.clone()
            temp[temp >= thresh] = 1
            temp[temp <= thresh] = 0
            accuracy = dice(temp, self.ground_truth_tensor, multiclass=False)
            accuracy_lst.append(accuracy)

        max_accuracy_index = accuracy_lst.index(max(accuracy_lst))
        threshold, accuracy = threshold_lst[max_accuracy_index], accuracy_lst[max_accuracy_index]
        # Now do the actual thresholding
        self.prediction_tensor[self.prediction_tensor >= threshold] = 1
        self.prediction_tensor[self.prediction_tensor <= threshold] = 0
        threshold = round(threshold.item(), 3)
        accuracy = round(accuracy.item(), 3)
        return self.prediction_tensor, threshold, accuracy
    

    def fixedThreshold(self):
        """
        Apply a fixed threshold to intensity volume.
        """
        # Do a fixed threshold
        print("\nApplying a fixed threshold...")
        self.prediction_tensor[self.prediction_tensor >= self.prediction_tensor] = 1
        self.prediction_tensor[self.prediction_tensor <= self.prediction_tensor] = 0
        if self.compute_accuracy == True:
            accuracy = dice(self.prediction_tensor, self.ground_truth_tensor, multiclass=False)
            accuracy = round(accuracy.item(), 3)
        elif self.compute_accuracy == False:
            accuracy = None
        else:
            print("Look, do you want me to compute the accuracy or not!")
            exit(0)
        return self.prediction_tensor, self.threshold, accuracy

#Thresholding()

def thresholding(
    prediction_tensor:torch.Tensor,
    ground_truth_tensor=None,
    threshold:bool=True,
    auto_threshold:bool=True,
    fixed_threshold:float=0.5,
    compute_accuracy:bool=True
    ) -> tuple:
    
    auto_threshold_settings = {
        "start": 0.05,
        "stop": 0.95,
        "step": 0.05,  
    }

    #out_filename = f"prediction_stepsz{step_size}"

    if threshold == True:
        if auto_threshold == True:
            # Decide if we can even do threshold
            if ground_truth_tensor is None:
                # Can't threshold because there was no gt tensor
                print("\nCan't threshold! You didn't give me a ground truth tensor!")
            elif isinstance(ground_truth_tensor, torch.Tensor):
                # All good. Go on to auto thresholding
                print("\nAuto thresholding...")
                threshold_lst = np.arange(
                    auto_threshold_settings["start"],
                    auto_threshold_settings['stop'],
                    auto_threshold_settings['step']
                    )
                accuracy_lst = []
                for thresh in threshold_lst:
                    temp = prediction_tensor.clone()
                    temp[temp >= thresh] = 1
                    temp[temp <= thresh] = 0
                    accuracy = dice(temp, ground_truth_tensor, multiclass=False)
                    accuracy_lst.append(accuracy)
                max_index = accuracy_lst.index(max(accuracy_lst))
                threshold, accuracy = threshold_lst[max_index], accuracy_lst[max_index]
                # Now do the actual thresholding
                prediction_tensor[prediction_tensor >= threshold] = 1
                prediction_tensor[prediction_tensor <= threshold] = 0

                threshold = round(threshold.item(), 3)
                accuracy = round(accuracy.item(), 3)
                return prediction_tensor, threshold, accuracy
            
        elif auto_threshold == False:
            # Do a fixed threshold
            print("\nApplying a fixed threshold...")
            prediction_tensor[prediction_tensor >= fixed_threshold] = 1
            prediction_tensor[prediction_tensor <= fixed_threshold] = 0
            if compute_accuracy == True:
                accuracy = dice(prediction_tensor, ground_truth_tensor, multiclass=False)
                accuracy = round(accuracy.item(), 3)
            else:
                accuracy = None
            return prediction_tensor, fixed_threshold, accuracy
    elif threshold == False:
        # Return a prob map
        print("\nNot thresholding...")
        return prediction_tensor, None, None


def volume_info(tensor, name=None, n:int=150, stats:bool=True, zero=False, unique=False, a:int=None, b:int=None, step:float=None):    
    
    if stats:
        if name is not None:
            print('\n')
            print('#' * 20,f'\n{name} Info')
            print('#' * 20)
        print("\nGeneral Volume Info:")
        print('-' * 20)
        print("  Requres grad:", tensor.requires_grad)
        print("  Shape:", list(tensor.shape))
        print("  dtype:", tensor.dtype)
        
        print("\nVolume Statistics:")
        print('-' * 20)
        print(f"  Mean: {tensor.mean().item():.1e}")
        print(f"  Median: {tensor.median().item():.1e}")
        print(f"  StDev: {tensor.std().item():.1e}")
        print(f"  Range: [{tensor.min().item():.1e}, {tensor.max().item():.1e}]")
        # Quantiles
        print("2nd Percentile:", round(torch.quantile(tensor, 0.02).item(), 3))
        print("25th Percentile:", round(torch.quantile(tensor, 0.25).item(), 3))
        print("75th Percentile:", round(torch.quantile(tensor, 0.75).item(), 3))
        print("98th Percentile:", round(torch.quantile(tensor, 0.98).item(), 3))

    #img = tensor.to('cpu').numpy().squeeze()
    #if a is None:
    #    a = pymath.floor(img.min())
    #if b is None:
    #    b = pymath.ceil(img.max()) + 2
    #if step is None:
    #    step = 1

    #if unique:
    #    print(np.unique(img))
    #else:
    #    pass

    # Histogram
    #frequency, intensity = np.histogram(img, bins=np.arange(a, b, step))
    # Figure
    #plt.figure()
    #f, axarr = plt.subplots(1, 2, figsize=(15, 8), constrained_layout=True)
    #axarr = axarr.flatten()
    #axarr[0].imshow(img[n], cmap='gray')
    #axarr[1].bar(intensity[:-1], frequency, width=0.1)


class PathTools(object):
    """
    Class to handle paths.
    """
    def __init__(self, path:str):
        """
        Parameters
        ----------
        path : str
            Path to deal with. 
        """
        self.path = path


    def destroy(self):
        """
        Delete all files and subdirectories.
        """
        shutil.rmtree(path=self.path, ignore_errors=True)


    def makeDir(self):
        """
        Make new directory. Delete then make again if dir exists.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            try:
                self.destroy()
                os.makedirs(self.path)
            except:
                pass
    

    def patternRemove(self, pattern):
        """
        Remove file in self.path that contains pattern

        Parameters
        ----------
        pattern : str
            Pattern to match to. Examples: {*.nii, *out*, 0001*}
        """
        regex = [
            f"{self.path}/**/{pattern}",
            f"{self.path}/{pattern}"
        ]
        for expression in regex:
            try:
                [os.remove(hit) for hit in glob.glob(
                    expression, recursive=True
                    )]
            except:
                pass

class JsonTools(object):
    """
    Class for handling json files.
    """
    def __init__(self, path):
        """
        Parameters
        ----------
        path : str
            Path to json file.
        """
        self.path = path
    
    def log(self, dict):
        """
        Save Python dictionary as json file.

        Parameters
        ----------
        dict : dict
            Python dictionary to save as json.
        path : str
            Path to new json file to create.
        """
        if not os.path.exists(self.path):
            self.json_object = json.dumps(dict, indent=4)
            file = open(self.path, 'x')
            file.write(self.json_object)
            file.close()

    def read(self):
        f = open(self.path)
        dic = json.load(f)
        return dic
    

class Checkpoint(object):
    """
    Checkpoint handler.
    """
    def __init__(self, checkpoint_dir):
        """
        Parameters
        ----------
        checkpoint_dir : str
            Directory that holds checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_paths = glob.glob(f"{self.checkpoint_dir}/*")

    def best(self):
        """
        Return the first checkpoint that includes 'epoch=' in its filename, or None if no such file exists.

        Returns
        -------
        str or None
            The path to the best checkpoint, or None if no checkpoint matches.
        """
        # Find all files that include 'epoch=' in their filename
        hits = [hit for hit in self.checkpoint_paths if 'epoch=' in hit]
        # Return the first hit if available, otherwise None
        return hits[0] if hits else None

    def last(self):
        """
        Return the last checkpoint file, specifically named 'last.ckpt'.

        Returns
        -------
        str or None
            The path to the last checkpoint, or None if no such file exists.
        """
        hits = [hit for hit in self.checkpoint_paths if 'last.ckpt' in hit]
        return hits[0] if hits else None

    def get(self, type):
        """
        Retrieve a checkpoint based on a specified type ('best' or 'last').

        Parameters
        ----------
        type : str
            The type of checkpoint to retrieve ('best' or 'last').

        Returns
        -------
        str or None
            The path to the requested checkpoint, or None if no suitable checkpoint exists.
        """
        if type == 'best':
            return self.best()
        elif type == 'last':
            return self.last()

def delete_folder(path):
    """
    Deletes a folder and all its contents.

    Args:
    path (str): The path of the folder to be deleted.
    """
    try:
        shutil.rmtree(path)
        print(f"Folder '{path}' has been deleted successfully.")
    except Exception as e:
        print(f"Failed to delete the folder: {e}")


def stretch_contrast(tensor: torch.Tensor, multiplier=1.5) -> torch.Tensor:
    """
    Stretch the contrast of a tensor that has been normalized to the range
    [-1, 1] by adjusting the values based on their standard deviation.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor with values normalized to [-1, 1].

    Returns
    -------
    torch.Tensor
        The contrast-stretched tensor, still within the range [-1, 1].

    """
    mean = torch.mean(tensor)
    std_dev = torch.std(tensor)
    # Adjust the contrast multiplier based on desired stretching
    stretched_tensor = mean + multiplier * (tensor - mean)
    # Clip the values to maintain the range between -1 and 1
    stretched_tensor = torch.clamp(stretched_tensor, min=-1, max=1)
    return stretched_tensor


def get_gaussian_window(tensor_shape=(64, 64, 64), sigma=2):
    # Create a 64^3 patch with values initialized to 1
    patch = torch.ones(tensor_shape, dtype=torch.float32)
    
    # Create Gaussian window along each dimension
    x = torch.linspace(-1, 1, tensor_shape[0])
    y = torch.linspace(-1, 1, tensor_shape[1])
    z = torch.linspace(-1, 1, tensor_shape[2])
    
    x_gaussian = torch.exp(-0.5 * (x / sigma) ** 2)
    y_gaussian = torch.exp(-0.5 * (y / sigma) ** 2)
    z_gaussian = torch.exp(-0.5 * (z / sigma) ** 2)
    
    # Create a 3D Gaussian window
    window = x_gaussian.view(-1, 1, 1) * y_gaussian.view(1, -1, 1) * z_gaussian.view(1, 1, -1)
    
    # Apply the window to the patch
    attenuated_patch = patch * window

    attenuated_patch = torch.stack((attenuated_patch, attenuated_patch, attenuated_patch), dim=0)
    
    return attenuated_patch.cuda()


def normalize(t):
    min_val = torch.min(t)
    max_val = torch.max(t)
    normalized_tensor = 2 * (t - min_val) / (max_val - min_val) - 1
    return normalized_tensor

def scale_tensor(tensor, min_target=0.1, max_target=1):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    # Normalize to [0, 1]
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    # Scale to [min_target, max_target]
    scaled_tensor = min_target + normalized_tensor * (max_target - min_target)
    return scaled_tensor

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
    noise = 1 + torch.randn_like(tensor)
    
    # Scale the noise and add it to the input tensor
    noisy_tensor = tensor + noise_factor * noise
    
    return noisy_tensor


def remove_small_components(tensor, min_size=20):
    # Label the connected components in the tensor
    labeled_tensor, num_features = ndimage.label(tensor)

    # Find the size of each component
    component_sizes = np.bincount(labeled_tensor.ravel())

    # Create an array to keep only components that are larger than min_size
    remove_mask = component_sizes < min_size

    # Zero out small components
    cleaned_tensor = labeled_tensor.copy()
    cleaned_tensor[remove_mask[labeled_tensor]] = 0

    # Convert back to binary tensor
    cleaned_tensor = cleaned_tensor > 0

    return cleaned_tensor


class FullPredict:

    def __init__(self, tensor, predictor, patch_size=128, step_size=64,
                 padding='reflect'):
        self.tensor = tensor
        # self.norm = transforms.Normalize(tensor.mean(), tensor.std())
        self.patch_size = patch_size
        self.step_size = step_size
        self.padding = padding
        self._padit()
        self.predictor = predictor
        self.imprint_tensor = torch.zeros((1, 3, self.tensor.shape[0],
                                           self.tensor.shape[1],
                                           self.tensor.shape[2]),
                                          dtype=torch.float32,
                                          device='cuda')
        print(self.imprint_tensor.shape)
        self.complete_patch_coords = self._get_patch_coords()
        self.num_patches = len(self.complete_patch_coords)

    def predict(self, gaussian_sigma=10):
        sys.stdout.write(f'\rNow Predicting.')
        sys.stdout.flush()
        for i in self.complete_patch_coords:
            in_tensor = self.tensor[i].unsqueeze(0).unsqueeze(0).to(
                torch.float32)
            in_tensor -= in_tensor.min()
            in_tensor /= in_tensor.max()
            in_tensor *= 2
            in_tensor -= 1
            prediction = self.predictor.predict_tensor(in_tensor).to(
                torch.float32) * get_gaussian_window(sigma=gaussian_sigma)
            prediction = F.softmax(prediction, dim=1)
            self.imprint_tensor[..., i[0], i[1], i[2]] += prediction.to(
                torch.float32)
        torch.cuda.empty_cache()
        self._reformat_imprint_tensor()

    def _get_patch_coords(self):
        patch_size = self.patch_size
        step_size = self.step_size
        parent_shape = self.tensor.shape
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

    def _reformat_imprint_tensor(self):
        if self._got_padded:
            self.imprint_tensor = self.imprint_tensor[:, :, 
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size,
                self.patch_size:-self.patch_size
                ]
        redundancy = (self.patch_size ** 3) // (self.step_size ** 3)
        print(f"\n\n{redundancy}x Averaging...")
        self.imprint_tensor /= redundancy

    def _padit(self):
        pad = [self.patch_size] * 6
        self.tensor = torch.nn.functional.pad(self.tensor.unsqueeze(0), pad,
                                              mode=self.padding)
        self.tensor = self.tensor.squeeze()
        self._got_padded = True

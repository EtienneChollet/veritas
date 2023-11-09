__all__ = [
    'Options',
    'Thresholding',
    'PathTools',
    'JsonTools',
    'Checkpoint'
]
# Standard Imports
import os
import glob
import json
import torch
import shutil
import numpy as np
import math as pymath
#from torchmetrics.functional import dice

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
        stem = f"{self.attribute_dict['volume_name']}-prediction"
        stem += f"_stepsz-{self.attribute_dict['step_size']}"
        try:
            stem += f"_{self.attribute_dict['accuracy_name']}-{self.attribute_dict['accuracy_val']}"
        except:
            pass
        stem += '.nii'
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
        #print("2nd Percentile:", round(torch.quantile(tensor, 0.02).item(), 3))
        #print("25th Percentile:", round(torch.quantile(tensor, 0.25).item(), 3))
        #print("75th Percentile:", round(torch.quantile(tensor, 0.75).item(), 3))
        #print("98th Percentile:", round(torch.quantile(tensor, 0.98).item(), 3))

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
        shutil.rmtree(path=self.path)


    def makeDir(self):
        """
        Make new directory. Delete then make again if dir exists.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        else:
            self.destroy()
            os.makedirs(self.path)
    

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
        self.json_object = json.dumps(dict, indent=4)
        file = open(self.path, 'x+')
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
        return [hit for hit in self.checkpoint_paths if 'epoch' in hit][0]
    
    def last(self):
        return [hit for hit in self.checkpoint_paths if 'last' in hit][0]
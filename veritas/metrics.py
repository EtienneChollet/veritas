import torch
import torch.nn as nn

__all__ = [
    'Confusion',
    'GeneralizedEnergyDistance'
]


class Confusion:
    """
    Class to compute confusion components for binary classification tasks.

    Attributes
    ----------
    gt_tensor : torch.Tensor
        Ground truth binary tensor.
    prediction_tensor : torch.Tensor
        Prediction binary tensor.

    Methods
    -------
    get_confusion()
        Returns a tensor describing the confusion matrix components.
    ensure_binary_(tensor)
        Validates if a tensor is binary (0s and 1s only).
    """

    def __init__(self,
                 gt_tensor: torch.Tensor,
                 prediction_tensor: torch.Tensor):
        """
        Initializes the Confusion object with ground truth and prediction
        tensors.

        Parameters
        ----------
        gt_tensor : torch.Tensor
            The ground truth binary tensor.
        prediction_tensor : torch.Tensor
            The prediction binary tensor.

        Raises
        ------
        ValueError
            If either tensor is not binary.
        """
        self.gt_tensor = self.ensure_binary_(gt_tensor)
        self.prediction_tensor = self.ensure_binary_(prediction_tensor)

    def get_confusion(self) -> torch.Tensor:
        """
        Computes the confusion matrix components as a tensor.

        Returns
        -------
        confusion_tensor : torch.Tensor
            Tensor with values representing TP (1's), FP (2's), and FN (3's).
        """
        tp_tensor = (self.gt_tensor 
                     * self.prediction_tensor).to(torch.uint8)
        fp_tensor = (self.gt_tensor 
                     < self.prediction_tensor).to(torch.uint8) * 2
        fn_tensor = (self.gt_tensor 
                     > self.prediction_tensor).to(torch.uint8) * 3
        confusion_tensor = (tp_tensor + fp_tensor + fn_tensor)
        return confusion_tensor


    def ensure_binary_(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Ensures the input tensor is binary.

        Parameters
        ----------
        tensor : torch.Tensor
            Tensor to be validated.

        Raises
        ------
        ValueError
            If the tensor is not strictly binary.

        Returns
        -------
        tensor : torch.Tensor
            The verified binary tensor.
        """
        if not torch.all((tensor == 0) | (tensor == 1)):
            raise ValueError('The tensor is not binary!!')
        else:
            return tensor


class GeneralizedEnergyDistance(nn.Module):
    def __init__(self, num_samples=1000):
        """
        Initialize the module to compute the generalized energy distance using
        PyTorch on GPU.
        
        Parameters:
        - num_samples (int): Number of points to sample from each volume.
        """
        super(GeneralizedEnergyDistance, self).__init__()
        self.num_samples = num_samples

    def forward(self, seg1, seg2):
        """
        Compute the generalized energy distance between two segmentation maps.
        
        Parameters:
        - seg1 (torch.Tensor): The first segmentation map (3D array).
        - seg2 (torch.Tensor): The second segmentation map (3D array).
        
        Returns:
        - float: The generalized energy distance.
        """
        sample1 = self.sample_volume(seg1, self.num_samples)
        sample2 = self.sample_volume(seg2, self.num_samples)
        
        # Calculate pairwise distances using broadcasting
        dist_x_y = torch.abs(sample1[:, None] - sample2[None, :])
        dist_x_x = torch.abs(sample1[:, None] - sample1[None, :])
        dist_y_y = torch.abs(sample2[:, None] - sample2[None, :])
        
        # Calculate expected values
        exy = dist_x_y.mean()
        exx = dist_x_x.mean()
        eyy = dist_y_y.mean()
        
        # Compute generalized energy distance
        energy_distance = 2 * exy - exx - eyy
        return energy_distance

    @staticmethod
    def sample_volume(volume, num_samples):
        """
        Randomly sample points from a 3D volume using PyTorch.
        
        Parameters:
        - volume (torch.Tensor): The 3D volume to sample from.
        - num_samples (int): Number of points to sample.
        
        Returns:
        - torch.Tensor: Sampled points as a 1D array.
        """
        flat_volume = volume.view(-1)
        indices = torch.randint(0,
                                flat_volume.size(0),
                                (num_samples,),
                                device=volume.device
                            )
        return flat_volume[indices]
    


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MedianFilter2D(nn.Module):
    """
    A module to apply a median filter to 2D images.

    This module extends `torch.nn.Module` and can be integrated into neural
    networks
    to perform median filtering as part of the network's forward pass.

    Parameters
    ----------
    kernel_size : int
        Size of the neighborhood from which the median is computed. Must be an
        odd number.

    Raises
    ------
    ValueError
        If kernel_size is not odd.
    """
    def __init__(self, kernel_size=3):
        super(MedianFilter2D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        self.kernel_size = kernel_size

    def forward(self, image):
        """
        Apply the median filter to an input image as part of the model's
        forward pass.

        Parameters
        ----------
        image : torch.Tensor
            Input 2D tensor representing the image. Should be of shape
            [height, width] or [batch_size, channels, height, width] for
            batches of images.

        Returns
        -------
        torch.Tensor
            Output 2D tensor after applying the median filter. Shape matches
            the input shape.
        """
        # Ensure image tensor is at least 4D (batch_size, channels, height,
        # width)
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        pad_size = self.kernel_size // 2
        padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size),
                             mode='reflect')

        # Efficiently compute patches and their medians
        patches = padded_image.unfold(
            2, self.kernel_size, 1).unfold(
                3, self.kernel_size, 1)
        patches = patches.contiguous().view(
            *image.shape[:2], *patches.shape[-4:-2], -1)
        medians = patches.median(dim=-1).values

        return medians


class MedianFilter3D(MedianFilter2D):
    """
    A module to apply a median filter to 3D volumes.

    This class extends `MedianFilter2D` to handle 3D data, applying a median
    filter across the smallest dimension of the input tensor.

    Inherits
    --------
    MedianFilter2D : An implementation of 2D median filtering as a PyTorch
    module.

    Parameters
    ----------
    kernel_size : int
        Size of the neighborhood from which the median is computed. Must be an
        odd number.
    """
    def __init__(self, kernel_size=3):
        super().__init__(kernel_size=kernel_size)

    def forward(self, volume):
        """
        Apply the median filter to an input 3D volume as part of the model's
        forward pass.

        Parameters
        ----------
        volume : torch.Tensor
            Input 3D tensor representing the volume. Should be of shape
            [depth, height, width]
            or [batch_size, channels, depth, height, width] for batches of
            volumes.

        Returns
        -------
        torch.Tensor
            Output 3D tensor after applying the median filter. Shape matches
            the input shape.
        """
        # Handle different dimensions of input
        if volume.dim() == 3:
            # Add batch and channel dimension
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.dim() == 4:
            volume = volume.unsqueeze(1)  # Add channel dimension

        smallest_dim = volume.size().index(min(volume.size()))
        slices = volume.unbind(dim=smallest_dim)

        modified_slices = [self.median_filter(slice) for slice in slices]

        return torch.stack(modified_slices, dim=smallest_dim)

    def median_filter(self, slice):
        """
        Helper function to apply 2D median filtering to each slice of the
        volume.

        Parameters
        ----------
        slice : torch.Tensor
            A 2D slice from the 3D volume.

        Returns
        -------
        torch.Tensor
            The 2D slice after applying the median filter.
        """
        # Use the forward method from MedianFilter2D
        return super().forward(slice)


class SobelFilter2D:
    def __init__(self, kernel_size=3):
        """
        Initialize the SobelFilter2D with a specific kernel size.

        Parameters
        ----------
        kernel_size : int
            Size of the Sobel kernels. Must be odd and at least 3.
        """
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd and at least 3")
        self.kernel_size = kernel_size
        self.sobel_x = self.create_sobel_kernel(kernel_size, 0)
        self.sobel_y = self.create_sobel_kernel(kernel_size, 1)

    def create_sobel_kernel(self, size, axis):
        """
        Create a Sobel kernel for a given size and axis.
        """
        range = torch.arange(size) - size // 2
        grid_x, grid_y = torch.meshgrid(range, range, indexing="ij")

        if axis == 0:  # Sobel x-axis kernel
            kernel = grid_x * torch.exp(-(grid_x**2 + grid_y**2) / (size**2))
        elif axis == 1:  # Sobel y-axis kernel
            kernel = grid_y * torch.exp(-(grid_x**2 + grid_y**2) / (size**2))

        kernel /= torch.sum(torch.abs(kernel))
        return kernel.view(1, 1, size, size)

    def apply_filter(self, image):
        """
        Apply the Sobel filter to the provided 2D image to highlight edges.

        Parameters
        ----------
        image : torch.Tensor
            Input 2D tensor representing the image.

        Returns
        -------
        torch.Tensor
            A 2D tensor of the gradient magnitude.
        """
        # Establishing backend
        backend = dict(device=image.device, dtype=image.dtype)

        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)

        edge_x = F.conv2d(image, self.sobel_x.to(**backend),
                          padding=self.kernel_size//2)
        edge_y = F.conv2d(image, self.sobel_y.to(**backend),
                          padding=self.kernel_size//2)

        magnitude = torch.sqrt(edge_x**2 + edge_y**2).squeeze(0).squeeze(0)
        return magnitude


class SobelFilter3D(SobelFilter2D):
    """
    Class to apply a Sobel filter to a 3D volume by extending SobelFilter2D.
    """
    def apply_filter(self, volume):
        """
        Apply the Sobel filter to the provided 3D volume.

        Parameters
        ----------
        volume : torch.Tensor
            Input 3D tensor representing the volume.

        Returns
        -------
        torch.Tensor
            A 3D tensor with the gradient magnitude computed for each slice.
        """
        smallest_dim = volume.size().index(min(volume.size()))
        slices = volume.unbind(dim=smallest_dim)

        modified_slices = []
        for slice in slices:
            filtered_slice = super().apply_filter(slice)
            modified_slices.append(filtered_slice)

        return torch.stack(modified_slices, dim=smallest_dim)


class VarianceFilter2D(nn.Module):
    """
    A PyTorch module to apply a variance filter to 2D images using a specified
    window size.
    """

    def __init__(self, window_size):
        """
        Initializes the VarianceFilter2D with a specific window size.

        Parameters
        ----------
        window_size : int
            Size of the window to compute the variance. Must be an odd number
            to maintain symmetry.
        """
        super().__init__()
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        self.window_size = window_size
        self.pad = window_size // 2

    def forward(self, image):
        """
        Apply variance filter to the provided 2D image.

        Parameters
        ----------
        image : torch.Tensor
            Input 2D tensor representing the image.

        Returns
        -------
        torch.Tensor
            A 2D tensor of local variance values.
        """
        if image.dim() == 2:
            # Add batch and channel dimensions if not present
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)  # Add channel dimension

        # Padding for same output size
        image = F.pad(image, (self.pad, self.pad, self.pad, self.pad),
                      mode='reflect')
        # Calculate local mean and local mean square
        kernel = torch.ones((1, 1, self.window_size, self.window_size),
                            device=image.device,
                            dtype=torch.float32) / (self.window_size**2)
        local_mean = F.conv2d(image, kernel)
        local_mean_sq = F.conv2d(image ** 2, kernel)
        local_variance = local_mean_sq - local_mean ** 2

        return local_variance


class VarianceFilter3D(VarianceFilter2D):
    """
    Class to apply a variance filter to a 3D image by extending
    VarianceFilter2D.
    """

    def apply_filter(self, volume):
        """
        Apply the variance filter to the provided 3D volume.

        Parameters
        ----------
        volume : torch.Tensor
            Input 3D tensor representing the volume.

        Returns
        -------
        torch.Tensor
            A 3D tensor of local variance values, with variance computed along
            the smallest dimension.
        """
        smallest_dim = volume.size().index(min(volume.size()))
        slices = volume.to(torch.float32).unbind(dim=smallest_dim)

        modified_slices = []
        # Process each slice and print progress every 10 slices
        for i, slice in enumerate(slices):
            modified_slices.append(super().apply_filter(slice))
            # if i % 10 == 0:
            #    print(i)
        return torch.stack(modified_slices, dim=smallest_dim)


class GaborFilter:
    def __init__(self, sizes: list = [31], sigmas: list = [4],
                 thetas: list = [1], lambdas: list = [10], psis: list = [0],
                 gammas: list = [0.5]):
        """
        Initialize the Gabor filter with specified parameters for multiple
        sizes and wavelengths.
        Generates multiple kernels based on combinations of sizes and
        wavelengths.

        Parameters
        ----------
        sizes : list of int
            List of sizes for the filter kernels.
        sigmas : list of float
            List of sigma values for the Gaussian envelope.
        thetas : list of float
            List of orientations for the Gabor filters.
        lambdas : list of float
            List of wavelengths for the sinusoidal factors.
        psis : list of float
            List of phase offsets.
        gammas : list of float
            List of spatial aspect ratios.
        """
        self.kernels = []
        for size, lambda_ in zip(sizes, lambdas):
            for sigma, theta, psi, gamma in zip(sigmas, thetas, psis, gammas):
                kernel = self.generate_gabor_kernel(
                    size, sigma, theta, lambda_, psi, gamma)
                self.kernels.append((size, kernel))

    def generate_gabor_kernel(self, size, sigma, theta, lambda_, psi, gamma):
        """
        Generate a Gabor filter kernel using the specified parameters.
        """
        x = torch.linspace(-size // 2, size // 2, size)
        y = torch.linspace(-size // 2, size // 2, size)
        x, y = torch.meshgrid(x, y)

        theta = torch.tensor(theta)

        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        gb = torch.exp(-0.5 * (
            x_theta**2 + gamma**2 * y_theta**2) / sigma**2) * torch.cos(
                2 * np.pi * x_theta / lambda_ + psi)
        return gb[0]

    def apply_filter(self, image):
        """
        Apply the Gabor filters to an image using the precomputed kernels.

        Parameters
        ----------
        image : torch.Tensor
            The input image tensor of shape [1, 1, H, W] or [H, W].

        Returns
        -------
        list of torch.Tensor
            List of filtered images, one for each kernel.
        """
        results = []
        if len(image.shape) == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif len(image.shape) == 3:
            image = image.unsqueeze(0)

        image = image.float()

        for size, kernel in self.kernels:
            kernel = kernel.to(image.device).float().unsqueeze(0).unsqueeze(0)
            filtered_image = F.conv2d(image, kernel, padding=size//2)
            results.append(filtered_image.squeeze())

        return results


class GaussianFilter:
    def __init__(self, kernel_size=3, sigma=1.0):
        """
        Initialize the GaussianFilter object with a specified kernel size and
        sigma.

        Parameters:
        - kernel_size (int): The size of the square kernel. Should be odd.
        - sigma (float): The standard deviation of the Gaussian kernel.
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd.")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self.create_gaussian_kernel(kernel_size, sigma)

    def create_gaussian_kernel(self, kernel_size, sigma):
        """
        Create a Gaussian kernel using the specified size and standard
        deviation.

        Parameters:
        - kernel_size (int): The size of the kernel (must be odd to have a
        center).
        - sigma (float): The standard deviation of the Gaussian.

        Returns:
        - torch.Tensor: A 2D tensor representing the Gaussian kernel.
        """
        # Create a grid of (x,y) coordinates
        x = torch.arange(kernel_size) - (kernel_size - 1) / 2
        y = x.view(-1, 1)
        x2 = x**2
        y2 = y**2

        # Exponential part of the Gaussian
        g = torch.exp(-(x2 + y2) / (2 * sigma**2))

        # Normalize to make the sum of all elements equal to 1
        g /= g.sum()

        return g.view(1, 1, kernel_size, kernel_size)

    def apply_filter(self, input_image):
        """
        Apply the Gaussian filter to an input image.

        Parameters:
        - input_image (torch.Tensor): A tensor of shape (H, W) or (1, H, W) or
        (N, C, H, W)

        Returns:
        - torch.Tensor: The filtered image.
        """
        if len(input_image.shape) == 2:
            # Add batch and channel dimensions if not present
            input_image = input_image.unsqueeze(0).unsqueeze(0)

        # Ensure input is floating point for the convolution
        input_image = input_image.float()

        # Apply the Gaussian kernel
        filtered_image = F.conv2d(input_image, self.kernel,
                                  padding=self.kernel_size//2)

        return filtered_image


class ErosionFilter:
    def __init__(self, kernel_size=3, padding=1):
        """
        Initialize the ErosionFilter object with a specified kernel size.

        Parameters:
        - kernel_size (int): The size of the square kernel for erosion.
        - padding (int): The padding applied to the image before the operation.
        """
        self.kernel_size = kernel_size
        self.padding = padding
        # Create a kernel with all ones
        self.kernel = torch.ones((1, 1, kernel_size, kernel_size),
                                 dtype=torch.float32)

    def apply_filter(self, input_image):
        """
        Apply an erosion filter to the input_image.

        Parameters:
        - input_image (torch.Tensor): A binary torch.Tensor of shape (H, W) or
        (1, H, W).

        Returns:
        - torch.Tensor: The eroded image as a torch.Tensor of the same shape.
        """
        if len(input_image.shape) == 2:
            # Add batch and channel dimensions if not present
            input_image = input_image.unsqueeze(0).unsqueeze(0)

        # Convert boolean image to float
        input_image = input_image.float()

        # Perform a convolution with the all-ones kernel
        result = F.conv2d(input_image, self.kernel, padding=self.padding)

        # A pixel in the erosion result is 1 only if the sum of the
        # convolution is equal to the area of the kernel
        kernel_area = self.kernel_size ** 2
        # Convert back to binary by checking for full kernel matches
        eroded_image = (result == kernel_area).float()

        return eroded_image.squeeze()  # Remove extra dimensions if any


# This should probably go in a morphology package
def fill_holes(binary_image, kernel_size=3, iterations=10):
    """
    Fill holes in a binary image using morphological dilation.

    Args:
    - binary_image (torch.Tensor): Binary image tensor where holes are
    represented by 0s.
    - kernel_size (int): The size of the square kernel for dilation.
    - iterations (int): Maximum number of iterations to apply dilation to fill
    holes.

    Returns:
    - torch.Tensor: Binary image with holes filled.
    """
    # Create a structuring element (kernel)
    struct_elem = torch.ones((1, 1, kernel_size, kernel_size),
                             dtype=torch.float32)

    # Start with the original image
    # Shape [1, 1, H, W]
    current_image = binary_image.unsqueeze(0).unsqueeze(0)

    # Iterate to fill holes
    for _ in range(iterations):
        # Apply dilation
        dilated_image = F.conv2d(current_image, struct_elem.cuda(),
                                 padding=kernel_size//2, stride=1)

        # Threshold the dilated image to get binary values back
        dilated_image = (dilated_image >= 1).float()

        # Combine with the original image to preserve boundaries
        new_image = dilated_image + current_image
        new_image = (new_image > 0).float()

        # If no change, stop iterating
        if torch.equal(new_image, current_image):
            break

        current_image = new_image

    return current_image.squeeze(0).squeeze(0)


class MinimumFilter2D(nn.Module):
    """
    A module to apply a minimum filter to 2D images.

    This module extends `torch.nn.Module` and can be integrated into neural
    networks to perform minimum filtering as part of the network's forward
    pass.

    Parameters
    ----------
    kernel_size : int
        Size of the neighborhood from which the minimum is computed. Must be
        an odd number.

    Raises
    ------
    ValueError
        If kernel_size is not odd.
    """
    def __init__(self, kernel_size=3):
        super(MinimumFilter2D, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        self.kernel_size = kernel_size

    def forward(self, image):
        """
        Apply the minimum filter to an input image as part of the model's
        forward pass.

        Parameters
        ----------
        image : torch.Tensor
            Input 2D tensor representing the image. Should be of shape
            [height, width] or [batch_size, channels, height, width] for
            batches of images.

        Returns
        -------
        torch.Tensor
            Output 2D tensor after applying the minimum filter. Shape matches
            the input shape.
        """
        # Ensure image tensor is at least 4D (batch_size, channels, height,
        # width)
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        pad_size = self.kernel_size // 2
        padded_image = F.pad(image, (pad_size, pad_size, pad_size, pad_size),
                             mode='reflect')

        # Efficiently compute patches and their minimums
        patches = padded_image.unfold(
            2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        patches = patches.contiguous().view(
            *image.shape[:2], *patches.shape[-4:-2], -1)
        minimums = patches.min(dim=-1).values

        return minimums


class MinimumFilter3D(MinimumFilter2D):
    """
    A module to apply a minimum filter to 3D volumes.

    This class extends `MinimumFilter2D` to handle 3D data, applying a minimum
    filter across the smallest dimension of the input tensor.

    Inherits
    --------
    MinimumFilter2D : An implementation of 2D minimum filtering as a PyTorch
    module.

    Parameters
    ----------
    kernel_size : int
        Size of the neighborhood from which the minimum is computed. Must be
        an odd number.
    """
    def __init__(self, kernel_size=3):
        super().__init__(kernel_size=kernel_size)

    def forward(self, volume):
        """
        Apply the minimum filter to an input 3D volume as part of the model's
        forward pass.

        Parameters
        ----------
        volume : torch.Tensor
            Input 3D tensor representing the volume. Should be of shape
            [depth, height, width] or [batch_size, channels, depth, height,
            width] for batches of volumes.

        Returns
        -------
        torch.Tensor
            Output 3D tensor after applying the minimum filter. Shape matches
            the input shape.
        """
        # Handle different dimensions of input
        if volume.dim() == 3:
            # Add batch and channel dimension
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif volume.dim() == 4:
            volume = volume.unsqueeze(1)  # Add channel dimension

        smallest_dim = volume.size().index(min(volume.size()))
        slices = volume.unbind(dim=smallest_dim)

        modified_slices = [self.minimum_filter(slice) for slice in slices]

        return torch.stack(modified_slices, dim=smallest_dim).squeeze()

    def minimum_filter(self, slice):
        """
        Helper function to apply 2D minimum filtering to each slice of the
        volume.

        Parameters
        ----------
        slice : torch.Tensor
            A 2D slice from the 3D volume.

        Returns
        -------
        torch.Tensor
            The 2D slice after applying the minimum filter.
        """
        # Use the forward method from MinimumFilter2D
        return super().forward(slice)

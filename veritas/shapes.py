import torch
import torch.nn as nn
import numpy as np

from cornucopia.cornucopia.random import Uniform, Fixed, RandInt


class RectangleGenerator(nn.Module):
    def __init__(self, device='cuda', channels=1, L=64, W=64, H=64, num_rectangles=50):
        super(RectangleGenerator, self).__init__()
        self.device = device
        self.channels = channels
        self.L = L
        self.W = W
        self.H = H
        self.num_rectangles = num_rectangles
    
    def generate_rectangle(self, size_L, size_W, size_H, fill_value):
        """
        Generate a 3D rectangle (cuboid) filled with the specified value.
        """
        rectangle = torch.full(
            (size_L, size_W, size_H),
            fill_value,
            dtype=torch.float32,
            device=self.device)
        return rectangle
    
    def inject_rectangle(self, tensor, rectangle, channel):
        """
        Inject the rectangle into the given tensor at a random position.
        """
        L, W, H = tensor.shape[1:]
        size_L, size_W, size_H = rectangle.shape
        
        # Random position within the tensor
        start_L = torch.randint(0, L - size_L + 1, (1,)).item()
        start_W = torch.randint(0, W - size_W + 1, (1,)).item()
        start_H = torch.randint(0, H - size_H + 1, (1,)).item()
        
        # Inject the rectangle into the tensor
        tensor[channel, start_L:start_L+size_L, start_W:start_W+size_W, start_H:start_H+size_H] = rectangle
    
    def forward(self, num_classes=5, max_size_L=64,
                max_size_W=64, max_size_H=64):
        tensor = torch.zeros(
            (self.channels, self.L, self.W, self.H), dtype=torch.float32)
        for _ in range(self.num_rectangles):
            size_L = torch.randint(1, max_size_L + 1, (1,)).item()
            size_W = torch.randint(1, max_size_W + 1, (1,)).item()
            size_H = torch.randint(1, max_size_H + 1, (1,)).item()
            fill_value = torch.rand(1).item()*2
            rectangle = self.generate_rectangle(
                size_L, size_W, size_H, fill_value)
            channel = torch.randint(0, self.channels, (1,)).item()
            self.inject_rectangle(tensor, rectangle, channel)
        return tensor


class SphereSampler(nn.Module):
    """
    A PyTorch module for randomly sampling non-overlapping spheres with
    irregular edges in a 3D volume using GPU acceleration.

    Attributes
    ----------
    volume_size : tuple
        The dimensions of the volume (depth, height, width).
    num_spheres : int
        Number of spheres to generate.
    radius_range : tuple
        Minimum and maximum radius of the spheres.
    label_range : tuple
        Minimum and maximum label values.
    """

    def __init__(self, volume_size, num_spheres, radius_range, label_range):
        super(SphereSampler, self).__init__()
        self.volume_size = volume_size
        self.num_spheres = num_spheres
        self.radius_range = radius_range
        self.label_range = label_range
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self):
        """
        Generates a volume with randomly sampled spheres with irregular edges
        and no overlapping spheres.

        Returns
        -------
        torch.Tensor
            The 3D volume tensor with sampled spheres.
        """
        volume = torch.zeros(
            self.volume_size,
            dtype=torch.int32).to(self.device)  # Use int32 for volume
        depth, height, width = self.volume_size
        centers = []
        radii = []

        while len(centers) < self.num_spheres:
            radius = torch.randint(
                self.radius_range[0],
                self.radius_range[1] + 1,
                (1,),
                device=self.device).item()
            center = (
                torch.randint(
                    radius, depth - radius, (1,), device=self.device).item(),
                torch.randint(
                    radius, height - radius, (1,), device=self.device).item(),
                torch.randint(
                    radius, width - radius, (1,), device=self.device).item()
            )

            # Check for overlaps
            overlap = False
            for c, r in zip(centers, radii):
                distance = torch.sqrt(
                    torch.tensor((center[0] - c[0]) ** 2 +
                                 (center[1] - c[1]) ** 2 +
                                 (center[2] - c[2]) ** 2, device=self.device))
                if distance < r + radius:
                    overlap = True
                    break
            if not overlap:
                centers.append(center)
                radii.append(radius)

        for center, radius in zip(centers, radii):
            # Generate irregular edges by adding random noise to the radius
            z, y, x = torch.meshgrid(
                torch.arange(depth, device=self.device) - center[0],
                torch.arange(height, device=self.device) - center[1],
                torch.arange(width, device=self.device) - center[2],
                indexing='ij')
            if radius < 3:
                # To prevent really small spheres from becoming jibberish
                noise = torch.normal(
                    0, 0.001, size=z.size(), device=self.device)
            else:
                noise = torch.normal(0, 0.5, size=z.size(), device=self.device)
            mask = (z ** 2 + y ** 2 + x ** 2) <= (radius + noise) ** 2
            label_value = Uniform(*self.label_range)()
            volume = volume.to(torch.float32)
            volume[mask] = label_value

        return volume

# Example usage:
# sampler = SphereSampler((64, 64, 64), 5, (10, 20), (1, 10)).to('cuda')
# volume = sampler()
# print(volume.sum())  # Sum of labels to verify non-overlapping irregular
# spheres

# Ensure data is moved to CPU and converted to numpy for plotting
# if volume.is_cuda:
#    volume = volume.cpu()
# plt.imshow(volume[30].numpy())  # Visualize a specific slice of the volume
# plt.show()
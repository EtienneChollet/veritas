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


class EllipsoidSampler(nn.Module):
    """
    A PyTorch module for randomly sampling non-overlapping ellipsoids with
    irregular edges in a 3D volume using GPU acceleration.

    Attributes
    ----------
    volume_size : tuple
        The dimensions of the volume (depth, height, width).
    num_ellipsoids : int
        Number of ellipsoids to generate.
    axes_range : tuple
        Minimum and maximum lengths for the semi-axes of the ellipsoids.
    label_range : tuple
        Minimum and maximum label values.
    """

    def __init__(self, volume_size, num_ellipsoids, axes_range, label_range):
        super(EllipsoidSampler, self).__init__()
        self.volume_size = volume_size
        self.num_ellipsoids = num_ellipsoids
        self.axes_range = axes_range
        self.label_range = label_range
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self):
        """
        Generates a volume with randomly sampled ellipsoids with irregular
        edges and no overlapping ellipsoids.

        Returns
        -------
        torch.Tensor
            The 3D volume tensor with sampled ellipsoids.
        """
        volume = torch.zeros(
            self.volume_size,
            dtype=torch.int32).to(self.device)  # Use int32 for volume
        depth, height, width = self.volume_size
        centers = []
        axes_list = []

        while len(centers) < self.num_ellipsoids:
            axes = torch.randint(
                self.axes_range[0],
                self.axes_range[1] + 1,
                (3,),
                device=self.device)
            center = (
                torch.randint(
                    axes[0], depth - axes[0], (1,), device=self.device).item(),
                torch.randint(
                    axes[1], height - axes[1], (1,), device=self.device).item(),
                torch.randint(
                    axes[2], width - axes[2], (1,), device=self.device).item()
            )

            # Check for overlaps
            overlap = False
            for c, a in zip(centers, axes_list):
                distance = torch.sqrt(
                    torch.tensor((center[0] - c[0]) ** 2 +
                                 (center[1] - c[1]) ** 2 +
                                 (center[2] - c[2]) ** 2, device=self.device))
                if distance < max(a) + max(axes):
                    overlap = True
                    break
            if not overlap:
                centers.append(center)
                axes_list.append(axes)

        for center, axes in zip(centers, axes_list):
            # Generate irregular edges by adding random noise to the axes
            z, y, x = torch.meshgrid(
                torch.arange(depth, device=self.device) - center[0],
                torch.arange(height, device=self.device) - center[1],
                torch.arange(width, device=self.device) - center[2],
                indexing='ij')
            noise = torch.normal(0, 0.5, size=z.size(), device=self.device)
            mask = ((z / (axes[0] + noise)) ** 2 +
                    (y / (axes[1] + noise)) ** 2 +
                    (x / (axes[2] + noise)) ** 2) <= 1
            label_value = Uniform(*self.label_range).sample().item()
            volume = volume.to(torch.float32)
            volume[mask] = label_value

        return volume

# Example usage
# if __name__ == "__main__":
#    volume_size = (100, 100, 100)
#    num_ellipsoids = 10
#    axes_range = (5, 15)
#    label_range = (1, 255)#

#    ellipsoid_sampler = EllipsoidSampler(volume_size, num_ellipsoids,
#    axes_range, label_range)
#    volume = ellipsoid_sampler()
#    print(volume.shape)


class AngularBlobbySampler(nn.Module):
    """
    A PyTorch module for randomly sampling non-overlapping blobby shapes with
    angular edges in a 3D volume using GPU acceleration.

    Attributes
    ----------
    volume_size : tuple
        The dimensions of the volume (depth, height, width).
    num_blobs : int
        Number of blobs to generate.
    axes_range : tuple
        Minimum and maximum lengths for the semi-axes of the blobs.
    label_range : tuple
        Minimum and maximum label values.
    """

    def __init__(self, volume_size, num_blobs, axes_range, label_range):
        super(AngularBlobbySampler, self).__init__()
        self.volume_size = volume_size
        self.num_blobs = num_blobs
        self.axes_range = axes_range
        self.label_range = label_range
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def forward(self):
        """
        Generates a volume with randomly sampled angular blobby shapes with
        irregular edges and no overlapping blobs.

        Returns
        -------
        torch.Tensor
            The 3D volume tensor with sampled blobs.
        """
        volume = torch.zeros(
            self.volume_size,
            dtype=torch.int32).to(self.device)  # Use int32 for volume
        depth, height, width = self.volume_size
        centers = []
        axes_list = []

        while len(centers) < self.num_blobs:
            axes = torch.randint(
                self.axes_range[0],
                self.axes_range[1] + 1,
                (3,),
                device=self.device)
            center = (
                torch.randint(
                    axes[0], depth - axes[0], (1,), device=self.device).item(),
                torch.randint(
                    axes[1], height - axes[1], (1,), device=self.device).item(),
                torch.randint(
                    axes[2], width - axes[2], (1,), device=self.device).item()
            )

            # Check for overlaps
            overlap = False
            for c, a in zip(centers, axes_list):
                distance = torch.sqrt(
                    torch.tensor((center[0] - c[0]) ** 2 +
                                 (center[1] - c[1]) ** 2 +
                                 (center[2] - c[2]) ** 2, device=self.device))
                if distance < max(a) + max(axes):
                    overlap = True
                    break
            if not overlap:
                centers.append(center)
                axes_list.append(axes)

        for center, axes in zip(centers, axes_list):
            # Generate angular blobby edges by adding random noise to the axes
            z, y, x = torch.meshgrid(
                torch.arange(depth, device=self.device) - center[0],
                torch.arange(height, device=self.device) - center[1],
                torch.arange(width, device=self.device) - center[2],
                indexing='ij')

            # Use a Perlin-like noise function to create more blobby shapes
            # with angular edges
            noise_scale = 0.25  # Control the "blobbiness"
            noise = torch.normal(0, noise_scale, size=z.size(),
                                 device=self.device)

            # Create angular edges by adding a term that introduces sharpness
            # [0.75, 3]
            sharpness = Uniform(0.75, 3)()
            mask = ((torch.abs(z / (axes[0] + noise)) ** sharpness +
                     torch.abs(y / (axes[1] + noise)) ** sharpness +
                     torch.abs(x / (axes[2] + noise)) ** sharpness) <= 1)

            label_value = Uniform(*self.label_range)()
            volume = volume.to(torch.float32)
            volume[mask] = label_value

        return volume

# Example usage
# if __name__ == "__main__":
#    volume_size = (100, 100, 100)
#    num_blobs = 20  # Increase the number of blobs for higher density
#    axes_range = (5, 15)
#    label_range = (1, 255)

#    blobby_sampler = AngularBlobbySampler(volume_size, num_blobs, axes_range, label_range)
#    volume = blobby_sampler()
#    print(volume.shape)


class BlobSampler(nn.Module):
    def __init__(self, axis_length_range=[3, 6], intensity_range=[1.1, 5],
                 max_blobs=25, max_sharpness=3, device='cuda', shape=64):
        super(BlobSampler, self).__init__()
        self.axis_length_range = axis_length_range
        self.intensity_range = intensity_range
        self.device = device
        self.shape = [shape, shape, shape]
        self.depth, self.height, self.width = self.shape
        self.n_blobs = RandInt(1, max_blobs)()
        self.max_sharpness = max_sharpness
        self.imprint_tensor = torch.zeros(self.shape, dtype=torch.float32,
                                          device=self.device)

    def sample_axis_lengths(self):
        """
        Define blob shape/size by sampling the length for each axis.

        Returns
        -------
        axis_lengths : torch.Tensor([int, int, int])
            List of lengths for each of the blob's axes.
        """
        axis_lengths = torch.randint(
            self.axis_length_range[0],
            self.axis_length_range[1] + 1,
            (3,),
            device=self.device)
        return axis_lengths

    def sample_centroid_coords(self, axis_lengths):
        """
        Define centroid coordinates within parent tensor.

        Parameters
        ----------
        axis_lengths : list
            List of lengths for each of the blob's axes.

        Returns
        -------
        centroid_coords : list
            List of coordinates for centroid.
        """
        centroid_coords = (
            torch.randint(
                axis_lengths[0],
                self.depth - axis_lengths[0],
                (1,), device=self.device).item(),
            torch.randint(
                axis_lengths[1],
                self.height - axis_lengths[1],
                (1,), device=self.device).item(),
            torch.randint(
                axis_lengths[2],
                self.width - axis_lengths[2],
                (1,), device=self.device).item()
            )
        return centroid_coords


    def check_overlap(self, axis_lengths, centroid_coords, axes_list=[],
                      centers=[]):
        """
        Check if incoming (querying) blob will overlap with existing ones.
        Parameters
        ----------
        axis_lengths : list
            Axis lengths for incoming blob.
        centroid_coords : list
            Coordinates for centroid of incoming blob.
        axes_list : list, optional
            Preexisting axes, by default []
        centers : list, optional
            Preexisting centroid coordinates, by default []
        Returns
        -------
        overlap_exists : bool
            Bool if overlap exists or not
        """
        overlap_exists = False
        # Iterate through all existing centroid-coordinate and axes-length 
        # pairs.
        for c, a in zip(centers, axes_list):
            # Calculate distance to the centroid coordinate of querying blob.
            distance = torch.sqrt(
                torch.tensor((centroid_coords[0] - c[0]) ** 2
                             + (centroid_coords[1] - c[1]) ** 2
                             + (centroid_coords[2] - c[2]) ** 2,
                             device=self.device)
            )
            # Overlap defined as blobs whose largest dimensions overlap/touch.
            if distance < max(a) + max(axis_lengths):
                overlap_exists = True
                break  # Exit loop if any overlap is found
        return overlap_exists

    def sample_nonoverlapping_geometries(self):
        """
        Sample non-overlapping geometries for blobs.

        Returns
        -------
        centers : list
            List of centroid coordinates for each blob.
        axes_list : list
            List of axis lengths for each blob.
        """
        centers = []
        axes_list = []

        while len(centers) < self.n_blobs:
            # Sample incoming blob's axes lengths.
            axis_lengths = self.sample_axis_lengths()
            # Sample incoming blob's centroid coordinates.
            centroid_coords = self.sample_centroid_coords(axis_lengths)
            # Check for overlaps between incoming blob and preexisting ones.
            overlap_exists = self.check_overlap(axis_lengths, centroid_coords,
                                                axes_list, centers)
            # If no overlap exists, add this blob's info to the lists.
            if not overlap_exists:
                centers.append(centroid_coords)
                axes_list.append(axis_lengths.tolist())

        return centers, axes_list

    def _meshgrid_origin_at_centroid(self, center_coords):
        meshgrid = torch.meshgrid(
            torch.arange(self.depth, device=self.device) - center_coords[0],
            torch.arange(self.height, device=self.device) - center_coords[1],
            torch.arange(self.width, device=self.device) - center_coords[2],
            indexing='ij')
        return meshgrid

    def _add_noise_to_axes(self, axes: list, noise_scale: float = 0.5):
        noise = torch.normal(0, noise_scale,
                             size=[self.depth, self.height, self.width],
                             device=self.device)
        axes[0] += noise
        axes[1] += noise
        axes[2] += noise
        return axes

    def _make_shape_prob(self, meshgrid: list, axes: list):
        sharpness = Uniform(0.75, self.max_sharpness)()
        shape_prob = (torch.abs(meshgrid[0] / (axes[0])) ** sharpness +
                      torch.abs(meshgrid[1] / (axes[1])) ** sharpness +
                      torch.abs(meshgrid[2] / (axes[2])) ** sharpness
                      )
        return shape_prob

    def _make_shape_from_prob(self, shape_prob):
        label_value = Uniform(*self.intensity_range)()
        mask = (shape_prob <= 1).to(torch.float32)
        mask *= label_value
        return mask

    def make_shapes(self):
        centers, axes_list = self.sample_nonoverlapping_geometries()
        for center, axes in zip(centers, axes_list):
            meshgrid = self._meshgrid_origin_at_centroid(center)
            axes = self._add_noise_to_axes(axes)
            shape_prob = self._make_shape_prob(meshgrid, axes)
            shape = self._make_shape_from_prob(shape_prob)
            self.imprint_tensor[shape > 0] = shape[shape > 0]
        return self.imprint_tensor


class MultiLobedBlobSampler(nn.Module):
    def __init__(self, axis_length_range=[3, 6],
                 max_blobs=20, sharpness=3, max_jitter: float = 0.5,
                 num_lobes_range=[1, 5], 
                 device='cuda', shape=64):
        super(MultiLobedBlobSampler, self).__init__()
        self.axis_length_range = axis_length_range
        self.device = device
        self.shape = [shape, shape, shape]
        self.depth, self.height, self.width = self.shape
        self.n_blobs = torch.randint(1, max_blobs + 1, (1,)).item()
        self.sharpness = sharpness
        self.max_jitter = max_jitter
        self.num_lobes_range = num_lobes_range
        self.imprint_tensor = torch.zeros(self.shape, dtype=torch.float32,
                                          device=self.device)
        self.current_label = 1

    def forward(self):
        return self.make_shapes()

    def sample_axis_lengths(self):
        axis_lengths = torch.randint(
            self.axis_length_range[0],
            self.axis_length_range[1] + 1,
            (3,),
            device=self.device)
        return axis_lengths

    def sample_centroid_coords(self, axis_lengths):
        centroid_coords = (
            torch.randint(
                axis_lengths[0],
                self.depth - axis_lengths[0],
                (1,), device=self.device).item(),
            torch.randint(
                axis_lengths[1],
                self.height - axis_lengths[1],
                (1,), device=self.device).item(),
            torch.randint(
                axis_lengths[2],
                self.width - axis_lengths[2],
                (1,), device=self.device).item()
            )
        return centroid_coords

    def check_overlap(self, axis_lengths, centroid_coords, axes_list=[],
                      centers=[]):
        overlap_exists = False
        for c, a in zip(centers, axes_list):
            distance = torch.sqrt(
                torch.tensor((centroid_coords[0] - c[0]) ** 2
                             + (centroid_coords[1] - c[1]) ** 2
                             + (centroid_coords[2] - c[2]) ** 2,
                             device=self.device)
            )
            if distance < max(a) + max(axis_lengths):
                overlap_exists = True
                break
        return overlap_exists

    def sample_nonoverlapping_geometries(self):
        centers = []
        axes_list = []

        while len(centers) < self.n_blobs:
            axis_lengths = self.sample_axis_lengths()
            centroid_coords = self.sample_centroid_coords(axis_lengths)
            overlap_exists = self.check_overlap(axis_lengths, centroid_coords,
                                                axes_list, centers)
            if not overlap_exists:
                centers.append(centroid_coords)
                axes_list.append(axis_lengths.tolist())

        return centers, axes_list

    def _meshgrid_origin_at_centroid(self, center_coords):
        meshgrid = torch.meshgrid(
            torch.arange(self.depth, device=self.device) - center_coords[0],
            torch.arange(self.height, device=self.device) - center_coords[1],
            torch.arange(self.width, device=self.device) - center_coords[2],
            indexing='ij')
        return meshgrid

    def _make_lobe_prob(self, meshgrid, axis_lengths):
        if isinstance(self.sharpness, float):
            sharpness = Uniform(0.75, self.sharpness)()
        elif isinstance(self.sharpness, list):
            sharpness = Uniform(*self.sharpness)()

        noise = torch.normal(0, self.max_jitter,
                             size=[self.depth, self.height, self.width],
                             device=self.device)
        lobe_prob = (torch.abs(meshgrid[0] / (axis_lengths[0] + noise)) ** sharpness +
                     torch.abs(meshgrid[1] / (axis_lengths[1] + noise)) ** sharpness +
                     torch.abs(meshgrid[2] / (axis_lengths[2] + noise)) ** sharpness)
        return lobe_prob

    def _make_lobe_from_prob(self, lobe_prob):
        # label_value = Uniform(*self.intensity_range)()
        mask = (lobe_prob <= 1).to(torch.float32)
        mask = torch.masked_fill(mask, mask.bool(), self.current_label)
        # mask[mask] = self.current_label  # label_value
        self.current_label += 1
        return mask

    def make_shapes(self):
        centers, axes_list = self.sample_nonoverlapping_geometries()
        for center, axes in zip(centers, axes_list):
            meshgrid = self._meshgrid_origin_at_centroid(center)
            lobe_tensor = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
            num_lobes = torch.randint(self.num_lobes_range[0], self.num_lobes_range[1] + 1, (1,)).item()

            for _ in range(num_lobes):
                #lobe_center_shift = torch.randint(-axes[0]//2, axes[0]//2, (3,), device=self.device)
                lobe_center_shift = torch.randint(-axes[0]+1, axes[0]-1, (3,), device=self.device)
                shifted_center = (center[0] + lobe_center_shift[0],
                                  center[1] + lobe_center_shift[1],
                                  center[2] + lobe_center_shift[2])
                shifted_meshgrid = self._meshgrid_origin_at_centroid(shifted_center)
                lobe_prob = self._make_lobe_prob(shifted_meshgrid, axes)
                lobe = self._make_lobe_from_prob(lobe_prob)
                lobe_tensor += lobe

            # self.imprint_tensor += lobe_tensor
            self.imprint_tensor[lobe_tensor > 0] = lobe_tensor[lobe_tensor > 0]
            # self.imprint_tensor[self.imprint_tensor > 1] = 1  # Clip values to 1 for overlap regions

        return self.imprint_tensor

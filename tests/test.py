import unittest
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from veritas.veritas import RealOct, RealOctPatchLoader, RealOctPredict, RealOctDataset

class TestRealOct(unittest.TestCase):

    def setUp(self):
        # Set up for the tests
        self.test_tensor = torch.rand(100, 100, 100)  # Random tensor for input
        self.test_path = 'dummy_path.nii'  # Example file path
        self.real_oct = RealOct(input=self.test_tensor, device='cpu')  # Change device to 'cpu' for testing
    
    def test_initialization(self):
        # Test initialization
        self.assertIsInstance(self.real_oct, RealOct)

    def test_load_tensor(self):
        # Test loading tensor functionality
        tensor, nifti, affine = self.real_oct.load_tensor()
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertIsNone(nifti)  # Should be None since we passed a tensor, not a file path
        self.assertIsNone(affine)

    def test_normalize_volume(self):
        # Test normalization of the volume
        normalized_tensor = self.real_oct.normalize_volume(self.test_tensor)
        self.assertEqual(torch.min(normalized_tensor).item(), 0)
        self.assertEqual(torch.max(normalized_tensor).item(), 1)

    def test_pad_volume(self):
        # Test padding of the volume
        padded_tensor = self.real_oct.pad_volume(self.test_tensor)
        expected_shape = tuple(dim + 2 * self.real_oct.patch_size for dim in self.test_tensor.shape)
        self.assertEqual(padded_tensor.shape, expected_shape)

class TestRealOctPatchLoader(unittest.TestCase):
    def setUp(self):
        self.patch_loader = RealOctPatchLoader(input=torch.rand(100, 100, 100), device='cpu')

    def test_patch_coordinates(self):
        # Test if patch coordinates are calculated correctly
        self.patch_loader.patch_coords()
        self.assertGreater(len(self.patch_loader.complete_patch_coords), 0)

class TestRealOctPredict(unittest.TestCase):
    def setUp(self):
        self.predictor = RealOctPredict(input=torch.rand(100, 100, 100), device='cpu')

    def test_prediction_initialization(self):
        # Test the initialization of predictor
        self.assertIsNotNone(self.predictor.imprint_tensor)

class TestRealOctDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_path = Path(__file__).parent / 'test_data'
        self.dataset = RealOctDataset(path=str(self.dataset_path))

    def test_dataset_length(self):
        # Test the length of the dataset
        self.assertEqual(len(self.dataset), len(list(self.dataset_path.glob('x/*'))))

if __name__ == '__main__':
    unittest.main()
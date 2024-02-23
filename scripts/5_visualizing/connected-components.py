import scipy
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from scipy.ndimage import label
import matplotlib.pyplot as plt
from skimage.measure import regionprops

class ConnectedComponents(object):
    """Connected component analysis of segmented volume"""

    #def __self__(self, segmentation_path):
    #    '''
    #    Parameters
   #     ----------
    #    segmentation_path : str
    #        Path to where prediction or binarized segmentation nifti is located.
#
  #      '''
#
    #    self.segmentation_path = segmentation_path
    #    segmentation_arr = self.load_segmentation()


    def load_segmentation(segmentation_path, threshold:float=None):
        '''
        Load and threshold segmentation array.

        Parameters
        ----------
        threshold : float or None
            Value at which to threshold segmentation_path if it is a prob map.

        Returns
        -------
        segmentation_arr : np.array[uint8]
            binarized segmentation array
        '''
        # loading segmentaion nifti
        segmentation_nifti = nib.load(segmentation_path)
        segmentation_aff = segmentation_nifti.affine
        # loading segmentaion volume and putting it onto gpu
        segmentation_arr = segmentation_nifti.get_fdata()
        # Thresholding volume so we can binarize for cc analysis
        if threshold is not None:
            print(f"Thresholding at {threshold}")
            segmentation_arr = torch.from_numpy(segmentation_arr).to('cuda')
            segmentation_arr[segmentation_arr < threshold] = 0
            segmentation_arr[segmentation_arr >= threshold] = 1
            segmentation_arr = segmentation_arr.cpu().numpy()
        else:
            print("Not thresholding")
        segmentation_arr = segmentation_arr.astype(np.uint8)
        return segmentation_arr, segmentation_aff
    
    def connected_components(segmentaion_arr:np.array):
        """
        Find connected components. Give each a unique ID.
        
        Parameters
        ----------
        segmentation_arr : numpy array
            Array of binarized segmentation.

        Returns
        -------
        labeled_arr : numpy array
            Array of connected components where each component has unique ID.
        n_components : int
            Number of unique connected components. 
        """
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        labeled_arr, n_components = label(segmentation_arr, structure)
        print(f'Found {n_components} connected components!')
        return labeled_arr, n_components
    
    def get_df(labeled_arr, n_components, min_vol:float=None, max_vol:float=None,
               print_df:bool=True):

        hist = scipy.ndimage.histogram(
            labeled_arr,
            min=0,
            max=n_components,
            bins=n_components+1
            )
        df = pd.DataFrame(
            data=hist,
            columns=['volume'],
            index=range(0, n_components+1)
            )
        
        if min_vol is not None:
            print(f'Removing volumes under {min_vol} voxels')
            df = df[df.volume > min_vol]
        if max_vol is not None:
            print(f'Removing volumes above {max_vol} voxels')
            df = df[df.volume < max_vol]
        df = df.sort_values('volume', ascending=False)
        df = df[1:]
        if print_df == True:
            print(df)
        return df
    
    def renumber_labels(labeled_arr, hits_df):
        try:
            labeled_arr = torch.from_numpy(labeled_arr).to('cuda')
        except:
            pass

        labels_to_keep = hits_df.index
        renumbered_labeled_arr = torch.zeros(labeled_arr.shape, dtype=torch.int32).to('cuda')
        for id in range(len(labels_to_keep)):
            renumbered_labeled_arr[labeled_arr == labels_to_keep[id]] = id
        return renumbered_labeled_arr
    
    def save_id(labeled_arr, i, affine):
        try:
            labeled_arr = labeled_arr.cpu().numpy().astype(np.uint8)
        except:
            pass

        regions = regionprops(labeled_arr)
        base_dir = '/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_2561/predictions/connected-components-analysis'

        region_coords = regions[i].bbox
        x = slice(region_coords[0], region_coords[3])
        y = slice(region_coords[1], region_coords[4])
        z = slice(region_coords[2], region_coords[5])
        region = labeled_arr[x, y, z]
        region[region != i+1] = 0

        # Making then applying transformation. Need dot product because some affines have non-zero values in non-id elements.
        obj = np.copy(region).astype(np.uint8)
        obj_affine = np.copy(affine)
        M = obj_affine[:3, :3]
        ijk = obj_affine[:3, 3]
        coords = [x.start, y.start, z.start]
        ijk += np.dot(M, coords)

        nift = nib.nifti1.Nifti1Image(obj, affine=obj_affine)
        out_name = f'{base_dir}/connected-component-{i}.nii.gz'
        nib.save(nift, out_name)

#path = "/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_2561/predictions/v2561_I46_Somatosensory_20um_crop-prediction_stepsz-64_thresh-0.5.nii.gz"
#path = "/autofs/cluster/octdata2/users/epc28/veritas/output/lets_get_small_vessels/version_2561/predictions/caa17_occipital-prediction_stepsz-64_thresh-0.5.nii.gz"
path = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa17/occipital/caa17_prediction-masked.nii.gz'


segmentation_arr, segmentation_aff = ConnectedComponents.load_segmentation(path)
labeled_arr, n_components = ConnectedComponents.connected_components(segmentation_arr)
label_df = ConnectedComponents.get_df(labeled_arr, n_components, min_vol=7000)
labeled_arr = ConnectedComponents.renumber_labels(labeled_arr, label_df)

for i in range(0, 10):
    ConnectedComponents.save_id(labeled_arr, i, segmentation_aff)
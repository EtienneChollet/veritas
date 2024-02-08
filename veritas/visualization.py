__all__ = [
    'Visualize',
    'Confusion'
]

# Standard Imports
import torch
import numpy as np
import nibabel as nib
import tifffile
# Custom Imports
from veritas.data import RealOct
from veritas.utils import volume_info

class Visualize(object):
    """
    Base class for visualization.
    """
    def __init__(self, in_path, out_path=None, out_name='movie'):
        self.in_path = in_path
        if out_path is None:
            # Out path is the same as the in path
            self.out_path = f"{'/'.join(self.in_path.split('/')[:-1])}/{out_name}.tiff"
        else:
            self.out_path = out_path
        self.vol = self.load_base_vol()

    def main(self,
             ground_truth:str=None,
             prediction:str=None,
             binary_threshold=0.5,
             model_dir:str='/autofs/cluster/octdata2/users/epc28/veritas/output/models',
             model_version:int=1
             ):
        if ground_truth is not None:
            pass
        if prediction is not None:
            pass
        if binary_threshold:
            prediction_tensor = RealOct(
                input=self.in_path,
                binarize=True,
                binary_threshold=binary_threshold,
                device='cpu',
                dtype=torch.float32
                )
            binarized_volume = prediction_tensor.tensor.cpu().numpy()
            #[binarized_volume < 0.5] = 0
            #binarized_volume[binarized_volume >= 0.5] = 1
            affine = prediction_tensor.affine
            out_nifti = nib.nifti1.Nifti1Image(dataobj=binarized_volume, affine=affine)
            binarized_volume_path = self.in_path.split('/')[-1].strip('.nii')
            binarized_volume_name = f'{binarized_volume_path}_thresh-{binary_threshold}_BINARIZED.nii'
            binarized_volume_path = self.in_path.split('/')[:-1]
            binarized_volume_path.append(binarized_volume_name)
            binarized_volume_path = '/'.join(binarized_volume_path)
            print('saving to:', binarized_volume_path)
            nib.save(out_nifti, binarized_volume_path)

                
    def load_base_vol(self):
        """
        Load base volume and convert to 3 channels.

        Returns
        -------
        self.vol : tensor
            Base volume.
        """
        vol = RealOct(
            input=self.in_path,
            device='cuda',
            normalize=True,
            pad_it=False,
            ).tensor.cpu().numpy()
        
        # Clamping
        vol[vol > 1] = 1
        vol[vol < 0] = 0
        # Converting to 8-bit color
        vol = np.uint8(vol* 255)
        vol = np.stack((vol, vol, vol), axis=-1)
        return vol


    def overlay(self, overlay_tensor:torch.Tensor, name:str, rgb=[0, 0, 255], binary_threshold=None):
        """
        Overlay something onto base volume.

        Parameters
        ----------
        overlay_tensor : tensor[bool]
            Tensor to use as overlay. shape = [x,y,z]
        """
        if isinstance(overlay_tensor, str):
            print(f'Loading overlay {name} from path...\n')
            overlay_tensor = RealOct(
                volume=overlay_tensor,
                binarize=True,
                binary_threshold=0.5,
                device='cpu',
                dtype=torch.uint8
                ).tensor.numpy()
        elif isinstance(overlay_tensor, torch.Tensor):
            print(f'Using tensor {name} as numpy arr...\n')
            overlay_tensor = overlay_tensor.numpy()
        elif isinstance(overlay_tensor, np.ndarray):
            print(f"Using numpy array {name}...")
        else:
            print(f'Having trouble with overlaying {name}')
            exit(0)

        #print(f"{name} max = {overlay_tensor.max()}")
        #print(f"{name} min = {overlay_tensor.min()}")
        
        for i in range(3):
            self.vol[..., i][overlay_tensor == 1] = 0
            self.vol[..., i] += (overlay_tensor * rgb[i])

        
    def make_tiff(self, zoom:bool=False):
        print('Making tiff...')
        self.vol[self.vol > 255] = 255
        if zoom == True:
            from scipy import ndimage
            self.vol = ndimage.zoom(self.vol, [1, 12, 12, 1], order=0)
        print(f'Saving to: {self.out_path}...')
        tifffile.imwrite(self.out_path, self.vol)



#vol_path = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/64x64x64_sub-I38_ses-OCT_sample-BrocaAreaS01_OCT-volume.nii'
#ground_truth = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/64_ground-truth.nii'
#prediction = '/autofs/cluster/octdata2/users/epc28/veritas/data/UO1/predictions/64x64x64_sub-I38_ses-OCT_sample-BrocaAreaS01_OCT-volume-prediction_stepsz-32.nii'

#vis = Visualize(vol_path, out_name='prediction')
#vis.overlay(ground_truth, name='ground_truth', rgb=[0, 0, 255])
#vis.overlay(prediction, name='prediction', rgb=[0, 255, 255])
#vis.make_tiff()



class Confusion(object):

    def __init__(self, ground_truth, prediction, binary_threshold:int=0.5):
        
        self.ground_truth = RealOct(
            input=ground_truth,
            binarize=True,
            device='cpu',
            dtype=torch.uint8
            ).tensor.numpy()
        
        self.prediction = RealOct(
            input=prediction,
            binarize=True,
            binary_threshold=binary_threshold,
            device='cpu',
            dtype=torch.float32
            ).tensor.numpy()
        #volume_info(self.prediction, 'prediction')
        
        
    def true_positives(self):
        """
        True positives (yellow)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.prediction == 1] += 1
        out_vol[self.ground_truth == 1] += 1
        out_vol[out_vol < 2] = 0
        out_vol[out_vol >= 2] = 1
        rgb = [255, 255, 0]
        print(f'TP: {out_vol.sum()}')
        return out_vol, rgb


    def false_positives(self):
        """
        False positives (red)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.prediction == 1] = 1
        out_vol[self.ground_truth == 1] = 0
        #out_vol = self.prediction - self.ground_truth
        #out_vol[out_vol <= 0] = 0
        #out_vol[out_vol >= 1] = 1
        rgb = [255, 0, 0]
        print(f'FP: {out_vol.sum()}')
        return out_vol, rgb
    

    def false_negatives(self):
        """
        False negatives (green)
        """
        out_vol = np.zeros(self.ground_truth.shape, dtype=np.uint8)
        out_vol[self.ground_truth >= 1] = 1
        out_vol[self.prediction == 1] = 0
        rgb = [0, 255, 0]
        print(f'FN: {out_vol.sum()}')
        return out_vol, rgb
    
my_segmentation = '/autofs/cluster/octdata2/users/epc28/veritas/output/paper_models_ablation/version_11/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-16.nii'
james_segmentation = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/ground_truth.nii'

confusion = Confusion(my_segmentation, james_segmentation, binary_threshold=0.005)
tp = confusion.true_positives()[0].sum()
fn = confusion.false_negatives()[0].sum()
fp = confusion.false_positives()[0].sum()

dice_score = round((2 * tp) / ((2 * tp) + fp + fn), 3)
print(dice_score)

exit(0)
prediction = '/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_8/predictions/caa17_occipital.nii.gz-prediction_stepsz-256.nii'
Visualize(in_path=prediction).main()


binary_threshold = 0.5
model_version = 1

vol_path = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/I46_Somatosensory_20um_crop.nii'
ground_truth = '/autofs/cluster/octdata2/users/epc28/data/caroline_data/ground_truth.nii'
prediction = f'/autofs/cluster/octdata2/users/epc28/veritas/output/paper_models_64/version_{model_version}/predictions/I46_Somatosensory_20um_crop-prediction_stepsz-16.nii'
prediction = '/autofs/cluster/octdata2/users/epc28/veritas/output/paper_models_small/version_1/predictions/post_processed_thresh-0.5/20vx-removed.nii'


out_name = prediction.split('/')[-1].strip('.nii')
prediction_dir = '/'.join(prediction.split('/')[:-1])

# true positive = yellow, false positive = red, false negative = green
confusion = Confusion(ground_truth, prediction, binary_threshold=binary_threshold)
tp, tp_rgb = confusion.true_positives()
fp, fp_rgb = confusion.false_positives()
fn, fn_rgb = confusion.false_negatives()

dice_score = round((2 * tp.sum()) / ((2 * tp.sum()) + fp.sum() + fn.sum()), 3)
print(f"DICE SCORE: {dice_score}")
out_path = f"{prediction_dir}/{out_name}_thresh-{binary_threshold}_dice-{dice_score}.tiff"
#out_path = f"{prediction_dir}/{out_name}.tiff"

vis = Visualize(vol_path, out_path=out_path)
vis.overlay(tp, name='true_positives', rgb=tp_rgb, binary_threshold=binary_threshold)
vis.overlay(fp, name='false_positives', rgb=fp_rgb, binary_threshold=binary_threshold)
vis.overlay(fn, name='false_negatives', rgb=fn_rgb, binary_threshold=binary_threshold)
vis.make_tiff()
print(f"{dice_score}\n{tp.sum()}\n{fp.sum()}\n{fn.sum()}")

##### Saving binarized volume as nifti to do segment analysis :)

prediction_tensor = RealOct(
    input=prediction,
    binarize=True,
    binary_threshold=binary_threshold,
    device='cpu',
    dtype=torch.float32
)
binarized_volume = prediction_tensor.tensor.cpu().numpy()
affine = prediction_tensor.affine
out_nifti = nib.nifti1.Nifti1Image(dataobj=binarized_volume, affine=affine)
binarized_volume_path = prediction.split('/')[-1].strip('.nii')
binarized_volume_name = f'{binarized_volume_path}_thresh-{binary_threshold}_BINARIZED.nii'
binarized_volume_path = prediction.split('/')[:-1]
binarized_volume_path.append(binarized_volume_name)
binarized_volume_path = '/'.join(binarized_volume_path)

nib.save(out_nifti, binarized_volume_path)
__init__ = [
    'PatchValidation',
    'ValidationEngine',
    'DataFrameManager',
    'PredictionValidator'
]

import torch
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

# It's better to move sub-module specific imports here, if they do not lead to circular dependencies.
from veritas.data import SubpatchExtractor
from veritas.stats import MetricsCalculator

class PatchValidation:
    """
    Compare each patch to ground truth.

    Parameters
    ----------
    case : str
        Medical case name of patches to validate
    roi : str
        Region of interest for patches to validate
    model_n : int
        Model version that was used for predictions
    subpatch_id : int
        Patch number to validate
    model_exp_name : str
        Subdir of output that contains models of interest

    Attributes
    ----------
    gts : list[torch.Tensor]
        List of ground truth tensors. (one tensor per rater)
    pred : torch.Tensor
        Prediction tensor binarized at 0.5
    """
    def __init__(
            self,
            case: str,
            roi: str,
            model_n: int = 1,
            subpatch_id:int = None,
            model_exp_name: str = 'models',
            ):
        self.case = case
        self.roi = roi
        self.model_n = model_n
        self.subpatch_id = subpatch_id
        self.case_roi_basedir = f'/autofs/cluster/octdata2/users/epc28/data/CAA/{case}/{roi}'
        self.model_prediction_dir = f'/autofs/cluster/octdata2/users/epc28/veritas/output/{model_exp_name}/version_{model_n}/predictions'
        self.gts = self.get_gts()
        self.pred = self.get_pred()


    def get_gts(self):
        """
        Load ground truth tensors (one for each rater)
        
        Returns
        -------
        gt_tensors : list[torch.Tensor]
            List of ground truth tensors
        """
        gt_tensors = []
        path = f'{self.case_roi_basedir}/ground_truth/etienne/gt_{self.subpatch_id}.nii'
        tensor = torch.from_numpy(nib.load(path).get_fdata()).to('cuda')
        gt_tensors.append(tensor)
        path = f'{self.case_roi_basedir}/ground_truth/james/gt_{self.subpatch_id}.nii'
        tensor = torch.from_numpy(nib.load(path).get_fdata()).to('cuda')
        gt_tensors.append(tensor)
        return gt_tensors


    def get_pred(self):
        """
        Load binarized prediction tensor.

        Returns
        -------
        pred : torch.Tensor
            Binarized prediction tensor
        """
        path = f'{self.model_prediction_dir}/{self.case}-{self.roi}_patch-{self.subpatch_id}.nii'
        pred = nib.load(path)
        self.affine = pred.affine
        pred = torch.from_numpy(pred.get_fdata()).to('cuda')
        pred = (pred>=0.5).int()
        return pred
    
    
    def get_metrics(self, who='both'):
        """
        Compute metrics between prediction and rater(s)

        Returns
        -------
        metrics : list[float]
            Metrics of accuracy (dice, fpr, fnr)
        """
        gt1_metrics = MetricsCalculator(self.gts[0], self.pred, metric='dice').get_all()
        gt2_metrics = MetricsCalculator(self.gts[1], self.pred, metric='dice').get_all()
        if who == 'both':
            metrics = (gt1_metrics + gt2_metrics) / 2
        elif who == 'r1':
            metrics = gt1_metrics
        elif who == 'r2':
            metrics = gt2_metrics
        return metrics
    
    def save_confusion(self):
        """
        Generates confusion volumes WRT Eti. Compatable with freeview.
        """
        pred = (self.pred >= 0.5).float()
        #print(torch.count_nonzero(pred))
        gt = (self.gts[0] > 0).float()
        #print(torch.count_nonzero(gt))
        # True positives = 17 (Yellow).
        tp = ((gt == 1) & (pred == 1)).float()
        tp *= 17
        # False positives = 3 (Red).
        fp = ((gt == 0) & (pred == 1)).float()
        fp *= 3
        # False negatives = 6 (green).
        fn = ((gt == 1) & (pred == 0)).float()
        fn *= 6
        out_vol = tp + fp + fn
        confusion_path = f'{self.model_prediction_dir}/{self.case}-{self.roi}_patch-{self.subpatch_id}_confusion.nii'
        nift = nib.nifti1.Nifti1Image(out_vol.cpu().numpy().astype('float'), self.affine)
        nib.save(nift, confusion_path)
        return out_vol


class ValidationEngine:
    """A class to handle validation and metric computation for 3D imaging data."""

    def __init__(self, parent_path=None, gt_paths: list = None):
        """
        Initialize the ValidationEngine with the parent volume data.

        Parameters
        ----------
        parent_path : str, optional
            Path to the file containing the parent 3D tensor data.
        gt_paths : list of str, optional
            List of paths to ground truth data tensors.
        """
        self.parent_path = parent_path
        self.gt_paths = gt_paths
        self.extractor = SubpatchExtractor(parent_path)
        print('Extractor Loaded')
        self.combined_gt = self._get_gts(self.gt_paths)
        self.coords = self.extractor.find_subpatch_coordinates_using_affine(self.subpatch_affine)
        self.prediction_tensor = self._get_prediction_patch()


    def get_metrics(self, threshold: float = 0.5):
        """Calculate and print various performance metrics based on a threshold.

        Parameters
        ----------
        threshold : float, optional
            Threshold value to classify predictions as positive or negative.
        """
        pred = self.prediction_tensor
        gt = self.combined_gt

        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1
        pred = pred.to(torch.bool)

        gt[gt < 1] = 0
        gt[gt >= 1] = 1
        gt = gt.to(torch.bool)

        tp = (gt * pred).to(torch.uint8) * 17
        n_tp = torch.count_nonzero(tp)
        # Red
        fp = (gt < pred).to(torch.uint8) * 3
        n_fp = torch.count_nonzero(fp)
        # Green
        fn = (gt > pred).to(torch.uint8) * 6
        n_fn = torch.count_nonzero(fn)

        n = torch.numel(gt)
        n_tn = n - (n_tp + n_fp + n_fn)
        dice_gt = (2 * n_tp) / ((2 * n_tp) + n_fp + n_fn)
        fpr = n_fp / (n_fp + n_fn + n_tp) #(n_fp + n_tn)
        fnr = n_fn / (n_fp + n_fn + n_tp) #(n_fn + n_tp)
        #print(f'{dice_gt}\t{fpr}\t{fnr}')
        return (dice_gt.item(), fpr.item(), fnr.item())

    def refresh_data(self, gt_paths):
        """
        Refresh the ground truth data and prediction tensor without reloading
        the parent volume.

        Parameters
        ----------
        gt_paths : list of str
            List of paths to new ground truth data tensors.
        """

        self.combined_gt = self._get_gts(gt_paths)
        self.coords = self.extractor.find_subpatch_coordinates_using_affine(
            self.subpatch_affine
            )
        # Assuming the coordinates and affine transformation do not need to be
        # re-found unless explicitly updated
        self.prediction_tensor = self._get_prediction_patch()

            

    def _get_prediction_patch(self):
        """Extract a prediction patch based on stored coordinates."""
        prediction_tensor = self.extractor.extract_subpatch(
            self.coords, [64, 64, 64], return_='tensor'
            )
        return prediction_tensor

    
    def _get_gts(self, paths: list):
        """Aggregate ground truth data from multiple paths.

        Parameters
        ----------
        paths : list of str
            Paths to ground truth data tensors.
        """
        combined_gt = torch.ones([64, 64, 64], device='cuda')
        self.subpatch_affine = nib.load(paths[0]).affine
        for path in paths:
            gt = torch.from_numpy(nib.load(path).get_fdata()).to('cuda')
            gt.clip_(0, 1)
            combined_gt = torch.logical_and(combined_gt, gt)
        self.combined_gt = combined_gt
        return combined_gt
    

class DataFrameManager:
    
    def __init__(self, file_path):
        """Initialize the DataFrame manager with a file path."""
        self.file_path = file_path
        self.dtype_spec = {
            'model': str,
            'case': str,
            'roi': str,
            'patch': str,
            'dice': float,
            'fpr': float,
            'fnr': float,
            }
        try:
            self.df = pd.read_csv(file_path, dtype=self.dtype_spec)
        except FileNotFoundError:
            # If the file does not exist, initialize an empty DataFrame with specified columns
            self.df = pd.DataFrame(columns=['model', 'case', 'roi', 'patch', 'dice', 'fpr', 'fnr'])
            print("File not found. Initialized empty DataFrame.")

    def update_or_append_entry(self, new_data, key_columns):
        """Update an entry if it exists based on key_columns, otherwise append it."""
        new_row = pd.DataFrame([new_data]).astype(self.dtype_spec)
        
        # Correctly create a mask for checking existence
        conditions = [self.df[col] == new_data[col] for col in key_columns]
        mask = pd.concat(conditions, axis=1).all(axis=1)
        
        if mask.any():
            # Entry exists. Overwrite it. 
            # Ensure all columns are aligned and updated
            for col in self.df.columns:
                if col in new_row.columns:
                    self.df.loc[mask, col] = new_row[col].iloc[0]            
            message = "Entry exists. It has been overwritten."
        else:
            # Entry does not exist, append it
            self.df = pd.concat([self.df, new_row], ignore_index=True)
            message = "New entry added."
        
        #print(message)
        
    def save_changes(self):
        """Save the changes back to the file."""
        self.df.to_csv(self.file_path, index=False)



class PredictionValidator:
    def __init__(self, case, roi, version, patches, base_data_path, database_path):
        self.case = case
        self.roi = roi
        self.version = str(version)
        self.patches = patches
        self.base_data_path = base_data_path
        self.database_path = database_path

    def validate_and_update(self):
        for patch in self.patches:
            gt1_path = f'{self.base_data_path}/{self.case}/{self.roi}/ground_truth/etienne/gt_{patch}.nii'
            gt2_path = f'{self.base_data_path}/{self.case}/{self.roi}/ground_truth/james/gt_{patch}.nii'
            gt_paths = [gt1_path, gt2_path]
            
            parent_path = f'/autofs/cluster/octdata2/users/epc28/veritas/output/models/version_{self.version}/predictions/{self.case}_{self.roi}-prediction_stepsz-32.nii.gz'
            val_engine = ValidationEngine(parent_path=parent_path, gt_paths=gt_paths)
            metrics = val_engine.get_metrics()
            #print(metrics)
            
            # Update data in DataFrame
            self.update_dataframe(patch, metrics)

    def update_dataframe(self, patch, metrics):
        manager = DataFrameManager(self.database_path)
        new_data = {
            'model': self.version,
            'case': self.case,
            'roi': self.roi,
            'patch': patch,
            'dice': metrics[0],
            'fpr': metrics[1],
            'fnr': metrics[2]
        }
        check_columns = ['model', 'case', 'roi', 'patch']
        manager.update_or_append_entry(new_data, check_columns)
        manager.save_changes()
        print("DataFrame updated for patch:", patch)

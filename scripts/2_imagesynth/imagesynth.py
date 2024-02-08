# Custom Imports
from veritas.synth import OctVolSynthDataset

if __name__ == "__main__":
    synth = OctVolSynthDataset(
        exp_path="output/synthetic_data/exp0128",
        label_type='label',
        synth_params='complex'
        )
    
    for i in range(20):
        synth.__getitem__(i, save_nifti=True, make_fig=True, save_fig=True)
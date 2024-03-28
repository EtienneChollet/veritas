# Custom Imports
from veritas.synth import OctVolSynthDataset
import time

if __name__ == "__main__":
    synth = OctVolSynthDataset(
        exp_path="/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0004",
        label_type='label',
        synth_params='complex',
        )
    t1 = time.time()
    for i in range(5):
        synth.__getitem__(i, save_nifti=True, make_fig=False, save_fig=False)
    t2 = time.time()
    print(t2-t1)
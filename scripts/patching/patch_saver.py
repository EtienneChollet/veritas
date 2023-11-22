import pandas as pd
import nibabel as nib
from veritas.data import RealOctPatchLoader

colnames = ['case', 'roi', 'raw_volume', 'prediction_volume', 'mean_gray', 'mean_white']

paths = pd.read_csv(
    '/autofs/cluster/octdata2/users/epc28/data/datasets_csv/caa.csv',
    dtype='category',
    names=colnames,
    index_col=colnames[:2],
    ).sort_index()

#case = paths.loc['caa6', 'frontal']
#mean_gray = case.mean_gray
#seed = RealOctPatchLoader(
#    input=case.raw_volume,
#    patch_size=64,
#    step_size=64,
#    normalize=False
#    ).random_patch_sampler(10, threshold=mean_gray)
#RealOctPatchLoader(
#    input=case.prediction_volume,
#    patch_size=64,

#    step_size=64,
#    normalize=False
#    ).random_patch_sampler(10, seed=seed, name_prefix='prediction')
#


for i in range(len(paths) - 1):
    print('\n')
    print('#' * 20)

    seed = RealOctPatchLoader(
        input=paths.iloc[i].raw_volume,
        patch_size=64,
        step_size=64,
        ).random_patch_sampler(10, threshold=paths.iloc[i].mean_gray)

    RealOctPatchLoader(
        input=paths.iloc[i].prediction_volume,
        patch_size=64,
        step_size=64
        ).random_patch_sampler(10, seed=seed, name_prefix='prediction')

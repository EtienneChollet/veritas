from veritas.data import RealOct

path = '/autofs/cluster/octdata2/users/epc28/data/CAA/caa6/occipital/caa6_occipital.nii'
caa6 = RealOct(path, device='cpu')

tissue_mask = caa6.get_mask('tissue-mask')
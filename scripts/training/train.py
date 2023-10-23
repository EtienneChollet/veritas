# Custom Imports
from veritas.models import Unet
from veritas.synth import OctVolSynth

import torch
torch.no_grad()

if __name__ == "__main__":
    unet = Unet(version_n=9)
    unet.new(nb_levels=3, nb_features=[32, 64, 128])
    data_path = '/autofs/cluster/octdata2/users/epc28/veritas/output/synthetic_data/exp0009'
    unet.train_it(data_path, epochs=25, batch_size=2, augmentation=OctVolSynth())
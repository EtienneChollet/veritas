import torch
from veritas.synth import VesselSynth

from vesselsynth.vesselsynth.utils import backend
backend.jitfields = True

if __name__ == "__main__":
    torch.no_grad()
    VesselSynth(
        experiment_dir='',
        experiment_number=256
        ).synth()
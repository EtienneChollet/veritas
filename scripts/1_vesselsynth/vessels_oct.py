import torch
from veritas.synth import VesselSynth
torch.no_grad()

from vesselsynth.vesselsynth.utils import backend
backend.jitfields = True

if __name__ == "__main__":
    VesselSynth(experiment_number=6).synth()
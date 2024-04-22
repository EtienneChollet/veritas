import torch
from veritas.synth import VesselSynth
from vesselsynth.vesselsynth.utils import backend

backend.jitfields = True

if __name__ == "__main__":
    torch.no_grad()
    VesselSynth(
        experiment_number=1,
        device='cuda'
        ).synth()
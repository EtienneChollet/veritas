from veritas.synth import VesselSynth

import torch
torch.no_grad()

if __name__ == "__main__":
    VesselSynth(experiment_number=1).synth()
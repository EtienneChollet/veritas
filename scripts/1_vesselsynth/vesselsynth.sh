#!/bin/bash
# dgx-a100, rtx8000, rtx6000

jobsubmit -A psoct -p dgx-a100 -m 50G -t 7-00:00:00 -c 32 -G 1 -o logs/vesselsynth_exp0256.log python3 scripts/1_vesselsynth/vessels_oct.py;
watch -n 0.1 "squeue -u $USER"

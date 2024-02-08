#!/bin/bash
# -m 20 -c8

jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 20 -G 1 -o logs/paper-model-context-128-v1.log python3 scripts/3_training/train.py;
watch -n 1 "squeue -u $USER"
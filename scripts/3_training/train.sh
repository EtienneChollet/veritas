#!/bin/bash
# -m 20 -c8

jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 8 -G 4 -o logs/train_context-64_v4.log python3 scripts/3_training/train.py;
watch -n 1 "squeue -u $USER"
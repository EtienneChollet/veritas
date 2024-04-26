#!/bin/bash
#dgx-a100

jobsubmit -A psoct -p rtx6000 -m 20G -t 7-00:00:00 -c 4 -G 1 -o logs/caroline_version_11111-train.log python3 scripts/3_training/train_caroline.py;
watch -n 1 "squeue -u $USER"
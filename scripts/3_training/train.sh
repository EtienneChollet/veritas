#!/bin/bash
#dgx-a100

jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 4 -G 1 -o logs/version_9999-train.log python3 scripts/3_training/train.py;
watch -n 1 "squeue -u $USER"
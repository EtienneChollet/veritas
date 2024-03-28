#!/bin/bash

jobsubmit -A psoct -p rtx8000 -m 20G -t 7-00:00:00 -c 2 -G 1 -o logs/version_444-train.log python3 scripts/3_training/train.py;
watch -n 1 "squeue -u $USER"
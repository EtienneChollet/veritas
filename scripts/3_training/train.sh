#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 8 -G 1 -o logs/train_2.log python3 scripts/3_training/train.py;
watch -n 0.1 "squeue -u $USER"
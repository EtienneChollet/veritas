#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 50G -t 7-00:00:00 -c 2 -G 1 -o logs/test.log python3 scripts/4_testing/test.py;
watch -n 0.1 "squeue -u $USER"
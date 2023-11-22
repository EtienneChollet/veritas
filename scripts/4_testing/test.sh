#!/bin/bash

jobsubmit -A psoct -p dgx-a100 -m 100G -t 7-00:00:00 -c 50 -G 1 -o logs/test.log python3 sandbox/dataloading_debugging.py;
watch -n 0.1 "squeue -u $USER"
#!/bin/bash

#jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 4 -G 1 -o logs/test.log python3 /autofs/cluster/octdata2/users/epc28/veritas/sandbox/dataloading_debugging.py;
jobsubmit -A psoct -p dgx-a100 -m 75G -t 7-00:00:00 -c 8 -G 1 -o logs/test/caa6-frontal.log python3 /autofs/cluster/octdata2/users/epc28/veritas/scripts/4_testing/test.py;
watch -n 0.1 "squeue -u $USER"
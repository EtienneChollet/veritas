#!/bin/bash

#jobsubmit -A psoct -p dgx-a100 -m 20G -t 7-00:00:00 -c 4 -G 1 -o logs/test.log python3 /autofs/cluster/octdata2/users/epc28/veritas/sandbox/dataloading_debugging.py;
jobsubmit -A psoct -p dgx-a100 -m 75G -t 7-00:00:00 -c 6 -G 1 -o logs/test/caa26-frontal.log python3 /autofs/cluster/octdata2/users/epc28/veritas/scripts/4_testing/test.py -v 2,22,222,3,33,333,4,44,444,5,55,555,6,66,666,7,77,777,8,88,888,101,102,103 -i /autofs/cluster/octdata2/users/epc28/data/CAA/caa26/frontal/caa26_frontal.nii;
watch -n 0.1 "squeue -u $USER"

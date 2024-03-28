cd /autofs/cluster/octdata2/users/epc28/veritas;
export PYTHONPATH=/autofs/cluster/octdata2/users/epc28/veritas;
mamba activate veritas;
alias veritas-test='python3 scripts/4_testing/test.py'
alias veritas-confusion='python3 scripts/4_testing/confusion.py'

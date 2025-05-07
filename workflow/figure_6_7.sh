#!/bin/bash
# (0) load in the appropriate modules, venvs
module load anaconda
conda activate my_envir


# (1) create folders with all outputs of 500 runs
PY_STR_PLOT="python ./figure_6_7.py"
SRUN_OPTS_PLOT="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=1"
srun $SRUN_OPTS_PLOT $PY_STR_PLOT &
wait

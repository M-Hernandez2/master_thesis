#!/bin/bash
#SBATCH --partition=open
#SBATCH --job-name=move_SEAWAT_outs
#SBATCH --output=../NEW_inputs/output/%x.out
#SBATCH --error=../NEW_inputs/error/%x.err
#SBATCH --time=1:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2


# (0) load in the appropriate modules, venvs
module load anaconda
conda activate dover_env


# (1) create folders with all outputs of 500 runs
PY_STR_PLOT="python ./move_concentration.py"
SRUN_OPTS_PLOT="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=2"
srun $SRUN_OPTS_PLOT $PY_STR_PLOT &
wait

#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --partition=open
#SBATCH --job-name=multiple_swt_runs
#SBATCH --output=../output/%x.out
#SBATCH --error=../error/%x.err
#SBATCH --time=48:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48


# (0) load in the appropriate modules
module load parallel
SH_STR="bash ../NEW_inputs/Final_Runs/worker.sh"


# (1*) use this one when you've got the first one working
SLR_SEQ=$(seq 1 5)
PR_SEQ=$(seq 1 100)
SRUN_OPTS="--exclusive --nodes=1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK"
PARALLEL_OPTS="--joblog ../New_inputs/Final_Runs/logfile.log --delay 0.1auto --jobs $SLURM_CPUS_PER_TASK -N 1"	
srun $SRUN_OPTS parallel $PARALLEL_OPTS $SH_STR ::: $SLR_SEQ ::: $PR_SEQ &
wait

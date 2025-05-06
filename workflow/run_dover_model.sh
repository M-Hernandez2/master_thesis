#!/bin/bash
#SBATCH --mail-user=mjh7517@psu.edu
#SBATCH --mail-type=ALL
#SBATCH --partition=open
#SBATCH --job-name=old_test
#SBATCH --output=../output/%x.out
#SBATCH --error=../error/%x.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

set -e
echo "Job started on $(hostname) at $(date)"

cd /storage/group/azh5924/default/Maddie/New_Runs/old_run/SEAWAT

#  execute
./swtv4 EastDoverSWI.SEAWAT.nam
 
echo "Job finished at $(date)"

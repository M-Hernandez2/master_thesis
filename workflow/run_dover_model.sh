#!/bin/bash
#SBATCH --partition=open
#SBATCH --job-name=run_seawat_model
#SBATCH --output=../output/%x.out
#SBATCH --error=../error/%x.err
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20

set -e
echo "Job started on $(hostname) at $(date)"

cd ../
#  execute
./swtv4 EastDoverSWI.SEAWAT.nam
 
echo "Job finished at $(date)"

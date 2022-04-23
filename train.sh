#!/bin/bash
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=256000
#SBATCH --partition=cbmm
#SBATCH --job-name="job1"
#SBATCH --output="job1.out"
#SBATCH -t 05:00:00
module load openmind/singularity/3.5.0
singularity exec --nv -B /om,/om2 /om2/user/chenn/env36.simg ./_train.sh
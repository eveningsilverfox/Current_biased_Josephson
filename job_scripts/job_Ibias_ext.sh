#!/bin/bash

#SBATCH -J cbjj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=64G
#SBATCH -t 0-23:29:59


cd /u/arila/Work/Current_biased_Josephson/
srun julia --threads 72 Josephson_Ibias_Floquetn_ext.jl

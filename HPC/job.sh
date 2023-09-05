#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -c 2
#SBATCH --mem 200GB
#SBATCH -q tr72
#SBATCH -t 5:00:00

# Output and error files
#SBATCH -o script_files/comp_phy_job.%J.out
#SBATCH -e script_files/comp_phy_job.%J.err

# Send email when job finishes
#SBATCH --mail-type=END
#SBATCH --mail-user=mps565@nyu.edu

#Resource requiremenmt commands end here

#activate any environments if required
source ~/.bashrc
conda activate /scratch/mps565/conda-envs/jupyter

#Execute the code
python simulation_1.py

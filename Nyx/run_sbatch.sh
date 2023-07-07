#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=01:00:00                  #  hh:mm:ss
#SBATCH --ntasks=16                      # number of mpi tasks
#SBATCH --cpus-per-task=1                
#SBATCH --account=nyx
#SBATCH --email-type=BEGIN,END,FAIL
#SBATCH --email-user=ajohnson3@lbl.gov
#SBATCH --qos=regular
#SBATCH --constraint=haswell

# make sure necessary modules are loaded
#module purge
#module load 

# run the executable
srun 

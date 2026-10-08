#!/bin/bash
#SBATCH -J job_tests
#SBATCH -N 1
#SBATCH --ntasks-per-node 112
#SBATCH -t 24:00:00
#SBATCH -p pbatch
#SBATCH -A pls2
#SBATCH -V
#SBATCH -o stdoutmsg

# module load cmake intel impi

# srun -N 1 -n 1 /p/lustre1/laubach2/lammps/build/lmp -i nequip_lmp.in  > out.lammps

module load cmake intel/2022.1.0 openmpi/4.1.2

export OMP_NUM_THREADS=56

# # Run directly without mpirun/srun
/p/lustre1/laubach2/lammps/build/lmp \
  -sf omp -pk omp 112 \
  -i nequip_lmp.in > out.lammps
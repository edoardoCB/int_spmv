#!/bin/sh
#SBATCH --job-name=INT_M25_KAY
#SBATCH --partition=LongQ
#SBATCH --account=hpce3ic5
#SBATCH --nodes=1
#SBATCH --time=2:00:00
module load intel/2017u8
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=1
./intSpmv M25_Serena.bin

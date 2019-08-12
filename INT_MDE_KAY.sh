#!/bin/sh
#SBATCH --job-name=INT_MDE_KAY
#SBATCH --partition=DevQ
#SBATCH --account=hpce3ic5
#SBATCH --nodes=1
#SBATCH --time=0:02:00
module load intel/2017u8
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=1
./intSpmv MDE.csr

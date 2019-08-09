#!/bin/bash



echo "icc -O3 -qopenmp -D _OMP_ -xCORE-AVX512 -D _KAY_ -D FP_TYPE=FP_DOUBLE -D NUM_ITE=$1 intSpmv_main.c -mkl -o intSpmv"
icc -O3 -qopenmp -D _OMP_ -xCORE-AVX512 -D _KAY_ -D FP_TYPE=FP_DOUBLE -D NUM_ITE=$1 intSpmv_main.c -mkl -o intSpmv




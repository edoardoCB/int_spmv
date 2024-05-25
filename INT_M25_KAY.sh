#!/bin/sh
#
#            int_spmv
# (SpMV product on Intel Xeon Gold processors)
#
# (C) 2019, University of Santiago de Compostela
#
# Author: Edoardo Coronado <eecb76@hotmail.com>
#
# Program: int_spmv
# File: INT_M25_KAY.sh
# code dated: 12-08-2019 (dd-mm-yyyy)
#
#	gpu_spmv is free software: you can redistribute it and/or modify
#	it under the terms of the GNU General Public License as published by
#	the Free Software Foundation, either version 3 of the License, or
#	(at your option) any later version.
#
#	This program is distributed in the hope that it will be useful,
#	but WITHOUT ANY WARRANTY; without even the implied warranty of
#	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#	GNU General Public License for more details.
#
#	You should have received a copy of the GNU General Public License
#	along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
////////////////////////////////////////////////////////////////

echo
echo "int_spmv - SpMV product on NVIDIA's GPU"
echo "File: INT_M25_KAY.sh"
echo "(C) 2019, University of Santiago de Compostela"
echo "Author: Edoardo Coronado"
echo
echo "This program comes with ABSOLUTELY NO WARRANTY."
echo "This is free software, and you are welcome to redistribute it under"
echo "certain conditions; see README.md and LICENSE.txt for details."
echo


#SBATCH --job-name=INT_M25_KAY
#SBATCH --partition=LongQ
#SBATCH --account=hpce3ic5
#SBATCH --nodes=1
#SBATCH --time=2:00:00
module load intel/2017u8
export OMP_NUM_THREADS=40
export MKL_NUM_THREADS=1
./intSpmv M25_Serena.bin

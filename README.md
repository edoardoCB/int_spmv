# int_spmv

This project contains the code to perform the SpMV product.

CSR format.
	- Naive version
	- MKL deprecated function
	- MKL new function without inspector
	- MKL new function with inspector

AXC format.
	- Original version from paper

K1 format.
	- Original version from paper.
	- Improved version (reordering integrated)

AXT format.
	- Uncompacted with tileHeight = 1
	- Uncompacted with tileHeight > 1
	- Compacted with tileHeight = 1 (there is a bug)
	
	
The AXT software is part of the AXT_SPL library registered under a GPL license. The Universidad de Santiago de Compostela partially owns the intellectual rights.

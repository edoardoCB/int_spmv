




#ifndef __INTEL_SPMV_HEADER__
#define __INTEL_SPMV_HEADER__



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include <mkl.h>
#include <mkl_spblas.h>
#ifdef _OMP_
	#include <omp.h>
#endif



#ifndef FP_FLOAT
	#define FP_FLOAT  1
#endif



#ifndef FP_DOUBLE
	#define FP_DOUBLE 2
#endif



#if FP_TYPE == FP_FLOAT
	typedef float  FPT;
	char fptMsg[6] = "float";
#endif



#if FP_TYPE == FP_DOUBLE
	typedef double FPT;
	char fptMsg[7] = "double";
#endif



#ifndef UIN
	typedef unsigned int UIN;
#endif



#ifndef HDL
	#define HDL { fflush(stdout); printf( "---------------------------------------------------------------------------------------------------------\n" ); fflush(stdout); }
#endif



#ifndef BM
	#define BM { fflush(stdout); printf( "\nFile: %s    Line: %d.\n", __FILE__, __LINE__ ); fflush(stdout); }
#endif



#ifndef NUM_ITE
	#define NUM_ITE 250
#endif



#ifndef HBRICK_SIZE
	#define HBRICK_SIZE 8
#endif



#ifndef CHUNK_SIZE
	#define CHUNK_SIZE 8
#endif



typedef struct { char matFileName[48]; UIN cudaBlockSize; UIN ompMaxThreads; } str_inputArgs;



static str_inputArgs checkArgs( const UIN argc, char ** argv )
{
	if ( argc < 2 )
	{
		fflush(stdout);
		printf( "\n\tMissing input arguments.\n" );
		printf( "\n\tUsage:\n\n\t\t%s <matFileName>\n\n", argv[0] );
		printf( "\t\t\t<matFileName>:   file's name that contains the matrix in CSR format [string].\n" );
		fflush(stdout);
		exit( EXIT_FAILURE );
	}
	str_inputArgs sia;
	strcpy( sia.matFileName, argv[1] );
	#ifdef _OMP_
		#pragma omp parallel
		{
			#pragma omp master
				sia.ompMaxThreads = omp_get_max_threads();
		}
	#else
		sia.ompMaxThreads = 1;
	#endif
	return( sia );
}



static  void printRunSettings( const str_inputArgs sia )
{
	HDL; printf( "run settings\n" ); HDL;
	#ifdef _CIRRUS_
	printf( "hostname:           %s\n", "cirrus.EPCC" );
	#endif
	#ifdef _KAY_
	printf( "hostname:           %s\n", "kay.ICHEC" );
	#endif
	#ifdef _CTGPGPU2_
	printf( "hostname:           %s\n", "ctgpgpu2.CITIUS" );
	#endif
	printf( "srcFileName:        %s\n", __FILE__ );
	printf( "matFileName:        %s\n", sia.matFileName );
	#ifdef _OMP_
	printf( "ompMaxThreads:      %d\n", sia.ompMaxThreads );
	#endif
	printf( "FPT:                %s\n", fptMsg );
	printf( "sizeof(FPT):        %zu bytes\n", sizeof(FPT) );
	printf( "NUM_ITE:            %d\n", (UIN) NUM_ITE );
	printf( "HBRICK_SIZE:        %d\n", (UIN) HBRICK_SIZE );
	printf( "CHUNK_SIZE:         %d\n", (UIN) CHUNK_SIZE ); fflush(stdout);
	return;
}



#ifndef TEST_POINTER
	#define TEST_POINTER( p ) { if ( p == NULL ) { fflush(stdout); printf( "\nFile: %s Line: %d Pointer: %s is null\n", __FILE__, __LINE__, #p ); fflush(stdout); exit( EXIT_FAILURE ); } }
#endif



#ifndef ABORT
	#define ABORT { fflush(stdout); printf( "\nFile: %s Line: %d execution is aborted.\n", __FILE__, __LINE__ ); fflush(stdout); exit( EXIT_FAILURE ); }
#endif



typedef struct { UIN nrows; UIN nnz; UIN rmin; FPT rave; UIN rmax; FPT rsd; UIN bw; FPT * val; UIN * row; UIN * rowStart; UIN * rowEnd; UIN * col; UIN * rl; } str_matCSR;



static str_matCSR matrixReading( const char * matFileName )
{
	str_matCSR matCSR;
	if ( strstr( matFileName, ".csr" ) != NULL )
	{
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .csr file\n\n" );
			exit( EXIT_FAILURE );
		}
		if ( fscanf( fh, "%d %d", &(matCSR.nrows), &(matCSR.nnz) ) != 2 ) ABORT;
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		int i;
		for ( i = 0; i < ( matCSR.nnz ); i++ )
		{
			#if FP_TYPE == FPT_FLOAT
				if ( fscanf( fh, "%f %d\n",  &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#else
				if ( fscanf( fh, "%lf %d\n", &( matCSR.val[i] ), &( matCSR.col[i] ) ) != 2 ) ABORT;
			#endif
		}
		for ( i = 0; i < ( matCSR.nrows + 1 ); i++ )
			if ( fscanf( fh, "%d", &(matCSR.row[i]) ) != 1 ) ABORT;
		fclose( fh );
	}
	else if ( strstr( matFileName, ".bin" ) != NULL )
	{
		size_t aux = 0;
		FILE * fh;
		fh = fopen( matFileName, "r" );
		if ( fh == NULL )
		{
			printf( "\nmatrixReading is unable to open .bin file\n\n" );
			exit( EXIT_FAILURE );
		}
		aux = fread( &(matCSR.nrows), sizeof(UIN), 1, fh );
		aux = fread( &(matCSR.nnz),   sizeof(UIN), 1, fh );
		matCSR.val = (FPT *) malloc(   matCSR.nnz        * sizeof(FPT) ); TEST_POINTER( matCSR.val );
		matCSR.col = (UIN *) malloc(   matCSR.nnz        * sizeof(UIN) ); TEST_POINTER( matCSR.col );
		matCSR.row = (UIN *) malloc( ( matCSR.nrows + 1) * sizeof(UIN) ); TEST_POINTER( matCSR.row );
		matCSR.rl  = (UIN *) malloc(   matCSR.nrows      * sizeof(UIN) ); TEST_POINTER( matCSR.rl  );
		aux = fread( matCSR.val, sizeof(FPT),   matCSR.nnz,         fh );
		aux = fread( matCSR.col, sizeof(UIN),   matCSR.nnz,         fh );
		aux = fread( matCSR.row, sizeof(UIN), ( matCSR.nrows + 1 ), fh );
		aux++;
		fclose(fh);
	}
	else
	{
		char buffer[128];
		strcpy( buffer, "matrixReading detected that " );
		strcat( buffer, matFileName );
		strcat( buffer, " has NOT .csr or .bin extension" );
		printf( "\n%s\n\n", buffer );
		exit( EXIT_FAILURE );
	}
	return( matCSR );
}



static  void printMatrixStats( const char * matFileName, str_matCSR * matCSR )
{
	UIN    i, rl, rmin = 1e9, rmax = 0, j, bw = 0;
	int    dif;
	double rave1 = 0.0, rave2 = 0.0, rsd = 0.0;
	for ( i = 0; i < matCSR->nrows; i++ )
	{
		rl            = matCSR->row[i + 1] - matCSR->row[i];
		matCSR->rl[i] = rl;
		rave1         = rave1 +   rl;
		rave2         = rave2 + ( rl * rl );
		rmin          = (rmin<rl) ? rmin : rl;
		rmax          = (rmax>rl) ? rmax : rl;
		for ( j = matCSR->row[i]; j < matCSR->row[i+1]; j++ )
		{
			dif = abs( ((int) i) - ((int) matCSR->col[j]) );
			bw  = ( dif > bw ) ? dif : bw ;
		}
	}
	rave1 = rave1 / (double) (matCSR->nrows);
	rave2 = rave2 / (double) (matCSR->nrows);
	rsd   = sqrt( rave2 - ( rave1 * rave1 ) );
	matCSR->rmin = rmin;
	matCSR->rave = rave1;
	matCSR->rmax = rmax;
	matCSR->rsd  = rsd;
	matCSR->bw   = bw;
	char name[64];
	strcpy( name, matFileName );
	char * token1;
	const char deli[2] = ".";
	token1 = strtok( name, deli );
	strcat( token1, ".sta" );
	FILE * fh;
	fh = fopen( name, "w+" );
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "matrix's statistics\n");
	fprintf( fh, "------------------------------------\n");
	fprintf( fh, "name:  %28s\n",    matFileName );
	fprintf( fh, "nrows: %28d\n",    matCSR->nrows );
	fprintf( fh, "nnz:   %28d\n",    matCSR->nnz );
	fprintf( fh, "rmin:  %28d\n",    matCSR->rmin );
	fprintf( fh, "rave:  %28.2lf\n", matCSR->rave );
	fprintf( fh, "rmax:  %28d\n",    matCSR->rmax );
	fprintf( fh, "rsd:   %28.2lf\n", matCSR->rsd );
	fprintf( fh, "rsdp:  %28.2lf\n", ( ( rsd / rave1 ) * 100 ) );
	fprintf( fh, "bw:    %28d\n",    matCSR->bw );
	fclose( fh );
	return;
}



typedef struct { char name[24]; double mfp; double beta; double ct; } str_formatData;



#ifndef GT
	#define GT( t ) { gettimeofday( &t, NULL ); }
#endif



static  double measureTime( const struct timeval t2, const struct timeval t1 )
{
	double t = (double) ( t2.tv_sec - t1.tv_sec ) + ( (double) ( t2.tv_usec - t1.tv_usec ) ) * 1e-6;
	return( t );
}



static str_formatData getFormatDataCSR( str_matCSR * matCSR )
{
	// thes
	UIN i, ii;
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	matCSR->rowStart = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowStart );
	matCSR->rowEnd   = (UIN *) calloc( matCSR->nrows, sizeof(UIN) ); TEST_POINTER( matCSR->rowEnd   );
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		for ( ii = 0; ii < matCSR->nrows; ii++ )
		{
			matCSR->rowStart[ii] = matCSR->row[ii];
			matCSR->rowEnd[ii]   = matCSR->row[ii+1];
		}
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// format's name
	str_formatData fd;
	strcpy( fd.name, "CSR" );
	// CSR memory footprint
	fd.mfp =          (double) (   matCSR->nnz         * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) (   matCSR->nnz         * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( ( matCSR->nrows + 1 ) * sizeof(UIN) ); // row
	fd.mfp = fd.mfp + (double) (   matCSR->nrows       * sizeof(FPT) ); // vec
	// CSR occupancy ( beta )
	fd.beta = ( (double) matCSR->nnz / (double) matCSR->nnz );
	// CSR conversion time
	fd.ct = tt;
	return( fd );
}



static  void initVec( const UIN len, FPT * vec )
{
	UIN i;
	for ( i = 0 ; i < len; i++ )
		vec[i] = (FPT) i;
	return;
}



static void fill_array( const UIN ompNT, const UIN len, FPT val, FPT * array )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( i = 0; i < len; i++ )
		array[i] = val;
	return;
}



static  void cf_CSR( const str_matCSR matCSR, const FPT * vec, FPT * res )
{
	UIN i, j;
	FPT aux;
	for ( i = 0; i < matCSR.nrows; i++ )
	{
		aux = (FPT) 0;
		for ( j = matCSR.row[i]; j < matCSR.row[i+1]; j++ )
		{
			aux = aux + matCSR.val[j] * vec[matCSR.col[j]];
		}
		res[i] = aux;
	}
	return;
}



typedef struct { double aErr; double rErr; UIN pos; } str_err;



static  void getErrors( const UIN len, const FPT * ar, const FPT * ac, str_err * sErr )
{
	double dif, maxDif = 0.0;
	double val, maxVal = 0.0;
	UIN pos = 0;
	UIN i;
	for ( i = 0; i < len; i++ )
	{
		val = fabs(ar[i]);
		if ( val > maxVal ) maxVal = val;
		dif = fabs( fabs(ac[i]) - val );
		if ( dif > maxDif )
		{
			maxDif = dif;
			pos    = i;
		}
	}
	sErr->aErr = maxDif;
	sErr->rErr = maxDif/maxVal;
	sErr->pos  = pos;
	return;
}



typedef struct { char name[25]; double et; double ot; double flops; str_err sErr; } str_res;



static str_res test_cf_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		cf_CSR( matCSR, vec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "cf_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) matCSR.nnz ) ) / sr.et;
	getErrors( matCSR.nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



static str_res test_mkl_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	//
	     char transa = 'N';
	const UIN nrows     = matCSR.nrows;
	const UIN nnz       = matCSR.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		mkl_cspblas_dcsrgemv( &transa , &nrows , matCSR.val , matCSR.row , matCSR.col , vec , res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "mkl_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



#ifndef TEST_MKL_FUNCTION
	#define TEST_MKL_FUNCTION( f ) { if (f!=SPARSE_STATUS_SUCCESS) { fflush(stdout); printf( "[MKL Error] function: %s\tfile: %s\tline: %d\n", #f, __FILE__, __LINE__ ); fflush(stdout); exit(EXIT_FAILURE);} }
#endif



static str_res test_sp_mkl_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	//
	const UIN nrows     = matCSR.nrows;
	const UIN nnz       = matCSR.nnz;
	const double one    = 1.0;
	const double zero   = 0.0;
	struct matrix_descr matrixDescriptor; matrixDescriptor.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t   matrixHandle;
	TEST_MKL_FUNCTION( mkl_sparse_d_create_csr( &matrixHandle, SPARSE_INDEX_BASE_ZERO, nrows, nrows, matCSR.rowStart, matCSR.rowEnd, matCSR.col, matCSR.val ) );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		TEST_MKL_FUNCTION( mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, one, matrixHandle, matrixDescriptor, vec, zero, res ) );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "sp_mkl_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



static str_res test_sp_insp_mkl_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	//
	const UIN nrows     = matCSR.nrows;
	const UIN nnz       = matCSR.nnz;
	const double one    = 1.0;
	const double zero   = 0.0;
	struct matrix_descr matrixDescriptor; matrixDescriptor.type = SPARSE_MATRIX_TYPE_GENERAL;
	sparse_matrix_t   matrixHandle;
	TEST_MKL_FUNCTION( mkl_sparse_d_create_csr( &matrixHandle, SPARSE_INDEX_BASE_ZERO, nrows, nrows, matCSR.rowStart, matCSR.rowEnd, matCSR.col, matCSR.val ) );
	TEST_MKL_FUNCTION( mkl_sparse_set_mv_hint( matrixHandle, SPARSE_OPERATION_NON_TRANSPOSE, matrixDescriptor, (MKL_INT) NUM_ITE ) );
	TEST_MKL_FUNCTION( mkl_sparse_optimize( matrixHandle) );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		TEST_MKL_FUNCTION( mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, one, matrixHandle, matrixDescriptor, vec, zero, res ) );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "sp_insp_mkl_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



/*
static str_res test_sp_insp_sym_mkl_CSR( const str_matCSR matCSR, const FPT * vec, const FPT * ref )
{
	//
	const UIN nrows     = matCSR.nrows;
	const UIN nnz       = matCSR.nnz;
	const double one    = 1.0;
	const double zero   = 0.0;
	struct matrix_descr matrixDescriptor; matrixDescriptor.type = SPARSE_MATRIX_TYPE_SYMMETRIC;
	                                      matrixDescriptor.mode = SPARSE_FILL_MODE_UPPER;
	                                      matrixDescriptor.diag = SPARSE_DIAG_NON_UNIT;
	sparse_matrix_t   matrixHandle;
	TEST_MKL_FUNCTION( mkl_sparse_d_create_csr( &matrixHandle, SPARSE_INDEX_BASE_ZERO, nrows, nrows, matCSR.rowStart, matCSR.rowEnd, matCSR.col, matCSR.val ) );
	TEST_MKL_FUNCTION( mkl_sparse_set_mv_hint( matrixHandle, SPARSE_OPERATION_NON_TRANSPOSE, matrixDescriptor, (MKL_INT) NUM_ITE ) );
	TEST_MKL_FUNCTION( mkl_sparse_optimize( matrixHandle) );
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		TEST_MKL_FUNCTION( mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, one, matrixHandle, matrixDescriptor, vec, zero, res ) );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "sp_insp_sym_mkl_CSR" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}
*/



typedef struct { UIN nrows; UIN nnz; UIN lenAX; UIN lenBRP; FPT * ax; UIN * brp; } str_matAXC;



static  UIN getArrayBrpAXC( const str_matCSR matCSR, str_matAXC * matAXC )
{
	UIN rowID = 0, brickNum = 0;
	for ( rowID = 0; rowID < matAXC->nrows; rowID++ )
	{
		brickNum                 = ( matCSR.rl[rowID] + HBRICK_SIZE - 1 ) / HBRICK_SIZE;
		matAXC->brp[rowID + 1]   = matAXC->brp[rowID]  + ( 2 * brickNum * HBRICK_SIZE );
	}
	return( matAXC->brp[matAXC->nrows] );
}



static  void getArrayAxAXC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	const UIN nrows = matAXC->nrows;
	UIN rowID, posAX, counter, posCSR;
	#pragma omp parallel for default(shared) private(rowID,posAX,counter,posCSR) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		  posAX = matAXC->brp[rowID];
		counter = 0;
		for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID + 1]; posCSR++ )
		{
			              matAXC->ax[posAX] = matCSR.val[posCSR];
			matAXC->ax[posAX + HBRICK_SIZE] = vec[matCSR.col[posCSR]];
			if ( counter == (HBRICK_SIZE - 1) )
			{
				posAX  = posAX + 1 + HBRICK_SIZE;
				counter = 0;
			}
			else
			{
				posAX++;
				counter++;
			}
		}
	}
	return;
}



static str_formatData getFormatDataAXC( const UIN ompNumThreads, const str_matCSR matCSR, const FPT * vec, str_matAXC * matAXC )
{
	// get AXC parameters
	 matAXC->nrows = matCSR.nrows;
	   matAXC->nnz = matCSR.nnz;
	matAXC->lenBRP = matCSR.nrows + 1;
	   matAXC->brp = (UIN *) calloc( matAXC->lenBRP, sizeof(UIN) ); TEST_POINTER( matAXC->brp  );
	// get matAXC
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matAXC->lenAX = getArrayBrpAXC( matCSR, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matAXC->ax = (FPT *) mkl_malloc( matAXC->lenAX * sizeof(FPT), 64 ); TEST_POINTER( matAXC->ax );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayAxAXC( ompNumThreads, matCSR, vec, matAXC );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "AXC" );
	// AXC memory footprint
	fd.mfp =          (double) ( matAXC->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXC->lenBRP * sizeof(UIN) ); // brp ( stores the starting address of a row )
	// AXC occupancy ( beta )
	fd.beta = ( (double) matAXC->nnz / (double) (matAXC->lenAX >> 1) );
	// AXC conversion time
	fd.ct = tc;
	return( fd );
}



static void int_axc( const UIN ompNT, const UIN nrows, const FPT * ax, const UIN * brp, FPT * y )
{
	const UIN stride = 2 * HBRICK_SIZE;
	      UIN rowID, posAX;
	      FPT red, sum;
	  __m512d val, vec, pro;
	#pragma omp parallel for default(shared) private(rowID,posAX,val,vec,pro,red,sum) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		sum =  0;
		for ( posAX = brp[rowID]; posAX < brp[rowID+1]; posAX = posAX + stride )
		{
			val = _mm512_load_pd( &ax[posAX]               );
			vec = _mm512_load_pd( &ax[posAX + HBRICK_SIZE] );
			pro = _mm512_mul_pd( val, vec );
			red = _mm512_reduce_add_pd( pro );
			sum = sum + red;
		}
		y[rowID] = sum;
	}
	return;
}



static str_res test_int_AXC( const UIN ompNT, const str_matAXC matAXC, const FPT * ref )
{
	//
	const UIN nrows =  matAXC.nrows;
	const UIN nnz   =  matAXC.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		int_axc( ompNT, nrows, matAXC.ax, matAXC.brp, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "int_AXC" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



typedef struct { UIN ind; UIN val; } str_pair;



typedef struct { UIN nrows; UIN nnz; UIN chunkNum; UIN lenVC; UIN * permi; UIN * nmc; UIN * chp; FPT * val; UIN * col; } str_matK1;



static int orderFunction( const void * ele1, const void * ele2 )
{
	return (  ( (str_pair *) ele2 )->val - ( (str_pair *) ele1 )->val  );
}



static void getArrayPermiK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	str_pair * list = (str_pair *) malloc( matCSR.nrows * sizeof(str_pair) ); TEST_POINTER( list );
	UIN i;
	for ( i = 0; i < matK1->nrows; i++ )
	{
		list[i].ind = i;
		list[i].val = matCSR.rl[i];
	}
	qsort( list, matK1->nrows, sizeof(str_pair), orderFunction );
	for ( i = 0; i < matK1->nrows; i++ )
		matK1->permi[i] = list[i].ind;
	free( list );
	return;
}



static UIN getArraysNmcChpK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	UIN i, p, n, l = 0, chunkNum = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	for ( i = 0 ; i < chunkNum; i++ )
	{
		p             = matK1->permi[i * CHUNK_SIZE];
		n             = matCSR.rl[p];
		matK1->nmc[i] = n;
		l             = l + CHUNK_SIZE * n;
	}
	for ( i = 1; i < matK1->chunkNum; i++ )
		matK1->chp[i] = matK1->chp[i-1] + ( matK1->nmc[i-1] * CHUNK_SIZE );
	return l;
}



static void getArraysValColK1( const str_matCSR matCSR, str_matK1 * matK1 )
{
	const UIN chunkNum = matK1->chunkNum;
	UIN chunkID, rid, row, posCSR, rowOff, posK1;
	for ( chunkID = 0; chunkID < chunkNum; chunkID++ )
	{
		for ( rid = 0; rid < CHUNK_SIZE; rid++ )
		{
			row = chunkID * CHUNK_SIZE + rid;
			if ( row == matCSR.nrows ) return;
			row = matK1->permi[row];
			for ( posCSR = matCSR.row[row], rowOff = 0; posCSR < matCSR.row[row + 1]; posCSR++, rowOff++ )
			{
				posK1             = matK1->chp[chunkID] + rowOff * CHUNK_SIZE + rid;
				matK1->val[posK1] = matCSR.val[posCSR];
				matK1->col[posK1] = matCSR.col[posCSR];
			}
		}
	}
	return;
}



static str_formatData getFormatDataK1( const UIN blockSize, const str_matCSR matCSR, const FPT * vec, str_matK1 * matK1 )
{
	// get K1 parameters
	matK1->nrows     = matCSR.nrows;
	matK1->nnz       = matCSR.nnz;
	matK1->chunkNum  = ( matCSR.nrows + CHUNK_SIZE - 1 ) / CHUNK_SIZE;
	matK1->permi     = (UIN *) _mm_malloc( ( matK1->chunkNum + 1 ) * CHUNK_SIZE * sizeof(UIN), 64 ); TEST_POINTER( matK1->permi );
	matK1->nmc       = (UIN *) calloc( matK1->chunkNum,   sizeof(UIN) ); TEST_POINTER( matK1->nmc   );
	matK1->chp       = (UIN *) calloc( matK1->chunkNum,   sizeof(UIN) ); TEST_POINTER( matK1->chp   );
	UIN i;
	for ( i = 0; i < ( matK1->chunkNum + 1 ) * CHUNK_SIZE; i++ )
		matK1->permi[i] = 0;
	// get matK1
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArrayPermiK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		matK1->lenVC = getArraysNmcChpK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	matK1->val = (FPT *) _mm_malloc( matK1->lenVC * sizeof(FPT), 64 ); TEST_POINTER( matK1->val );
	matK1->col = (UIN *) _mm_malloc( matK1->lenVC * sizeof(UIN), 64 ); TEST_POINTER( matK1->col );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysValColK1( matCSR, matK1 );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// format's name
	str_formatData fd;
	strcpy( fd.name, "K1" );
	// K1 memory footprint
	fd.mfp =          (double) ( matK1->chunkNum * sizeof(UIN) ); // nmc
	fd.mfp = fd.mfp + (double) ( matK1->chunkNum * sizeof(UIN) ); // chp
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(FPT) ); // val
	fd.mfp = fd.mfp + (double) ( matK1->lenVC    * sizeof(UIN) ); // col
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(UIN) ); // permi
	fd.mfp = fd.mfp + (double) ( matK1->nrows    * sizeof(FPT) ); // vec
	// K1 occupancy ( beta )
	fd.beta = ( (double) matK1->nnz / (double) (matK1->lenVC) );
	// K1 conversion time
	fd.ct = tc;
	return( fd );
}





static void print__m512d( char * varName, __m512d var )
{
	const UIN len = 8;
	      UIN i;
	      double * arr = _mm_malloc( len * sizeof(double), 64 ); TEST_POINTER( arr );
	_mm512_store_pd( arr, var );
	printf( "%s = [ ", varName );
	for ( i = 0; i < len; i++ )
		printf( "%8.2lf ", arr[i] );
	printf( "]\n" );
	_mm_free( arr );
	return;
}



static void print__m512i( char * varName, __m512i var )
{
	const UIN len = 16;
	      UIN i;
	      UIN * arr = _mm_malloc( len * sizeof(UIN), 64 ); TEST_POINTER( arr );
	_mm512_store_epi32( arr, var );
	printf( "%s = [ ", varName );
	for ( i = 0; i < len; i++ )
		printf( "%3d ", arr[i] );
	printf( "]\n" );
	_mm_free( arr );
	return;
}



static void int_k1( const UIN ompNT, const UIN chunkNum, const FPT * val, const UIN * col, const UIN * chp, const UIN * nmc, const FPT * vec, FPT * res )
{
	UIN chunkID, colID, offset;
	__m512d vtVal, vtVec, vtPro, vtSum;
	__m512i vtCol;
	#pragma omp parallel for default(shared) private(chunkID,colID,offset,vtSum,vtVal,vtCol,vtVec,vtPro) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( chunkID = 0; chunkID < chunkNum; chunkID++ )
	{
		vtSum = _mm512_setzero_pd( );
		offset = chp[chunkID];
		for ( colID = 0; colID < nmc[chunkID]; colID++ )
		{
			vtVal = _mm512_load_pd( &val[offset] );
			vtCol = _mm512_load_epi32( &col[offset] );
			vtVec = _mm512_i32logather_pd( vtCol, vec, 8 );
			vtPro = _mm512_mul_pd( vtVal, vtVec );
			vtSum = _mm512_add_pd( vtSum, vtPro );
			offset = offset + 8;
		}
		_mm512_store_pd( &res[chunkID<<3], vtSum );
	}
	return;
}



static void orderArrayK1( const UIN ompNT, const UIN len, const UIN * permi, const FPT * arrDis, FPT * arrOrd )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( i = 0; i < len; i++ )
		arrOrd[permi[i]] = arrDis[i];
	return;
}



static str_res test_int_k1( const UIN ompNT, const str_matK1 matK1, const FPT * vec, const FPT * ref )
{
	//
	const UIN chunkNum = matK1.chunkNum;
	const UIN   resLen = chunkNum * CHUNK_SIZE;
	const UIN    nrows = matK1.nrows;
	const UIN      nnz = matK1.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0, te = 0.0, to = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( resLen, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		int_k1( ompNT, chunkNum, matK1.val, matK1.col, matK1.chp, matK1.nmc, vec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	te = tt;
	FPT * resOrd = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( resOrd );
	tt = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		orderArrayK1( ompNT, nrows, matK1.permi, res, resOrd );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	to = tt;
	// store results
	str_res sr;
	strcpy( sr.name, "int_k1" );
	sr.et    = te / (double) NUM_ITE;
	sr.ot    = to / (double) NUM_ITE;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, resOrd, &(sr.sErr) );
	free( res );
	free( resOrd );
	return( sr );
}



static void int_k1p( const UIN ompNT, const __mmask8 mask, const UIN chunkNum, const FPT * val, const UIN * col, const UIN * chp, const UIN * nmc, const UIN * permi, const FPT * vec, FPT * res )
{
	UIN chunkID, colID, offset;
	__m512d vtVal, vtVec, vtPro, vtSum;
	__m512i vtCol, vtRow;
	#pragma omp parallel for default(shared) private(chunkID,colID,offset,vtSum,vtVal,vtCol,vtVec,vtPro) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( chunkID = 0; chunkID < (chunkNum-1); chunkID++ )
	{
		vtSum = _mm512_setzero_pd();
		offset = chp[chunkID];
		for ( colID = 0; colID < nmc[chunkID]; colID++ )
		{
			vtVal = _mm512_load_pd( &val[offset] );
			vtCol = _mm512_load_epi32( &col[offset] );
			vtVec = _mm512_i32logather_pd( vtCol, vec, 8 );
			vtPro = _mm512_mul_pd( vtVal, vtVec );
			vtSum = _mm512_add_pd( vtSum, vtPro );
			offset = offset + 8;
		}
		vtRow = _mm512_load_epi32( &permi[chunkID*8] );
		//print__m512d( "vtSum", vtSum );
		//print__m512i( "vtRow", vtRow );
		_mm512_i32loscatter_pd( res, vtRow, vtSum, 8 );
	}
	#pragma omp parallel for default(shared) private(chunkID,colID,offset,vtSum,vtVal,vtCol,vtVec,vtPro) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( chunkID = (chunkNum - 1); chunkID < chunkNum; chunkID++ )
	{
		vtSum = _mm512_setzero_pd();
		offset = chp[chunkID];
		for ( colID = 0; colID < nmc[chunkID]; colID++ )
		{
			vtVal = _mm512_load_pd( &val[offset] );
			vtCol = _mm512_load_epi32( &col[offset] );
			vtVec = _mm512_i32logather_pd( vtCol, vec, 8 );
			vtPro = _mm512_mul_pd( vtVal, vtVec );
			vtSum = _mm512_add_pd( vtSum, vtPro );
			offset = offset + 8;
		}
		vtRow = _mm512_load_epi32( &permi[chunkID*8] );
		_mm512_mask_i32loscatter_pd( res, mask, vtRow, vtSum, 8 );
	}
	return;
}



static str_res test_int_k1p( const UIN ompNT, const str_matK1 matK1, const FPT * vec, const FPT * ref )
{
	//
	const UIN chunkNum = matK1.chunkNum;
	const UIN    nrows = matK1.nrows;
	const UIN      nnz = matK1.nnz;
	const UIN   resLen = (nrows + 1);
	const UIN   posDif = ( chunkNum * CHUNK_SIZE ) - nrows;
	__mmask8 mask;
	if (posDif == 0) mask = 0xFF;
	if (posDif == 1) mask = 0x7F;
	if (posDif == 2) mask = 0x3F;
	if (posDif == 3) mask = 0x1F;
	if (posDif == 4) mask = 0x0F;
	if (posDif == 5) mask = 0x07;
	if (posDif == 6) mask = 0x03;
	if (posDif == 7) mask = 0x01;
	// timed iterations
	double ti = 0.0, tt = 0.0, te = 0.0, to = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( resLen, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		int_k1p( ompNT, mask, chunkNum, matK1.val, matK1.col, matK1.chp, matK1.nmc, matK1.permi, vec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	te = tt;
	// store results
	str_res sr;
	strcpy( sr.name, "int_k1p" );
	sr.et    = te / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}



typedef struct{ UIN nrows; UIN nnz; char mode[8]; UIN tileHW; UIN tileH; UIN logTH; UIN tileN; UIN lenAX; UIN lenSEC; UIN lenCON; UIN log; FPT * ax; UIN * sec; UIN * con; } str_matAXT;



static void getArraysLenAXT( const str_matCSR matCSR, str_matAXT * matAXT )
{
	const UIN  nrows = matAXT->nrows;
	const UIN    thw = matAXT->tileHW;
	const UIN     th = matAXT->tileH;
	const UIN    ths = thw * th;
	const UIN grpLen = (th == 1) ? (thw) : (th) ;
	char mode[8];
	strcpy( mode, matAXT->mode );
	      UIN rowID = 0, rowStartPos = 0, rowOffT, rowOffR, rowOffC, pos, rowLen, positions, columns, totalColumns = 0, totalTiles;
	for ( ; rowID < nrows; rowID++ )
	{
		           rowOffT = ( (rowStartPos + ths)/ths ) - 1;
		           rowOffR =    rowStartPos % th;
		           rowOffC = ( (rowStartPos + th)/th ) - 1 - (rowOffT * thw);
		               pos = rowOffT * (2 * ths) + rowOffR * (2 * thw) + rowOffC;
		matAXT->con[rowID] = pos;
		            rowLen = matCSR.rl[rowID];
		         positions = ( strcmp( mode, "NOC" ) == 0 ) ? ( ( ( rowLen + grpLen - 1 ) / grpLen ) * grpLen ) : ( rowLen ) ;
		           columns = ( positions + th - 1 ) / th;
		      totalColumns = totalColumns + columns;
		       rowStartPos = rowStartPos + positions;
	}
	     totalTiles = ( totalColumns + thw - 1 ) / thw;
	 matAXT->tileN = totalTiles;
	 matAXT->lenAX = totalTiles * 2 * ths;
	if      ( (strcmp(mode, "NOC")==0) && (th==1) ) matAXT->lenSEC = totalTiles;
	else if ( (strcmp(mode, "NOC")==0) && (th!=1) ) matAXT->lenSEC = totalTiles * thw;
	else                                            matAXT->lenSEC = totalTiles * ths;
	return;
}



static void getArraysAxSecAXT_NOC_H1( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths));
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX++;
				ctrEle++;
				if ((ctrEle%thw)==0)
				{
					posAX = posAX + thw;
					posSEC++;
				}
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_NOC( const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	const UIN    th = matAXT->tileH;
	const UIN   ths = thw * th;
	      UIN rowID, rowLen, posAX, posSEC, posCSR, ctrEle;
	#pragma omp parallel for default(shared) private(rowID,rowLen,posAX,posSEC,posCSR,ctrEle) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			posAX  = matAXT->con[rowID];
			posSEC = (posAX/(2*ths))*thw + posAX%thw;
			ctrEle = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				matAXT->sec[posSEC]   = rowID;
				posAX                 = posAX  + 2 * thw;
				ctrEle++;
				if ((ctrEle%th) == 0)
				{
					posAX = posAX + 1 - (2 * th * thw);
					posSEC++;
					if (posAX%thw==0) posAX = posAX + ((2*th)-1) * thw;
				}
			}
		}
	}
	return;
}



static void getArraysAxSecAXT_COM_H1( const UIN bs, const UIN log, const UIN ompNT, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	const UIN nrows = matAXT->nrows;
	const UIN   thw = matAXT->tileHW;
	      UIN rowID, rowLen, eleCtr, posCSR, bid, bco, tid, tco, posAX, posSEC, posBLK, q1, q2, offset, blk;
	#pragma omp parallel for default(shared) private(rowID,rowLen,eleCtr,posCSR,bid,bco,tid,tco,posAX,posSEC,posBLK,q1,q2,offset) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( rowID = 0; rowID < nrows; rowID++ )
	{
		rowLen = matCSR.rl[rowID];
		if (rowLen>0)
		{
			eleCtr = 0;
			for ( posCSR = matCSR.row[rowID]; posCSR < matCSR.row[rowID+1]; posCSR++ )
			{
				bid                   = ((posCSR+bs)/bs)-1;
				bco                   =   posCSR%bs;
				tid                   = ((posCSR+thw)/thw)-1;
				tco                   =  posCSR%thw;
				posAX                 = tid * 2 * thw + tco;
				posSEC                = tid     * thw + tco;
				posBLK                = bid     * bs  + bco;
				matAXT->ax[posAX]     = matCSR.val[posCSR];
				matAXT->ax[posAX+thw] = vec[matCSR.col[posCSR]];
				if ( (eleCtr==0) || (bco==0))
				{
					q1     = rowLen - eleCtr - 1;
					q2     = bs - 1 - bco;
					offset = (q1 > q2) ? q2 : q1;
					matAXT->sec[posSEC] = rowID<<log | offset;
				}
				eleCtr++;
			}
		}
	}
	return;
}



static void fill_array_uin( const UIN ompNT, const UIN len, const UIN value, UIN * array )
{
	UIN i;
	#pragma omp parallel for default(shared) private(i) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for( i = 0; i < len; i++ )
	array[i] = value;
	return;
}



static str_formatData getFormatDataAXT( const UIN ompNT, const UIN bs, const UIN thw, const UIN th, const char * mode, const str_matCSR matCSR, const FPT * vec, str_matAXT * matAXT )
{
	// set AXT parameters
	matAXT->nrows  = matCSR.nrows;
	matAXT->nnz    = matCSR.nnz;
	matAXT->tileHW = thw;
	matAXT->tileH  = th;
	strcpy( matAXT->mode, mode );
	matAXT->lenCON = matCSR.nrows;
	   matAXT->con = (UIN *) calloc( matAXT->lenCON, sizeof(UIN) ); TEST_POINTER( matAXT->con );
	UIN i;
	for ( i = 0; i < 10; i++ )
		if ( ((matAXT->tileH) >> i) == 1 ) matAXT->logTH = i;
	// get AXT arrays' length
	struct timeval t1, t2;
	double ti = 0.0, tt = 0.0, tc = 0.0;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		GT( t1 );
		getArraysLenAXT( matCSR, matAXT );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;
	// get arrays ax[] and sec[]
	matAXT->ax  = (FPT *) _mm_malloc( matAXT->lenAX  * sizeof(FPT), 64 ); TEST_POINTER( matAXT->ax  );
	matAXT->sec = (UIN *) _mm_malloc( matAXT->lenSEC * sizeof(UIN), 64 ); TEST_POINTER( matAXT->sec );
	fill_array( ompNT, matAXT->lenAX,  0.0, matAXT->ax );
	fill_array_uin( ompNT, matAXT->lenSEC,   0, matAXT->sec );
	tt = 0.0;
	char buffer[128];
	if (strcmp(mode,"NOC")==0)
	{
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_NOC_H1( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
			strcpy( buffer, "AXT_NOC_H1_HW8" );
		}
		else
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_NOC( ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
			strcpy( buffer, "AXT_NOC_H" );
			char TH[3]; sprintf( TH, "%d", th );
			strcat( buffer, TH );
			strcat( buffer, "_HW8" );
		}
	}
	else
	{
		for ( i = 1; i < 10; i++ )
		{
			if ((bs>>i) == 1)
			{
				matAXT->log = i;
				break;
			}
		}
		if (th==1)
		{
			for ( i = 0; i < NUM_ITE; i++ )
			{
				GT( t1 );
				getArraysAxSecAXT_COM_H1( bs, matAXT->log, ompNT, matCSR, vec, matAXT );
				GT( t2 );
				ti = measureTime( t2, t1 );
				tt = tt + ti;
			}
			strcpy( buffer, "AXT_COM_H1_HW8_BS" );
			char BS[5]; sprintf( BS, "%d", bs );
			strcat( buffer, BS );
		}
	}
	ti = tt / (double) NUM_ITE;
	tc = tc + ti;





	// AXTC specific name
	str_formatData fd;
	strcpy( fd.name, buffer );
	// AXTC memory footprint
	fd.mfp =          (double) ( matAXT->lenAX  * sizeof(FPT) ); // ax
	fd.mfp = fd.mfp + (double) ( matAXT->lenSEC * sizeof(UIN) ); // sec
	// AXTC occupancy ( beta )
	fd.beta = ( (double) matAXT->nnz / (double) (matAXT->lenAX >> 1) );
	// AXTC conversion time
	fd.ct = tc;
	return( fd );
}



static void int_axt_noc_h1( const UIN ompNT, const UIN tileNum, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN stride = 2 * HBRICK_SIZE;
	      UIN tileID, posAX, rowID;
	      FPT red;
	  __m512d vtMat, vtVec, vtPro;
	#pragma omp parallel for default(shared) private(tileID,posAX,vtMat,vtVec,vtPro,red,rowID) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( tileID = 0; tileID < tileNum; tileID++ )
	{
		posAX    = tileID * stride;
		vtMat    = _mm512_load_pd( &ax[posAX]               );
		vtVec    = _mm512_load_pd( &ax[posAX + HBRICK_SIZE] );
		vtPro    = _mm512_mul_pd( vtMat, vtVec );
		red      = _mm512_reduce_add_pd( vtPro );
		rowID    = rwp[tileID];
		#pragma omp atomic
		y[rowID] = y[rowID] + red;
	}
	return;
}



static str_res test_int_axt_noc_h1( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN tileNum = matAXT.tileN;
	const UIN nrows   = matAXT.nrows;
	const UIN nnz     = matAXT.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) calloc( nrows, sizeof(FPT) ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_noc_h1( ompNT, tileNum, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	str_res sr;
	strcpy( sr.name, "int_axt_noc_h1_hw8" );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	free( res );
	return( sr );
}





static void int_axt_noc( const UIN ompNT, const UIN tileNum, const UIN tileHeight, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN stride   = 2 * HBRICK_SIZE;
	const UIN tileSize = tileHeight * stride;
	      UIN tileID, posAX, posRWP, rowID, i;
	      FPT red, tmp[HBRICK_SIZE];
	  __m512d vtMat, vtVec, vtPro, vtSum;
	  __m512i vtRid;
	#pragma omp parallel for default(shared) private(tileID,posAX,posRWP,vtMat,vtVec,vtPro,vtSum,tmp,i,rowID) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( tileID = 0; tileID < tileNum; tileID++ )
	{
		posAX = tileID * tileSize;
		vtSum = _mm512_setzero_pd();
		for ( posAX = tileID * tileSize; posAX < (tileID + 1) * tileSize; posAX = posAX + stride )
		{
			vtMat = _mm512_load_pd( &ax[posAX]               );
			vtVec = _mm512_load_pd( &ax[posAX + HBRICK_SIZE] );
			vtPro = _mm512_mul_pd( vtMat, vtVec );
			vtSum = _mm512_add_pd( vtSum, vtPro );
		}
		posRWP = tileID * HBRICK_SIZE;
		_mm512_store_pd( tmp, vtSum );
		#pragma omp parallel for default(shared) private(i,rowID) num_threads(HBRICK_SIZE) schedule(static) if(_OPENMP)
		for ( i = 0; i < HBRICK_SIZE; i++ )
		{
			rowID = rwp[posRWP+i];
			#pragma omp atomic
			y[rowID] = y[rowID] + tmp[i];
		}
	}
	return;
}



static str_res test_int_axt_noc( const UIN ompNT, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN tileNum    = matAXT.tileN;
	const UIN tileHeight = matAXT.tileH;
	const UIN nrows      = matAXT.nrows;
	const UIN nnz        = matAXT.nnz;
	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	FPT * res = (FPT *) _mm_malloc( nrows * sizeof(FPT), 64 ); TEST_POINTER( res );
	UIN i;
	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_noc( ompNT, tileNum, tileHeight, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}
	// store results
	char buffer[128];
	strcpy( buffer, "int_axt_noc_h" );
	char TH[3]; sprintf( TH,  "%d", tileHeight );
	strcat( buffer, TH );
	strcat( buffer, "_hw8" );
	str_res sr;
	strcpy( sr.name, buffer );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



static void int_axt_com_h1( const UIN ompNT, const UIN bn, const UIN bs, const UIN log, const UIN thw, const UIN tn, const UIN lenAX, const UIN lenRWP, const FPT * ax, const UIN * rwp, FPT * y )
{
	const UIN ts  = 2 * thw;
	const UIN tpb = bs / thw;
	const UIN lenBlk1 = bn * bs;
	const UIN lenBlkA = bn * tpb;
	      UIN posAX, posBLK, bid, tid, ro, r, o;
	      FPT v;
	  __m512d vtVal, vtRed, vtAcu;
	  __m512i vtId1  = _mm512_set_epi64( 6, 5, 4, 3, 2, 1, 0, 0 ); __mmask8 mask1 = 0xFE;
	  __m512i vtId2  = _mm512_set_epi64( 5, 4, 3, 2, 1, 0, 1, 0 ); __mmask8 mask2 = 0xFC;
	  __m512i vtId3  = _mm512_set_epi64( 3, 2, 1, 0, 3, 2, 1, 0 ); __mmask8 mask3 = 0xF0;
	      FPT * blk1 = _mm_malloc( lenBlk1 * sizeof(FPT), 64 );
	      FPT * blk2 = _mm_malloc( lenBlk1 * sizeof(FPT), 64 );
	      FPT * blkA = _mm_malloc( lenBlkA * sizeof(FPT), 64 );

	//#pragma omp parallel for default(shared) private(posAX,posBLK,tid,vtVal,vtRed) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( posAX = 0, posBLK = 0, tid = 0; posAX < lenAX; posAX = posAX + ts, posBLK = posBLK + thw, tid++ )
	{
		vtVal = _mm512_mul_pd( _mm512_load_pd(&ax[posAX]), _mm512_load_pd(&ax[posAX+thw]) );
		_mm512_store_pd( &blk1[posBLK], vtVal );
		vtRed = _mm512_mask_add_pd( vtVal, mask1, vtVal, _mm512_permutexvar_pd( vtId1, vtVal ) );
		vtRed = _mm512_mask_add_pd( vtRed, mask2, vtRed, _mm512_permutexvar_pd( vtId2, vtRed ) );
		vtRed = _mm512_mask_add_pd( vtRed, mask3, vtRed, _mm512_permutexvar_pd( vtId3, vtRed ) );
		_mm512_store_pd( &blk2[posBLK], vtRed );
		blkA[tid] = blk2[posBLK+thw-1];
	}

UIN i;
for ( i = 0; i < 256; i++ )
	printf( "pos:%3d  blk1:%17.1lf  blk2:%17.1lf\n", i, blk1[i], blk2[i] );
printf( "\n" );

for ( i = 0; i < 32; i++ )
	printf( "pos:%3d  blkA:%17.1lf", i, blkA[i] );

	//#pragma omp parallel for default(shared) private(bid,vtAcu,tid,vtVal,vtRed) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( bid = 0; bid < bn; bid++ )
	{
		vtAcu = _mm512_setzero_pd();
		for ( tid = 0; tid < tpb; tid = tid + thw )
		{
			vtVal = _mm512_load_pd( &blkA[bid*tpb+tid] );
			vtRed = _mm512_mask_add_pd( vtVal, mask1, vtVal, _mm512_permutexvar_pd( vtId1, vtVal ) );
			vtRed = _mm512_mask_add_pd( vtRed, mask2, vtRed, _mm512_permutexvar_pd( vtId2, vtRed ) );
			vtRed = _mm512_mask_add_pd( vtRed, mask3, vtRed, _mm512_permutexvar_pd( vtId3, vtRed ) );
			vtRed = _mm512_add_pd( vtRed, vtAcu );
			_mm512_store_pd( &blkA[bid*tpb+tid], vtRed );
			vtAcu = _mm512_set1_pd( blkA[bid*tpb+tid+thw-1] );
		}
	}

for ( i = 0; i < 32; i++ )
	printf( "pos:%3d  blkA:%17.1lf", i, blkA[i] );

	//#pragma omp parallel for default(shared) private(bid,vtAcu,tid,vtVal,vtRed) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( bid = 0; bid < bn; bid++ )
	{
		vtAcu = _mm512_setzero_pd();
		for ( tid = 0; tid < tpb; tid = tid + thw )
		{
			vtVal = _mm512_load_pd( &blk2[bid*tpb+tid] );
			vtRed = _mm512_add_pd( vtVal, vtAcu );
			if ( (bid==0) || (bid==1) ) print__m512d( "vtRed", vtRed);
			_mm512_store_pd( &blk2[bid*tpb+tid], vtRed );
			vtAcu = _mm512_set1_pd( blkA[bid*tpb+tid] );
		}
	}

for ( i = 0; i < 32; i++ )
	printf( "pos:%3d  blkA:%17.1lf", i, blkA[i] );

for ( i = 0; i < 256; i++ )
	printf( "pos:%3d  blk1:%17.1lf  blk2:%17.1lf\n", i, blk1[i], blk2[i] );

/*
	#pragma omp parallel for default(shared) private(tid,posBLK,vtVal,vtAcu,vtRed) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( tid = 1, posBLK = 8; tid < tn; tid++, posBLK = posBLK + thw )
	{
		vtVal = _mm512_load_pd( &blk2[posBLK] );
		vtAcu = _mm512_set1_pd( blkA[tid-1] );
		vtRed = _mm512_add_pd( vtVal, vtAcu );
		_mm512_store_pd( &blk2[posBLK], vtRed );
	}
*/




	#pragma omp parallel for default(shared) private(posBLK,ro,r,o,v) num_threads(ompNT) schedule(dynamic) if(_OPENMP)
	for ( posBLK = 0; posBLK < lenRWP; posBLK++ )
	{
		ro = rwp[posBLK];
		if (ro!=0)
		{
			r = ro >> log;
			o = ro & (bs-1);
			v = blk2[posBLK+o] - blk2[posBLK] + blk1[posBLK];
			if (r==5) printf( "r:%4d, posBLK:%3d, o:%3d, blk2o:%22.1lf, blk2:%22.1lf, blk1:%22.1lf, v:%22.1lf\n", r, posBLK, o, blk2[posBLK+o], blk2[posBLK], blk1[posBLK], v );
			#pragma omp atomic
			y[r] = y[r] + v;
		}
	}

	return;
}



static str_res test_int_axt_com_h1( const UIN ompNT, const UIN bs, const str_matAXT matAXT, const FPT * ref )
{
	//
	const UIN nrows  = matAXT.nrows;
	const UIN nnz    = matAXT.nnz;
	const UIN lenAX  = matAXT.lenAX;
	const UIN lenSEC = matAXT.lenSEC;
	const UIN log    = matAXT.log;
	const UIN tn     = matAXT.tileN;
	const UIN thw    = matAXT.tileHW;
	const UIN ts     = 2 * thw;
	const UIN bn     = ( (tn * HBRICK_SIZE) + bs - 1 ) / bs;
	const UIN tpb    = bs / HBRICK_SIZE;
	FPT * res = (FPT *) _mm_malloc( nrows   * sizeof(FPT), 64 ); TEST_POINTER( res );

	// timed iterations
	double ti = 0.0, tt = 0.0;
	struct timeval t1, t2;
	UIN i;

printf( "nrows:  %d\n", nrows  );
printf( "nnz:    %d\n", nnz    );
printf( "lenAX:  %d\n", lenAX  );
printf( "lenSEC: %d\n", lenSEC );
printf( "tn:     %d\n", tn     );
printf( "thw:    %d\n", thw    );
printf( "ts:     %d\n", ts     );
printf( "bn:     %d\n", bn     );
printf( "bs:     %d\n", bs     );
printf( "log:    %d\n", log    );
printf( "tpb:    %d\n", tpb    );

/*
for ( i = 0; i < 112; i++ )
	printf( "ax[%3d]: %26.16le\n", i, matAXT.ax[i] );
for ( i = 0; i < 56; i++ )
	printf( "hdr[%3d]: %5d\n", i, matAXT.sec[i] );
fflush(stdout);
*/

	for ( i = 0; i < NUM_ITE; i++ )
	{
		fill_array( ompNT, nrows, 0, res );
		GT( t1 );
		int_axt_com_h1( ompNT, bn, bs, log, thw, tn, lenAX, lenSEC, matAXT.ax, matAXT.sec, res );
		GT( t2 );
		ti = measureTime( t2, t1 );
		tt = tt + ti;
	}


FPT dif;
for ( i = 0; i < 6; i++ )
{
	dif = fabs( fabs(ref[i]) - fabs(res[i]) );
	printf( "[ERROR]  row:%7d  ref:%22.1lf  res:%22.1lf   dif:%22.1lf\n", i, ref[i], res[i], dif );
}


	// store results
	str_res sr;
	strcpy( sr.name, "int_axt_com_h1_hw8_bs" );
	char BS[5]; sprintf( BS, "%d", bs );
	strcat( sr.name, BS );
	sr.et    = tt / (double) NUM_ITE;
	sr.ot    = 0.0;
	sr.flops = ( 2.0 * ( (double) nnz ) ) / sr.et;
	getErrors( nrows, ref, res, &(sr.sErr) );
	_mm_free( res );
	return( sr );
}



#endif




int main( int argc, char ** argv )
{
	// check input arguments
	str_inputArgs sia = checkArgs( argc, argv );

	// print run settings
	printRunSettings( sia );

	// CSR format  ------------------------------------------------------------------------------------------------------------------
	// read matrix in CSR format
	str_matCSR matCSR = matrixReading( sia.matFileName );
	// print matrix's statistics
	printMatrixStats( sia.matFileName, &matCSR );

	// get memory footprint, occupancy (beta) and conversion time
	str_formatData fd01 = getFormatDataCSR( &matCSR );

	// init vectors to perform SpMV multiplication and check errors (spM * vr = yr)
	FPT * vr = (FPT *) calloc( matCSR.nrows, sizeof(FPT) ); TEST_POINTER( vr );
	initVec( matCSR.nrows, vr );
	FPT * yr = (FPT *) calloc( matCSR.nrows,  sizeof(FPT) ); TEST_POINTER( yr );
	cf_CSR( matCSR, vr, yr );
	// test CSR kernels
	str_res sr01 = test_cf_CSR( matCSR, vr, yr );
	str_res sr02 = test_mkl_CSR( matCSR, vr, yr );
	str_res sr03 = test_sp_mkl_CSR( matCSR, vr, yr );
	str_res sr04 = test_sp_insp_mkl_CSR( matCSR, vr, yr );



FPT acu;
UIN i, j;
for ( i = 5; i < 6; i++ )
{
	acu = 0.0;
	for ( j = matCSR.row[i]; j < matCSR.row[i+1]; j++ )
	{
		acu = acu + matCSR.val[j]*vr[matCSR.col[j]];
		printf( "[CSR] row:%d  pos%2d  pro:%22.1lf  acu:%22.1lf\n", i, j, (matCSR.val[j] * vr[matCSR.col[j]]), acu );
		//printf( "row:%3d  rowLen:%3d  pos:%3d  val:%20.10le:  vec:%20.10le  pro:%20.10le  acu:%20.10le\n", i, matCSR.rl[i], j, matCSR.val[j], vr[matCSR.col[j]], matCSR.val[j]*vr[matCSR.col[j]], acu );
	}
}



	// CSR format  ------------------------------------------------------------------------------------------------------------------

	// AXC format  ------------------------------------------------------------------------------------------------------------------
	str_matAXC matAXC; str_formatData fd02 = getFormatDataAXC( sia.ompMaxThreads, matCSR, vr, &matAXC );
	str_res sr05 = test_int_AXC( sia.ompMaxThreads, matAXC, yr );
	// AXC format  ------------------------------------------------------------------------------------------------------------------

	// K1 format  -------------------------------------------------------------------------------------------------------------------
	str_matK1 matK1; str_formatData fd03 = getFormatDataK1( CHUNK_SIZE, matCSR, vr, &matK1 );
	str_res sr06 = test_int_k1( sia.ompMaxThreads, matK1, vr, yr );
	str_res sr07 = test_int_k1p( sia.ompMaxThreads, matK1, vr, yr );
	// K1 format  -------------------------------------------------------------------------------------------------------------------

	// AXT format  ------------------------------------------------------------------------------------------------------------------
	UIN bs = 128;
	str_matAXT matAXT1; str_formatData fd04 = getFormatDataAXT( sia.ompMaxThreads, bs, 8,  1, "NOC", matCSR, vr, &matAXT1 );
	str_matAXT matAXT2; str_formatData fd05 = getFormatDataAXT( sia.ompMaxThreads, bs, 8,  4, "NOC", matCSR, vr, &matAXT2 );
	str_matAXT matAXT3; str_formatData fd06 = getFormatDataAXT( sia.ompMaxThreads, bs, 8,  8, "NOC", matCSR, vr, &matAXT3 );
	str_matAXT matAXT4; str_formatData fd07 = getFormatDataAXT( sia.ompMaxThreads, bs, 8, 16, "NOC", matCSR, vr, &matAXT4 );
	str_matAXT matAXT5; str_formatData fd08 = getFormatDataAXT( sia.ompMaxThreads, bs, 8, 32, "NOC", matCSR, vr, &matAXT5 );
	str_matAXT matAXT6; str_formatData fd09 = getFormatDataAXT( sia.ompMaxThreads, bs, 8, 48, "NOC", matCSR, vr, &matAXT6 );
	str_matAXT matAXT7; str_formatData fd10 = getFormatDataAXT( sia.ompMaxThreads, bs, 8,  1, "COM", matCSR, vr, &matAXT7 );

	HDL; printf( "formats' data \n" ); HDL;
	printf( "%22s %20s %10s %20s\n", "format", "memory [Mbytes]", "occupancy", "convTime [s]" );
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd01.name, ( fd01.mfp * 1e-6 ), fd01.beta, fd01.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd02.name, ( fd02.mfp * 1e-6 ), fd02.beta, fd02.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd03.name, ( fd03.mfp * 1e-6 ), fd03.beta, fd03.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd04.name, ( fd04.mfp * 1e-6 ), fd04.beta, fd04.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd05.name, ( fd05.mfp * 1e-6 ), fd05.beta, fd05.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd06.name, ( fd06.mfp * 1e-6 ), fd06.beta, fd06.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd07.name, ( fd07.mfp * 1e-6 ), fd07.beta, fd07.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd08.name, ( fd08.mfp * 1e-6 ), fd08.beta, fd08.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd09.name, ( fd09.mfp * 1e-6 ), fd09.beta, fd09.ct ); fflush(stdout);
	printf( "%22s %20.2lf %10.2lf %20.6lf\n", fd10.name, ( fd10.mfp * 1e-6 ), fd10.beta, fd10.ct ); fflush(stdout);

	str_res sr08 = test_int_axt_noc_h1( sia.ompMaxThreads, matAXT1, yr );
	str_res sr09 = test_int_axt_noc( sia.ompMaxThreads, matAXT2, yr );
	str_res sr10 = test_int_axt_noc( sia.ompMaxThreads, matAXT3, yr );
	str_res sr11 = test_int_axt_noc( sia.ompMaxThreads, matAXT4, yr );
	str_res sr12 = test_int_axt_noc( sia.ompMaxThreads, matAXT5, yr );
	str_res sr13 = test_int_axt_noc( sia.ompMaxThreads, matAXT6, yr );
	str_res sr14 = test_int_axt_com_h1( sia.ompMaxThreads, bs, matAXT7, yr );
	// AXT format  ------------------------------------------------------------------------------------------------------------------


	HDL; printf( "kernels' results\n" ); HDL;
	printf( "%25s %15s %8s %15s %13s %13s %10s\n", "kernel", "exeTime [s]", "Gflops", "ordTime [s]", "aErr||.||inf", "rErr||.||inf", "rowInd" );
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr01.name, sr01.et, ( sr01.flops * 1e-9 ), sr01.ot, sr01.sErr.aErr, sr01.sErr.rErr, sr01.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr02.name, sr02.et, ( sr02.flops * 1e-9 ), sr02.ot, sr02.sErr.aErr, sr02.sErr.rErr, sr02.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr03.name, sr03.et, ( sr03.flops * 1e-9 ), sr03.ot, sr03.sErr.aErr, sr03.sErr.rErr, sr03.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr04.name, sr04.et, ( sr04.flops * 1e-9 ), sr04.ot, sr04.sErr.aErr, sr04.sErr.rErr, sr04.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr05.name, sr05.et, ( sr05.flops * 1e-9 ), sr05.ot, sr05.sErr.aErr, sr05.sErr.rErr, sr05.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr06.name, sr06.et, ( sr06.flops * 1e-9 ), sr06.ot, sr06.sErr.aErr, sr06.sErr.rErr, sr06.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr07.name, sr07.et, ( sr07.flops * 1e-9 ), sr07.ot, sr07.sErr.aErr, sr07.sErr.rErr, sr07.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr08.name, sr08.et, ( sr08.flops * 1e-9 ), sr08.ot, sr08.sErr.aErr, sr08.sErr.rErr, sr08.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr09.name, sr09.et, ( sr09.flops * 1e-9 ), sr09.ot, sr09.sErr.aErr, sr09.sErr.rErr, sr09.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr10.name, sr10.et, ( sr10.flops * 1e-9 ), sr10.ot, sr10.sErr.aErr, sr10.sErr.rErr, sr10.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr11.name, sr11.et, ( sr11.flops * 1e-9 ), sr11.ot, sr11.sErr.aErr, sr11.sErr.rErr, sr11.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr12.name, sr12.et, ( sr12.flops * 1e-9 ), sr12.ot, sr12.sErr.aErr, sr12.sErr.rErr, sr12.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr13.name, sr13.et, ( sr13.flops * 1e-9 ), sr13.ot, sr13.sErr.aErr, sr13.sErr.rErr, sr13.sErr.pos ); fflush(stdout);
	printf( "%25s %15.7lf %8.3lf %15.7lf %11.3le %13.3le %12d\n", sr14.name, sr14.et, ( sr14.flops * 1e-9 ), sr14.ot, sr14.sErr.aErr, sr14.sErr.rErr, sr14.sErr.pos ); fflush(stdout);

	return( EXIT_SUCCESS );
}



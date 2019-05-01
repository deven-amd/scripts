ROCBLAS_BASE_DIR=/root/rocBLAS
ROCBLAS_BENCH_DIR=build/release/clients/staging/

cd $ROCBLAS_BASE_DIR/$ROCBLAS_BENCH_DIR &&  \
    ltrace -x hipMalloc* -f -C -b -n 1 -L ./rocblas-bench \
	-f gemm -r f32_r --transposeA N --transposeB N -m 4096 -n 512 -k 9216 --alpha 1 --lda 4096 --ldb 9216 --beta 0 --ldc 4096



    # ltrace -x hip* -f -d -b -n 1 -x -L ./rocblas-bench \

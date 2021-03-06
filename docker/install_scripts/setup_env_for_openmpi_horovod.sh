export HOROVOD_WITHOUT_MXNET=1
export HOROVOD_WITHOUT_PYTORCH=1
export HOROVOD_WITH_TENSORFLOW=1

export HOROVOD_GPU_ROCM=1
export HOROVOD_GPU=ROCM

export HOROVOD_ROCM_PATH=$ROCM_PATH
export HOROVOD_ROCM_HOME=$ROCM_PATH

export HOROVOD_GPU_ALLREDUCE=NCCL
export HOROVOD_GPU_BROADCAST=NCCL

OPENMPI_HOME=$ROCM_PATH/openmpi
export PATH="${OPENMPI_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="$OPENMPI_HOME/lib:${LD_LIBRARY_PATH}"

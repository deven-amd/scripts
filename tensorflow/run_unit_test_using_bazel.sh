####################

set -e
# set -x

N_BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
TF_GPU_COUNT=$(lspci|grep 'controller'|grep 'AMD/ATI'|wc -l)
TF_TESTS_PER_GPU=1
N_TEST_JOBS=$(expr ${TF_GPU_COUNT} \* ${TF_TESTS_PER_GPU})

echo ""
echo "Bazel will use ${N_BUILD_JOBS} concurrent build job(s) and ${N_TEST_JOBS} concurrent test job(s)."
echo ""

export -n TF_NEED_ROCM=1
#####################

options=""

# options="$options --config=opt"
# options="$options --config=rocm"
# options="$options --action_env=HIP_PLATFORM=hcc"
# options="$options --config=cuda"
# options="$options --config=monolithic"


# options="$options --subcommands"

options="$options --test_sharding_strategy=disabled"
options="$options --test_timeout 600,900,2400,7200"
options="$options --cache_test_results=no"
options="$options --flaky_test_attempts=1"
options="$options --test_size_filters=small,medium,large"
# options="$options --runs_per_test=$TF_TESTS_PER_GPU"
# options="$options --test_output=streamed"

# options="$options --define=no_tensorflow_py_deps=true"

# options="$options --test_env=MIOPEN_DEBUG_CONVOLUTION_ATTRIB_FP16_ALT_IMPL=0"
# options="$options --test_env=ROCBLAS_INTERNAL_FP16_ALT_IMPL=0"

# options="$options --test_env=MIOPEN_LOG_LEVEL=7"
# options="$options --test_env=MIOPEN_ENABLE_LOGGING=1"
# options="$options --test_env=MIOPEN_ENABLE_LOGGING_CMD=1"
# options="$options --test_env=MIOPEN_DEBUG_CONV_FFT=0"
# options="$options --test_env=MIOPEN_DEBUG_CONV_FIRECT=0"
# options="$options --test_env=MIOPEN_DEBUG_CONV_GEMM=0"
# options="$options --test_env=MIOPEN_GEMM_ENFORCE_BACKEND=2"
# options="$options --test_env=AMD_OCL_BUILD_OPTIONS_APPEND=\"-save-temps-all\""

# options="$options --test_env=ROCBLAS_LAYER=1"  # enable trace logging
# options="$options --test_env=ROCBLAS_LAYER=2"  # enable bench logging
# options="$options --test_env=ROCBLAS_LAYER=4"  # enable profile logging
# options="$options --test_env=ROCBLAS_LAYER=7"

# options="$options --test_env=ROCFFT_LAYER=8"

# options="$options --test_env=HIP_HIDDEN_FREE_MEM=500"
# options="$options --test_env=HIP_TRACE_API=1"
# options="$options --test_env=LOG_LEVEL=3"
# options="$options --test_env=HIP_DB=api+mem+copy"
# options="$options --test_env=HIP_LAUNCH_BLOCKING=1"
# options="$options --test_env=HIP_API_BLOCKING=1"
# options="$options --test_env=HIP_LAUNCH_BLOCKING_KERNELS=kernel1,kernel2,... "

# options="$options --test_env=HCC_DB=0x48a"
# options="$options --test_env=HCC_SERIALIZE_KERNEL=3"
# options="$options --test_env=HCC_SERIALIZE_COPY=3"
# options="$options --test_env=HCC_PROFILE=2"

# options="$options --test_env=AMD_LOG_LEVEL=7"
# options="$options --test_env=AMD_SERIALIZE_KERNEL=3"
# options="$options --test_env=AMD_SERIALIZE_COPY=3"
# options="$options --test_env=LOADER_ENABLE_LOGGING=1" 

# options="$options --test_env=HSAKMT_DEBUG_LEVEL=7"
# options="$options --test_env=HSAKMT_LOG_LEVEL=7"

# options="$options --action_env=KMDUMPISA=1"
# options="$options --action_env=KMDUMPLLVM=1"

# options="$options --action_env=HIPCC_COMPILE_FLAGS_APPEND=-flegacy-pass-manager"
# options="$options --action_env=HIPCC_COMPILE_FLAGS_APPEND=-fno-legacy-pass-manager"

# options="$options --test_env=TF_CPP_MIN_LOG_LEVEL=1"
# options="$options --test_env=TF_CPP_MIN_VLOG_LEVEL=1"
# options="$options --test_env=TF_CPP_MAX_VLOG_LEVEL=3"

# vmodules="dummy=1"
# vmodules="$vmodules,rocm_tracer=3"
# vmodules="$vmodules,device_tracer_rocm=3"
# vmodules="$vmodules,meta_optimizer=4"
# vmodules="$vmodules,conv_ops_3d=3"
# vmodules="$vmodules,conv_grad_ops_3d=3"
# vmodules="$vmodules,gpu_kernel_helper=3"
# options="$options --test_env=TF_CPP_VMODULE=$vmodules"

# options="$options --test_env=XLA_FLAGS=\"--xla_dump_optimized_hlo_proto_to=/common/LOGS/\""


# tf_debug_output_dir="/common/tf_debug_output/"
# tf_debug_output_graph="$tf_debug_output_dir/graph"
# tf_debug_output_xla="$tf_debug_output_dir/xla"

# options="$options --test_env=TF_DUMP_GRAPH_PREFIX=$tf_debug_output_graph"

# options="$options --test_env=TF_XLA_FLAGS=--tf_xla_clustering_debug"
# options="$options --test_env=XLA_FLAGS=--xla_dump_to=$tf_debug_output_xla"

# options="$options --test_env=XLA_FLAGS=--xla_dump_hlo_as_text"

# options="$options --test_env=XLA_FLAGS=--xla_gpu_bef_executable"
# options="$options --config=experimental_enable_xlir"

# options="$options --test_env=TF_ROCM_FUSION_ENABLE=1"
# options="$options --test_env=TF_ROCM_FUSION_DUMP_GRAPH_BEFORE=1"
# options="$options --test_env=TF_ROCM_FUSION_DUMP_GRAPH_AFTER=1"

# options="$options --test_env=TF_ROCM_RETURN_BEST_ALGO_ONLY=1"
# options="$options --test_env=TF_ROCM_USE_BFLOAT16_FOR_CONV=1"
# options="$options --test_env=TF_ROCM_USE_IMMEDIATE_MODE=1"
# options="$options --test_env=TF_CUDNN_WORKSPACE_LIMIT_IN_MB=8192"
# options="$options --test_env=TF_ROCM_BW_POOL_CACHE=1"
# options="$options --test_env=TF_ROCM_KEEP_XLA_TEMPFILES=1"
# options="$options --test_env=TF_NUM_INTEROP_THREADS=16"

# options="$options --test_env=TF_GPU_ALLOCATOR=memory_guard"

# options="$options --test_env=HSA_TOOLS_LIB=\"librocr_debug_agent64.so\""
# options="$options --test_env=LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib"
# options="$options --test_env=PATH=$PATH:/opt/rocm/hcc/bin"

# options="$options --test_env=LD_DEBUG=all"

# options="$options --test_env="
# options="$options "
# echo $options


all_tests=""

testlist=""

while (( $# )); do

    if [ $1 == "-xla" ]; then
	options="$options --config=xla"
    elif [ $1 == "-v1" ]; then
	options="$options --config=v1"
	# options="$options --define=tf_api_version=1"
	# options="$options --test_env=TF2_BEHAVIOR=0"
    elif [ $1 == "-v2" ]; then
	options="$options --config=v2"
	# options="$options --define=tf_api_version=2"
	options="$options --test_env=TF2_BEHAVIOR=1"
	options="$options --test_tag_filters=-v1only"
    elif [ $1 == "-dbg" ]; then
	options="$options --compilation_mode=dbg"
    elif [ $1 == "-f" ]; then
	options="$options --jobs=$N_BUILD_JOBS"
	options="$options --local_test_jobs=$N_TEST_JOBS"
	options="$options --test_env=TF_GPU_COUNT=$TF_GPU_COUNT"
	options="$options --test_env=TF_TESTS_PER_GPU=$TF_TESTS_PER_GPU"
	options="$options --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute"
	testlist=$2
	shift
    else
	options="$options --test_env=HIP_VISIBLE_DEVICES=0"
	# options="$options --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=128"
	# options="$options --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=1024"
	# options="$options --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=2048"
	options="$options --test_env=TF_PER_DEVICE_MEMORY_LIMIT_MB=4096"
	# options="$options --run_under=ltrace"
	# options="$options --run_under=strace"
	# options="$options --run_under=pdb"
	all_tests=$1
    fi

    shift
done

if [[ ! -z $testlist ]]; then
    while read testname
    do
	if [[ $testname != \#* ]]; then
	    all_tests="$all_tests $testname"
	fi
    done < <(cat $testlist)
fi

if [[ ! -z $all_tests ]]; then
    rm -rf /tmp/amdgpu_xla*
    bazel test $options $all_tests
else
    echo "no testcase specified"
fi

# bazel query buildfiles(deps($testname))

# llvm-objdump -disassemble -mcpu=gfx900 your.hsaco

# bazel run --config=rocm --config=opt //tensorflow/compiler/xla/tools:hlo_proto_to_json -- --input_file=/common/LOGS/Types.4.pb --output_file=/common/LOGS/Types.4.pb.json
# sudo apt-get install -y strace ltrace

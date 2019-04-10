####################
set -e
# set -x

N_JOBS=$(grep -c ^processor /proc/cpuinfo)
N_GPUS=$(lspci|grep 'VGA'|grep 'AMD/ATI'|wc -l)

echo ""
echo "Bazel will use ${N_JOBS} concurrent build job(s) and ${N_GPUS} concurrent test job(s)."
echo ""

export TF_NEED_ROCM=1
export TF_GPU_COUNT=${N_GPUS}
#####################

options=""

options="$options --config=opt"
options="$options --config=rocm"
# options="$options --config=cuda"
# options="$options --config=monolithic"


options="$options --test_sharding_strategy=disabled"
options="$options --test_timeout 600,900,2400,7200"
options="$options --cache_test_results=no"
options="$options --flaky_test_attempts=3"
options="$options --test_output=all"

# options="$options --test_env=MIOPEN_ENABLE_LOGGING=1"
# options="$options --test_env=MIOPEN_ENABLE_LOGGING_CMD=1"
# options="$options --test_env=MIOPEN_DEBUG_CONV_FFT=0"
# options="$options --test_env=MIOPEN_DEBUG_CONV_FIRECT=0"
# options="$options --test_env=MIOPEN_DEBUG_CONV_GEMM=0"
# options="$options --test_env=MIOPEN_GEMM_ENFORCE_BACKEND=2"
# options="$options --test_env=AMD_OCL_BUILD_OPTIONS_APPEND=\"-save-temps-all\""

# options="$options --test_env=ROCBLAS_LAYER=1"
# options="$options --test_env=ROCBLAS_LAYER=2"
# options="$options --test_env=ROCBLAS_LAYER=3"

# options="$options --test_env=KMDUMPISA=1"
# options="$options --test_env=KMDUMPLLVM=1"

# options="$options --test_env=HCC_DB=0x48a"
# options="$options --test_env=HIP_TRACE_API=2"

# options="$options --test_env=TF_CPP_MIN_LOG_LEVEL=1"
# options="$options --test_env=TF_CPP_MIN_VLOG_LEVEL=3"
# options="$options --test_env=TF_ROCM_FUSION_ENABLE=1"
# options="$options --test_env=XLA_FLAGS=\"--xla_dump_optimized_hlo_proto_to=/common/LOGS/\""

# options="$options --test_env=LD_DEBUG=all"

# options="$options --test_env="
# options="$options "
# echo $options


all_tests=""

testlist=""

while (( $# )); do

    if [ $1 == "-xla" ]; then
	options="$options --config=xla"
    elif [ $1 == "-v2" ]; then
	options="$options --config=v2"
    elif [ $1 == "-dbg" ]; then
	options="$options --compilation_mode=dbg"
    elif [ $1 == "-f" ]; then
	options="$options --jobs=$N_JOBS"
	# options="$options --local_test_jobs=1"
	options="$options --local_test_jobs=$TF_GPU_COUNT"
	options="$options --run_under=//tensorflow/tools/ci_build/gpu_build:parallel_gpu_execute"
	testlist=$2
	shift
    else
	options="$options --test_env=HIP_VISIBLE_DEVICES=0"
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
    bazel test $options $all_tests
else
    echo "no testcase specified"
fi

# bazel query buildfiles(deps($testname))

# llvm-objdump -disassemble -mcpu=gfx900 your.hsaco

# bazel run --config=rocm --config=opt //tensorflow/compiler/xla/tools:hlo_proto_to_json -- --input_file=/common/LOGS/Types.4.pb --output_file=/common/LOGS/Types.4.pb.json

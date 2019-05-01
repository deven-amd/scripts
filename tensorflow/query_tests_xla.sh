
all_tests="tests(//tensorflow/compiler/...)"
#echo $all_tests

excluded_tests="attr(tags, \"no_oss|benchmark-test\", $all_tests)"
# echo $gpu_ci_excluded_tests

ci_tests="$all_tests except $excluded_tests"
# echo $ci_tests

no_gpu_tests="attr(tags, \"no_gpu[^_]\", $ci_tests)"
# echo $no_gpu_tests

no_cuda_tests="attr(tags, \"no_cuda[^_]\", $ci_tests)"
# echo $no_cdua_tests

no_gpu_no_cuda_tests="attr(tags, \"no_gpu[^_]|no_cuda[^_]\", $ci_tests)"
# echo $no_gpu_no_cuda_tests

cuda_ci_tests="$ci_tests except $no_gpu_no_cuda_tests"
# echo $cuda_ci_tests

no_rocm_tests="attr(tags, \"no_rocm[^_]\", $ci_tests)"
# echo $no_rocm_tests

no_gpu_no_rocm_tests="attr(tags, \"no_gpu[^_]|no_rocm[^_]\", $ci_tests)"
# echo $no_gpu_no_rocm_tests

rocm_ci_tests="$ci_tests except $no_gpu_no_rocm_tests"
# echo $rocm_ci_tests

no_rocm_no_cuda_tests="attr(tags, \"no_rocm[^_]|no_cuda[^_]\", $ci_tests)"
# echo $no_rocm_no_cuda_tests

echo "--- XLA ---"

# echo "all tests :" `bazel cquery $all_tests | wc -l`
# echo "excluded tests :" `bazel cquery $excluded_tests | wc -l`

# echo "ci tests :" `bazel cquery $ci_tests | wc -l`
# echo "--------"
# echo "no_gpu tests :" `bazel cquery $no_gpu_tests | wc -l`
# echo "no_rocm tests :" `bazel cquery $no_rocm_tests | wc -l`
# echo "no_cuda tests :" `bazel cquery $no_cuda_tests | wc -l`
# echo "--------"
# echo "rocm_ci tests :" `bazel cquery $rocm_ci_tests | wc -l`
# echo "cuda_ci tests :" `bazel cquery $cuda_ci_tests | wc -l`
# echo "--------"

# echo "no_gpu_no_rocm tests :" `bazel cquery $no_gpu_no_rocm_tests | wc -l`
# echo "no_gpu_no_cuda tests :" `bazel cquery $no_gpu_no_cuda_tests | wc -l`
# echo "no_rocm_no_cuda tests :" `bazel cquery $no_rocm_no_cuda_tests | wc -l`
# echo "--------"

bazel cquery $no_rocm_no_cuda_tests
# bazel cquery $no_rocm_tests
# echo "--------"


# docker_repo=rocm/eigen-test
# tag=rocm-4.1.0-210325

# docker_repo=rocm/tensorflow
# tag=

# docker_repo=rocm/tensorflow-autobuilds
# tag=rocm4.1.0-latest
# tag=rocm4.1.0-csb-latest

# docker_repo=rocm/tensorflow-private
# tag=

# docker_repo=rocm/tensorflow-testing
# tag=rocm4.0.1-tensorflow2.5-ubuntu18.04-manylinux2010

# docker_repo=devenamd/tensorflow
# tag=rocmNV_3989-tf-rocmfork-210421
# tag=rocm41-upstream-r25-210422
# tag=rocm41-upstream-r25-210409
# tag=rocm42rc1-tf-rocmfork-210408
# tag=rocm41-rocmfork-r24enhanced-210408
# tag=rocm41-upstream-210329
# tag=rocm42_6738-tf-rocmfork-210324
# tag=rocm41rc4-tf-rocmfork-210322
# tag=rocm42rc2-tf-rocmfork-210420
# tag=rocm41-rocmfork-r24_rocm_enhanced-manylinux-210331
# tag=rocm41-upstream-210329
# tag=rocm41-rocmfork-r24_rocm_enhanced-210326
# tag=rocm41-rocmfork-r23_rocm_enhanced-210327

# docker_repo=devenamd/rocm
# tag=

# docker_repo=devenamd/mlir
# tag=

# docker_repo=devenamd/tfrt
# tag=

# docker_repo=devenamd/jax
# tag=

# docker_repo=devenamd/nhwc
# tag=

# docker_repo=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang
# tag=6776_ubuntu_py3_tensorflow_develop-upstream-QA-rocm42
# tag=6776_ubuntu_py3_tensorflow_develop-upstream-QA-rocm43
# tag=6833_ubuntu_py3_tensorflow_develop-upstream-QA-rocm43

# docker_repo=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang-tf-manylinux-env
# tag=

# docker_repo=manylinux2014-rocm-centos7-tf-test
# tag=

# docker_repo=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-navi21
# tag=3940_ubuntu_py3_tensorflow_develop-upstream-QA-rocm42

# docker_repo=rocmqa/staging-tf2.1
# tag=6562-py3-ub18.04-b24ee29-compiler-stg-build-job3091-perf


docker_image=$docker_repo:$tag
container_name=deven_14_SWDEV_280210_BASELINE

docker pull $docker_image

options=""
options="$options -it"
options="$options --network=host"
options="$options --ipc=host"
options="$options --shm-size 16G"
options="$options --group-add video"
options="$options --cap-add=SYS_PTRACE"
options="$options --security-opt seccomp=unconfined"

options="$options --device=/dev/kfd"
options="$options --device=/dev/dri"

options="$options -v $HOME/deven/common:/common"
# options="$options -v /data-bert:/data-bert"
options="$options -v /data/imagenet:/imagenet"
# options="$options -v /data/imagenet-inception:/data/imagenet-inception:"

docker run $options --name $container_name $docker_image

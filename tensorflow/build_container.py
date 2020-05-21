#!/usr/bin/env python3

import subprocess
import argparse
import sys
import os
import shutil
from datetime import date

TF_REPO_LOC = "/home/deven/deven/repos/tensorflow-upstream"


def run_shell_command(cmd, workdir):

    cwd = os.getcwd()
    os.chdir(workdir)

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    os.chdir(cwd)
    return result.returncode


def update_tf_repo(tf_branch):
    shutil.copy("/home/deven/deven/common/scripts/tensorflow/Dockerfile.rocm-internal", os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build"))
    
def get_docker_config(tf_branch):
    docker_image = "devenamd/tensorflow:{}-{}".format(tf_branch, date.today().strftime("%y%m%d"))
    docker_file = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build/Dockerfile.rocm")
    return docker_image, docker_file, None


def get_docker_config_hipclang_internal_build(tf_branch, internal_build_number):
    build_name="compute-rocm-dkms-no-npi-hipclang-{}".format(internal_build_number)
    docker_image = "devenamd/tensorflow:{}-{}".format(build_name, date.today().strftime("%y%m%d"))
    docker_file = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build/Dockerfile.rocm-internal")
    docker_build_args = [
        "--build-arg", "ROCM_INTERNAL_ARTIFACTORY=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang",
        "--build-arg", "ROCM_BUILD_NUM={}".format(internal_build_number),
        ]
    return docker_image, docker_file, docker_build_args


def get_docker_config_hipclang_bkc_build(tf_branch, bkc_major, bkc_minor):
    build_name="compute-rocm-dkms-no-npi-hipclang-int-bkc-{}-{}".format(bkc_major, bkc_minor)
    docker_image = "devenamd/tensorflow:{}-{}".format(build_name, date.today().strftime("%y%m%d"))
    docker_file = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build/Dockerfile.rocm-internal")
    docker_build_args = [
        "--build-arg", "ROCM_INTERNAL_ARTIFACTORY=http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-deb/",
        "--build-arg", "ROCM_BUILD_NAME=compute-rocm-dkms-no-npi-hipclang-int-bkc-{}".format(bkc_major),
        "--build-arg", "ROCM_BUILD_NUM={}".format(bkc_minor),
        ]
    return docker_image, docker_file, docker_build_args


def develop_upstream_hipclang_internal_build():
    tf_branch = "develop-upstream"
    internal_build_number = 2363
    update_tf_repo(tf_branch)
    return get_docker_config_hipclang_internal_build(tf_branch, internal_build_number)


def upstream_r21_build():
    tf_branch = "r2.1_rocm33"
    return get_docker_config(tf_branch)


if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("commit")
    # args = parser.parse_args()
    # commit = args.commit

    # docker_image, docker_file, docker_build_args = develop_upstream_hipclang_internal_build()
    docker_image, docker_file, docker_build_args = upstream_r21_build()
    
    docker_context = os.path.join(TF_REPO_LOC, "tensorflow/tools/ci_build")

    docker_build_command = ["docker", "build", "-t", docker_image, "-f", docker_file]
    if docker_build_args is not None:
        docker_build_command.extend(docker_build_args)
    docker_build_command.append(docker_context)

    # print (docker_build_command)
    run_shell_command(docker_build_command, TF_REPO_LOC)
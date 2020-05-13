#!/usr/bin/env python3

import subprocess
import argparse
import sys


def run_shell_command(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print("         {}".format(result.stdout.decode()))
        print("         {}".format(result.stderr.decode()))
        sys.exit(result.returncode)
    return result.returncode


def update_branch_to_commit(branch, commit):
    run_shell_command(["git", "checkout", commit])
    run_shell_command(["git", "branch", "-f", branch])
    run_shell_command(["git", "checkout", branch])
    subprocess.run(["git", "status"])
    subprocess.run(["git", "log", "-1"])
    print ("\n","git push -f origin {}".format(branch), "\n")



if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("commit")
    # args = parser.parse_args()
    # commit = args.commit

    commit = "3ba00b04c16ed75cfa97237adf8089c711f030ea"
    
    # Add eugene's fork as a remote for merging PR 39429
    # run_shell_command(["git", "remote", "add", "eugene_fork", "https://github.com/ekuznetsov139/tensorflow"])
    # run_shell_command(["git", "fetch", "eugene_fork"])
    
    # run_shell_command(["git", "fetch", "google_upstream"])
    # run_shell_command(["git", "fetch", "origin"])
    # run_shell_command(["git", "rev-parse", "--verify", commit])
    
    branch = "master-with-triaged-diffs"
    update_branch_to_commit(branch, commit)
    # merge_pending_upstream_prs(branch)
    # merge_rocm_enhanced_prs(branch)

    
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Adapted from to not run in Shell mode which is unsecure.
https://github.com/aws/sagemaker-rl-container/blob/master/src/vw-serving/src/vw_serving/sagemaker/gpu.py
"""

import logging
import subprocess
import time


_num_gpus = None


def _get_num_gpus(cmd: str) -> int:
    """
    Returns the number of available GPUs based on configuration parameters and available hardware GPU devices.
    Gpus are detected by running a provided command as a subprocess.
    :return: (int) number of GPUs
    """
    try:
        with open("std.out", "w") as stdout:
            proc = subprocess.Popen(cmd, shell=True, stdout=stdout)
        max_trials = 0
        while proc.poll() is None and max_trials < 100:
            time.sleep(0.1)
            max_trials += 1

        if proc.poll() is None:
            raise ValueError(f"{cmd} timed out after 10 secs.")

        if proc.poll() != 0:
            # In cases when command fails, no GPU is available.
            return 0
        else:
            # In cases of the command success, we read the number of GPU available
            # communicated by the provided command.
            with open("std.out", "r") as stdout:
                num_gpus = len(stdout.readlines())
            return num_gpus

    except (OSError, FileNotFoundError):
        logging.info(
            f"Error launching {cmd}, no GPU could be detected."
        )
        return 0
        

def get_num_gpus() -> int:
    global _num_gpus
    if _num_gpus is None:
        num_gpus = _get_num_gpus("nvidia-smi --list-gpus")
        print(num_gpus)
        if num_gpus == 0:
            print(
                    "Error launching /usr/bin/nvidia-smi, no GPU could be detected. Trying rocm-smi..."
                )
            num_gpus = _get_num_gpus("rocm-smi --showid | grep 'GPU ID'")
            if num_gpus == 0:
                print(
                        "Error launching rocm-smi, no GPU could be detected."
                    )

        _num_gpus = num_gpus

    return _num_gpus
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test file to ensure that in general certain situational setups for notebooks work.
"""

import os

from pytest import raises
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available


def basic_function():
    # Just prints the PartialState
    print(f"PartialState:\n{PartialState()}")


NUM_TRIALS = 1


def function_with_retry_bug():
    global NUM_TRIALS
    if NUM_TRIALS < 3:
        NUM_TRIALS += 1
        raise RuntimeError("Trial failed")
    print(f"PartialState:\n{PartialState()}")


def function_with_fractional_bug():
    state = PartialState()
    if not state.is_main_process:
        raise RuntimeError("Fractional bug")
    state.wait_for_everyone()
    print(f"PartialState:\n{state}")


NUM_PROCESSES = int(os.environ.get("ACCELERATE_NUM_PROCESSES", 1))


def test_can_initialize():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


@require_bnb
def test_problematic_imports():
    with raises(RuntimeError, match="Please keep these imports"):
        import bitsandbytes as bnb  # noqa: F401

        notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


def test_fault_tolerant():
    notebook_launcher(function_with_retry_bug, (), num_processes=NUM_PROCESSES, max_restarts=3)


def test_fractional_bug():
    with raises(ChildFailedError, match="Fractional bug"):
        notebook_launcher(function_with_fractional_bug, (), num_processes=NUM_PROCESSES)


def main():
    print("Test basic notebook can be ran")
    test_can_initialize()
    if is_bnb_available():
        print("Test problematic imports (bnb)")
        test_problematic_imports()
    print("Test fault tolerant")
    test_fault_tolerant()
    print("Test fractional bug")
    test_fractional_bug()


if __name__ == "__main__":
    main()

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
import time

from pytest import raises
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError

from accelerate import PartialState, notebook_launcher
from accelerate.test_utils import require_bnb
from accelerate.utils import is_bnb_available

from multiprocessing import Queue



def basic_function():
    # Just prints the PartialState
    print(f"PartialState:\n{PartialState()}")


def tough_nut_function(queue:Queue):
    if not queue.full():
        queue.put("knock knock")
        raise RuntimeError("The nut hasn't cracked yet! Try again.")
    
    print(f"PartialState:\n{PartialState()}")


def bipolar_sleep_function(sleep_sec: int):
    state = PartialState()
    if state.process_index % 2 == 0:
        time.sleep(sleep_sec)
    else:
        raise RuntimeError("sad because i throw")


NUM_PROCESSES = int(os.environ.get("ACCELERATE_NUM_PROCESSES", 1))


def test_can_initialize():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


def test_static_rdzv_backend():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES, rdzv_backend="static")


def test_c10d_rdzv_backend():
    notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES, rdzv_backend="c10d")


def test_fault_tolerant(max_restarts:int=3):
    Queue = Queue(maxsize=max_restarts-1)
    notebook_launcher(tough_nut_function, (), num_processes=NUM_PROCESSES, max_restarts=max_restarts)


def test_monitoring(monitor_interval:float=0.01):
    # Assert that the monitor_interval is working
    start_time = time.time()
    notebook_launcher(bipolar_sleep_function, (100,), num_processes=NUM_PROCESSES, monitor_interval=monitor_interval)
    assert time.time() - start_time < 2 * monitor_interval, "Monitoring is not working"


@require_bnb
def test_problematic_imports():
    with raises(RuntimeError, match="Please keep these imports"):
        import bitsandbytes as bnb  # noqa: F401

        notebook_launcher(basic_function, (), num_processes=NUM_PROCESSES)


def main():
    print("Test basic notebook can be ran")
    test_can_initialize()
    print("Test static rendezvous backend")
    test_static_rdzv_backend()
    print("Test c10d rendezvous backend")
    test_c10d_rdzv_backend()
    print("Test fault tolerant")
    test_fault_tolerant()
    print("Test monitoring")
    test_monitoring()
    if is_bnb_available():
        print("Test problematic imports (bnb)")
        test_problematic_imports()


if __name__ == "__main__":
    main()

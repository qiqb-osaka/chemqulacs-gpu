# type:ignore
import glob
import os
import subprocess
import sys

import pytest

HAVE_CUDA = False
if os.environ.get("CUDA_PATH") is not None:
    HAVE_CUDA = True


class cuQuantumSampleTestError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


samples_path = os.path.join(os.path.dirname(__file__), "..", "examples", "09.cuquantum")
sample_files = glob.glob(samples_path + "/**/*.py", recursive=True)


def run_sample(path, *args):
    fullpath = os.path.join(samples_path, path)
    result = subprocess.run(
        (sys.executable, fullpath) + args, capture_output=True, env=os.environ
    )
    if result.returncode:
        msg = "Got error:\n"
        msg += f"{result.stderr.decode()}"
        if "ModuleNotFoundError: No module named 'torch'" in msg:
            pytest.skip("PyTorch uninstalled, skipping related tests")
        else:
            raise cuQuantumSampleTestError(msg)
    else:
        print(result.stdout.decode())


@pytest.mark.parametrize("sample", sample_files)
class TestSamples:
    def test_sample(self, sample):
        if HAVE_CUDA is False:
            pytest.skip("test requires CUDA_PATH")
        run_sample(sample)

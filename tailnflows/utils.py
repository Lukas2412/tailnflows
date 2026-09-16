import os
from pathlib import Path
from filelock import FileLock
import subprocess
import pickle
from typing import Any
import torch
import torch.multiprocessing as mp
import io


def get_project_root():
    root_path = os.environ.get("TAILNFLOWS_HOME", Path(__file__).parent.parent)
    return root_path

def get_data_path():
    return os.environ.get("TAILNFLOWS_DATA_HOME", f"{get_project_root()}/data")

def get_experiment_output_path():
    return os.environ.get("TAILNFLOWS_EXPERIMENT_DIR", f"{get_project_root()}/experiment_output")

def add_experiment_output_data(path: str, label: str, data: Any, force_write: bool = False) -> None:
    rd_path = f"{get_experiment_output_path()}/{path}.p"
    lock_path = rd_path + ".lock"

    data_file = Path(rd_path)
    lock = FileLock(lock_path)

    with lock:  # Ensures only one process accesses the file at a time
        if not data_file.is_file():
            data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(rd_path, "wb") as f:
                pickle.dump({}, f)
        elif not force_write:
            raise RuntimeError(
                f"Experiment data already present at {rd_path}. "
                "Use force_write=True to overwrite, or avoid duplicate runs."
            )

        # Now safe to read
        with open(rd_path, "rb") as f:
            raw_data = pickle.load(f)

        if label not in raw_data:
            raw_data[label] = []
        raw_data[label].append(data)

        # Write back
        with open(rd_path, "wb") as f:
            pickle.dump(raw_data, f)

def add_raw_data(path: str, label: str, data: Any, force_write: bool = False) -> int:
    rd_path = f"{path}.p"
    lock_path = rd_path + ".lock"

    data_file = Path(rd_path)
    lock = FileLock(lock_path)

    with lock:
        if not data_file.is_file():
            data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(rd_path, "wb") as f:
                pickle.dump({}, f)
        elif not force_write:
            raise RuntimeError(
                f"Data file already exists at {rd_path}. "
                "Use force_write=True to overwrite."
            )

        with open(rd_path, "rb") as f:
            raw_data = pickle.load(f)

        if label not in raw_data:
            raw_data[label] = []

        ix = len(raw_data[label])
        raw_data[label].append(data)

        with open(rd_path, "wb") as f:
            pickle.dump(raw_data, f)

    return ix

class CPU_Unpickler(pickle.Unpickler):
    # thanks https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def load_experiment_output_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/experiment_output/{path}.p"
    try:
        return pickle.load(open(rd_path, "rb"))
    except RuntimeError:
        return CPU_Unpickler(open(rd_path, "rb")).load()
    
def load_raw_data(path: str) -> Any:
    rd_path = f"{path}.p"
    try:
        return pickle.load(open(rd_path, "rb"))
    except RuntimeError:
        return CPU_Unpickler(open(rd_path, "rb")).load()


def load_torch_data(path: str) -> Any:
    rd_path = f"{get_project_root()}/data/{path}.p"
    if not torch.cuda.is_available():
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))
    else:
        data = torch.load(open(rd_path, "rb"), map_location=torch.device("cpu"))

    return data


class RunWrapper:
    def __init__(self, run_experiment):
        self.run_experiment = run_experiment

    def __call__(self, exp_ix_kwargs):
        exp_ix, kwargs = exp_ix_kwargs
        self.run_experiment(experiment_ix=exp_ix + 1, **kwargs)


def parallel_runner(run_experiment, experiments, max_runs=3):
    print(f"{len(experiments)} experiments to run...")
    with mp.Pool(max_runs) as p:
        p.map(RunWrapper(run_experiment), list(enumerate(experiments)), chunksize=1)

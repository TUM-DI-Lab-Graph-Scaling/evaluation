import os
import pandas as pd

class OCPDataFrameLoader:

    def __init__(self, ocp_metrics_path, directory, runs):
        """
        Initialize a data frame loader to get ocp data.

        ocp_metrics_path: path to ocp metrics directory
        directory: directory within the ocp metrics directory for which runs should be considered
        runs: dict containing runs as items with abbreviations as keys
        """
        self.ocp_metrics_path = ocp_metrics_path
        self.directory = directory
        self.runs = runs

    def get_paths(self, csv_type=None) -> dict:
        """
        Returns paths to the specific .csv files.

        directory:  directory in which the runs are stored
        sv_type:    ["resources"|"torch_cuda"|"runtimes"|None] 
        """
        if csv_type == "resources":
            return {key: os.path.join(self.ocp_metrics_path, self.directory, f"{run}_resources.csv") for key, run in self.runs.items()}
        elif csv_type == "torch_cuda":
            return {key: os.path.join(self.ocp_metrics_path, self.directory, f"{run}_torch_cuda.csv") for key, run in self.runs.items()}
        elif csv_type == "runtimes":
            return {key: os.path.join(self.ocp_metrics_path, self.directory, f"{run}_runtimes.csv") for key, run in self.runs.items()}
        elif csv_type is None:
            return self.get_paths(csv_type="resources"), self.get_paths(csv_type="torch_cuda"), self.get_paths(csv_type="runtimes")
        else:
            raise ValueError(f"CSV type {csv_type} is not supported.")

    def get_metrics_csv(self, file_path, csv_type) -> pd.DataFrame:
        """
        Return a pd.DataFrame loaded from a specific .csv file, transformed according to its type.

        file_path:  path to the metrics csv file within the ocp metrics folder
        csv_type:   ["resources"|"torch_cuda"|"runtimes"]
        """
        df = pd.read_csv(os.path.join(self.ocp_metrics_path, file_path))
        if csv_type == "resources":
            for header in ["memory_used", "memory_free"]:
                df[header] = df[header] / 1_000_000
        elif csv_type == "torch_cuda":
            for header in ["gpu_memory_allocated", "gpu_memory_reserved"]:
                df[header] = df[header] / 1_000_000
        elif csv_type == "runtimes":
            df["rest"] = df["epoch_time"] - df["dataloading_time"] - df["forward_time"] - df["backward_time"]
        else:
            raise ValueError(f"CSV type {csv_type} is not supported.")
        return df
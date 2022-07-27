import os

import pandas as pd
import matplotlib.pyplot as plt

from data_loading import OCPDataFrameLoader

class CUDAMemory:

    def __init__(self, data_loader):
        """
        Initialize CUDAMemory instance with which OCP cuda memory consumptions can be evaluated and visualized.

        data_loader: OCPDataFrameLoader that provides the necessary OCP data.
        """
        self.data_loader = data_loader
        self.colors  = {
            "blue":     (100/255, 160/255, 200/255),
            "orange":   (227/255, 114/255, 34/255),
            "green":    (112/255, 194/255, 165/255),
            "gray":     (153/255, 153/255, 153/255)
        }

    def summary_per_gpu(self, run) -> pd.DataFrame:
        """
        Returns statistical summary of the cuda memory usage per rank.

        run: specific run from self.data_loader to compute the mean.
        """
        torch_cuda_paths = self.data_loader.get_paths(csv_type="torch_cuda")
        cuda_df = self.data_loader.get_metrics_csv(torch_cuda_paths[run], csv_type="torch_cuda")
        return cuda_df.groupby("rank")[["gpu_memory_allocated", "gpu_memory_reserved"]].describe()

    def summary_averaged(self, run) -> pd.DataFrame:
        """
        Returns statistical summary of the cuda memory usage averaged over all ranks.

        run: specific run from self.data_loader to compute the mean.
        """
        torch_cuda_paths = self.data_loader.get_paths(csv_type="torch_cuda")
        cuda_df = self.data_loader.get_metrics_csv(torch_cuda_paths[run], csv_type="torch_cuda")
        return cuda_df[["gpu_memory_allocated", "gpu_memory_reserved"]].describe()

    def plot_cuda_memory(
        self,
        run, 
        mode="individual", 
        show=True,
        save=False, 
        save_dir="outputs/cuda_memory"
    ):
        """
        Plot the GPU memory allocated by or reserved for Cuda tensors over time.

        run: run for which the memory over time should be displayed.
        mode:
            "individual":   Plot memory consumption for each rank individually.
            "averaged":     Plot memory consumption averaged over the GPUs.
            "sum":          Plot total memory consumption summed up over all GPUs.
        show: if the created plot(s) should be shown as output in the notebook.
        save: if the created plot(s) should be saved to a pdf file.
        save_dir: directory to which the created plots should be saved.
        """
        plt.rcParams['text.usetex'] = True
        torch_cuda_paths = self.data_loader.get_paths(csv_type="torch_cuda")
        cuda_df = self.data_loader.get_metrics_csv(torch_cuda_paths[run], csv_type="torch_cuda")
        if mode == "individual":
            cuda_plot_df = cuda_df.set_index("datetime")
            cuda_plot_df.index = pd.to_datetime(cuda_plot_df.index).strftime("%H:%M")
            for rank, group in cuda_plot_df.groupby("rank"):
                group[["gpu_memory_allocated", "gpu_memory_reserved"]].plot(
                    xlabel="time", 
                    ylabel="memory in MB",
                    title=f"Rank {rank} GPU memory",
                    color=[self.colors["blue"], self.colors["orange"]]
                )
                plt.legend([
                    "allocated GPU CUDA memory",
                    "reserved GPU CUDA memory"
                ])
                if save:
                    if not os.path.exists(os.path.join(save_dir, run)):
                        os.makedirs(os.path.join(save_dir, run))
                    plt.savefig(os.path.join(save_dir, run, f"rank{rank}.pdf"))
                if not show:
                    plt.close()
        if mode in ["averaged", "sum"]:
            cuda_df_aggr = sum([
                cuda_df[cuda_df["rank"] == rank].reset_index(drop=True)[["gpu_memory_allocated", "gpu_memory_reserved"]]
                for rank in cuda_df["rank"].unique()
            ])
            cuda_df_aggr.index = pd.to_datetime(cuda_df[cuda_df["rank"] == 0].set_index("datetime").index).strftime("%H:%M")
            if mode == "averaged":
                cuda_df_aggr = cuda_df_aggr / cuda_df["rank"].nunique()
                title = "Averaged GPU memory"
            elif mode == "sum":
                title = "Total GPU memory"
            cuda_df_aggr.plot(
                xlabel="time",
                ylabel="memory in MB",
                title=title,
                color=[self.colors["blue"], self.colors["orange"]]
            )
            plt.legend([
                "allocated GPU CUDA memory",
                "reserved GPU CUDA memory"
            ])
            if save:
                if not os.path.exists(os.path.join(save_dir, run)):
                    os.makedirs(os.path.join(save_dir, run))
                plt.savefig(os.path.join(save_dir, run, f"{mode}.pdf"))
            if not show:
                plt.close()

    def compare(self) -> pd.DataFrame:
        """
        Compare cuda memory usage over all runs that are provided.
        """
        torch_cuda_paths = self.data_loader.get_paths(csv_type="torch_cuda")
        memory_runs = {}
        for key in self.data_loader.runs.keys():
            cuda_df = self.data_loader.get_metrics_csv(torch_cuda_paths[key], csv_type="torch_cuda")
            memory_runs[key] = cuda_df[["gpu_memory_allocated", "gpu_memory_reserved"]].mean()
        return pd.DataFrame(memory_runs)

    def compare_plot(
        self,
        xticks=None, 
        show=True, 
        save=False, 
        save_dir="outputs/cuda_memory"
    ):
        """
        Create bar plot of the allocated and reserved cuda memory usage over
        all runs that were concucted.

        xticks: bar names if these should not be the default row names of the runs data frame.
        show: if the created plot should be shown as output in the notebook.
        save: if the created plot should be saved.
        save_dir: directory to which the figure should be saved.
        """
        plt.rcParams['text.usetex'] = True
        runs_df = self.compare().T
        runs_df.plot.bar(
            ylabel="memory in MB",
            title=r"\textbf{GPU CUDA memory comparison}",
            rot=0,
            color=[self.colors["blue"], self.colors["orange"]],
            figsize=(15, 7)
        )
        if xticks:
            plt.xticks(range(len(xticks)), xticks)
        plt.legend([
            "allocated GPU CUDA memory",
            "reserved GPU CUDA memory"
        ])
        if save:
            plt.savefig(os.path.join(save_dir, "memory_comparison.pdf"))
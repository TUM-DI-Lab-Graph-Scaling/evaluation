import os

import pandas as pd
import matplotlib.pyplot as plt

from data_loading import OCPDataFrameLoader

class Runtimes:

    def __init__(self, data_loader):
        """
        Initialize Runtimes instance with which OCP runtimes can be evaluated and visualized.

        data_loader: OCPDataFrameLoader that provides the necessary OCP data.
        """
        self.data_loader = data_loader
        self.colors  = {
            "blue":     (100/255, 160/255, 200/255),
            "orange":   (227/255, 114/255, 34/255),
            "green":    (112/255, 194/255, 165/255),
            "gray":     (153/255, 153/255, 153/255)
        }

    def means(self, run, format="%H:%M:%S") -> pd.Series:
        """
        Outputs the runtimes of each phases in training steps of a specified run.

        run: specific run from self.data_loader to compute the mean.
        format: datetime format to which the runtimes data should be converted.
        """
        runtimes_paths = self.data_loader.get_paths(csv_type="runtimes")
        runtimes_df = self.data_loader.get_metrics_csv(runtimes_paths[run], csv_type="runtimes")
        runtimes_means = runtimes_df[["epoch_time", "dataloading_time", "forward_time", "backward_time", "rest"]].mean()
        for header in ["epoch_time", "dataloading_time", "forward_time", "backward_time", "rest"]:
            if format:
                runtimes_means[header] = pd.to_datetime(runtimes_means[header], unit="s")
                runtimes_means[header] = runtimes_means[header].strftime(format)
        return runtimes_means

    def compare(self, format="%H:%M:%S") -> pd.DataFrame:
        """
        Compare runtimes over all runs that are provided.

        format: datetime format to which the runtimes data should be converted.
        """
        return pd.DataFrame({key: self.means(key, format) for key in self.data_loader.runs.keys()})

    def compare_plot(
        self, 
        yticks=None,
        show=True,
        save=False,
        save_dir="outputs/runtimes"
    ):
        """
        Create a horizontal bar plot of the runtimes of all runs that were conducted.

        yticks: bar names if these should not be the default row names of the runs data frame.
        show: if the created plot should be shown as output in the notebook.
        save: if the created plot should be saved.
        save_dir: directory to which the figure should be saved.
        """
        plt.rcParams['text.usetex'] = True
        runs_df = self.compare(format=None).T
        for header in ["dataloading_time", "forward_time", "backward_time", "rest"]:
            runs_df[header] = runs_df[header] / 3600
        runs_df[["dataloading_time", "forward_time", "backward_time", "rest"]].plot.barh(
            title=r"\textbf{Runtimes comparison}",
            stacked=True,
            color=[self.colors["green"], self.colors["blue"], self.colors["orange"], self.colors["gray"]],
            figsize=(15, 7)
        )
        if yticks:
            plt.yticks(range(len(yticks)), yticks)
        plt.gca().invert_yaxis()
        plt.xlabel("runtime in hrs")
        plt.legend([
            "dataloading time",
            "forward time",
            "backward time",
            "rest"
        ])
        if save:
            plt.savefig(os.path.join(save_dir, "runtimes_comparison.pdf"))

import datetime
import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import underdeep as U
from PIL import Image
from src.grapher import plot_time_series_matplotlib

from .base_logger import BaseLogger


class UnderdeepLogger(BaseLogger):
    """Quickstart: https://docs.yandex-team.ru/underdeep/experiments/quickstart"""

    def __init__(
        self,
        project_code: str = "test",
        experiment_code: str = "Awesome experiment",
        run_name: str = "my cool run",
        detector_config: Optional[Dict] = None,
        **kwargs,
    ):
        experiment_code = experiment_code.replace(' ', '_').replace(',', '-')[:50]
        self.params = kwargs
        try:
            self.client = U.Client(project=project_code, experiment=experiment_code)
        except U.common.utils.UnderdeepException as error:
            if 'There is no experiment' in str(error):
                self.client = U.Client(project=project_code)
                new_exp = self.client.experiments.add(
                    code=experiment_code,
                    name=experiment_code,
                    description="This is a new experiment",
                )
                self.client.change_experiment(new_exp)
            else:
                raise error
        detector_config = detector_config or {}
        self.run = self.client.init_run(parameters=convert_timedelta_to_seconds(detector_config), name=run_name)

    def log_single_series_metrics(
        self,
        series_name: str,
        metrics: Dict,
        anomalies: pd.DataFrame,
        overall_metrics: Dict,
        step_id: int,
        **kwargs,
    ):
        metrics_df = pd.DataFrame.from_dict(overall_metrics, orient="index")

        with TemporaryDirectory() as tmp_dir:
            img_path = os.path.join(tmp_dir, "img.png")
            labeling_predicted = (anomalies["score"] > (metrics["best_threshold"])).astype("int32")
            plot_time_series_matplotlib(
                timestamp=anomalies.index,
                value=anomalies["value"],
                labeling_gt=anomalies["ground_truth"].astype("int32"),
                labeling_predicted=labeling_predicted,
                scores=anomalies["score"],
                threshold=metrics["best_threshold"],
                title=f"{series_name}",
                save_path=img_path,
                show=False,
            )
            PIL_image = Image.open(img_path)
            image = U.UImage(value=PIL_image)
            plt.clf()

        self.run.log(
            {
                **{metric: metrics[metric] for metric in metrics_df.columns},
                **{
                    f"{step_id:03d}_{metric}": metrics[metric]
                    for metric in metrics_df.columns
                    if metric not in ["n_observations", "time_length", "processing_time"]
                },
                "image": image,
                "step": step_id,
            },
            step=step_id,
        )

    def __del__(self):
        if hasattr(self, 'run'):
            self.run.finish()


def convert_timedelta_to_seconds(data):
    if isinstance(data, dict):
        return {key: convert_timedelta_to_seconds(value) for key, value in data.items()}
    elif isinstance(data, datetime.timedelta):
        return data.total_seconds()
    elif isinstance(data, (list, tuple, set)):
        return type(data)(convert_timedelta_to_seconds(item) for item in data)
    else:
        return data


import os
import json
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import numpy as np
import pandas as pd
from comet_ml import Experiment
from src.grapher import plot_time_series_matplotlib

from .base_logger import BaseLogger

class CometLogger(BaseLogger):
    def __init__(
            self,
            project_name: str='anomaly-detection-benchmark',
            workspace: Optional[str]=None,
            experiment_name: str='my-awesome-run',
            api_token_path: str='.comet_token',
            **kwargs
        ):
        super().__init__(**kwargs)
        
        # Load API token from file
        api_key = None
        if os.path.exists(api_token_path):
            with open(api_token_path, 'r') as f:
                api_key = f.read().strip()
        
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
            auto_metric_logging=True,
            auto_param_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
        
        if experiment_name:
            self.experiment.set_name(experiment_name)
            
        # Log any additional parameters
        self.experiment.log_parameters(kwargs)
        self.step = 0
        self.fallback_count = 0
        self.total_series = 0

    def log_single_series_metrics(
            self, 
            series_name: str, 
            metrics: Dict, 
            anomalies: pd.DataFrame, 
            overall_metrics: Dict, 
            **kwargs
        ):
        """Log metrics and artifacts for a single time series to Comet.ml."""
        self.total_series += 1
        if metrics.get("used_fallback", False):
            self.fallback_count += 1

        # Track fallback percentage
        fallback_pct = (self.fallback_count / self.total_series) * 100
        self.experiment.log_metric("fallback_percentage", fallback_pct, step=self.step)

        # Log metrics for this specific series
        # Prefixing with series name to distinguish in Comet
        series_metrics = {f"{series_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float, str))}
        self.experiment.log_metrics(series_metrics, step=self.step)
        
        # Update overall average metrics
        self._log_overall_stats(overall_metrics)
        
        # Create plots and save data
        with TemporaryDirectory() as tmp_dir:
            # Save plot
            plot_path = os.path.join(tmp_dir, f"{series_name}_plot.png")
            
            labeling_predicted = (anomalies["score"] > metrics["best_threshold"]).astype("int32")
            
            plot_time_series_matplotlib(
                timestamp=anomalies.index,
                value=anomalies['value'],
                labeling_gt=anomalies['ground_truth'].astype('int32'),
                labeling_predicted=labeling_predicted,
                title=f"{series_name} (F1-best: {metrics['f1_best']:.4f})",
                save_path=plot_path,
                scores=anomalies["score"],
                threshold=metrics["best_threshold"],
                save_path=f"./results/{series_name}_anomaly_plot.png"
            )
            
            self.experiment.log_image(plot_path, name=f"{series_name}_plot")
            
            # Save data sample as asset
            csv_path = os.path.join(tmp_dir, f"{series_name}_data.csv")
            anomalies.head(100).to_csv(csv_path) # Just first 100 rows as sample
            self.experiment.log_asset(csv_path, file_name=f"{series_name}_sample.csv")

        self.step += 1

    def _log_overall_stats(self, overall_metrics: Dict):
        """Calculate and log average metrics across all series."""
        if not overall_metrics:
            return
            
        metrics_df = pd.DataFrame.from_dict(overall_metrics, orient='index')
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        avg_metrics = metrics_df[numeric_cols].mean().to_dict()
        
        # Log as 'avg_metric_name'
        avg_metrics = {f"avg_{k}": v for k, v in avg_metrics.items()}
        self.experiment.log_metrics(avg_metrics, step=self.step)

    def __del__(self):
        if hasattr(self, 'experiment'):
            self.experiment.end()

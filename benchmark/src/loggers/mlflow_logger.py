import json
import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import mlflow
import numpy as np
import pandas as pd
from src.grapher import plot_time_series_matplotlib, plot_time_series_plotly

from .base_logger import BaseLogger


class MLflowLogger(BaseLogger):
    def __init__(
            self,
            experiment_name: str='anomaly_detection_benchmark',
            run_name: str='my awesome run',
            log_html: bool=True,
            save_to_root: bool=True,
            **kwargs
        ):
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(
            # experiment_id=mlflow.get_experiment_by_name(experiment_name),
            run_name=run_name
        )
        self.run_id = mlflow.active_run().info.run_id
        self.log_html = log_html
        self.params = kwargs
        self.save_to_root = save_to_root
        self.step = 0

    def __del__(self):
        if hasattr(self, 'run'):
            mlflow.end_run()

    def log_single_series_metrics(
            self, 
            series_name: str, 
            metrics: Dict, 
            anomalies: pd.DataFrame, 
            overall_metrics: Dict, 
            **kwargs
        ):
        """Log metrics and artifacts for a single time series to MLflow.
        
        Args:
            series_name: Name of the series (e.g., "series_001")
            metrics: Dictionary containing metrics for this series
            result: Dictionary containing anomalies DataFrame and metadata
        """
        # Log updated metrics table (instead of individual metrics to main Metrics section)
        self._log_metrics_table(overall_metrics)
        self._update_overall_metrics(overall_metrics)
        
        # Create temporary directory for this series
        with TemporaryDirectory() as tmp_dir:
            series_dir = os.path.join(tmp_dir, series_name)
            os.makedirs(series_dir)
            
            # Create and save series CSV
            series_df = pd.DataFrame({
                'timestamp': anomalies.index.astype('int64') // 10**6,  # Convert to Unix timestamp in milliseconds
                'value_0': anomalies['value'],
                'ground_truth': anomalies['ground_truth'].astype('int32'),
                'predicted': anomalies['predicted'].astype('int32')
            }, index=anomalies.index)
            
            csv_path = os.path.join(series_dir, "data.csv")
            series_df.to_csv(csv_path, index=False)

            labeling_predicted = (anomalies["score"] > metrics["best_threshold"]).astype("int32")
            
            if self.log_html:
                # Create and save interactive plot
                fig = plot_time_series_plotly(
                    timestamp=anomalies.index,
                    value=anomalies['value'],
                    labeling_gt=anomalies['ground_truth'].astype('int32'),
                    labeling_predicted=labeling_predicted,
                    title=f"{series_name}"
                )

                plot_html_path = os.path.join(series_dir, "plot.html")
                fig.write_html(
                    plot_html_path,
                    include_plotlyjs=True,
                    full_html=True
                )
            
            if not self.save_to_root:
                plot_path = os.path.join(series_dir, "plot.png")
            else:
                plot_path = os.path.join(tmp_dir, f"_{series_name}_plot.png")
            plot_time_series_matplotlib(
                timestamp=anomalies.index,
                value=anomalies['value'],
                labeling_gt=anomalies['ground_truth'].astype('int32'),
                labeling_predicted=labeling_predicted,
                # scores=anomalies['score'],
                title=f"{series_name}",
                save_path=plot_path
            )

            metrics_path = os.path.join(series_dir, "metrics.json")
            with open(metrics_path, 'w') as fp:
                json.dump(metrics, fp, indent=2)
            
            # Log the entire series directory
            mlflow.log_artifacts(series_dir, os.path.join('items', series_name), run_id=self.run_id)

            if self.save_to_root:
                mlflow.log_artifact(plot_path, run_id=self.run_id)

    def _log_metrics_table(self, overall_metrics):
        """Log metrics table for all processed series to MLflow as CSV artifact.
        
        Creates a CSV table with all series metrics and logs it as an MLflow artifact.
        The table is recreated from self.metrics each time to include all processed series.
        This approach ensures proper overwriting instead of accumulating records.
        """
        if len(overall_metrics) == 0:
            return
            
        metrics_df = pd.DataFrame.from_dict(overall_metrics, orient='index')
        
        # Add series_name column from index
        metrics_df = metrics_df.reset_index()
        metrics_df = metrics_df.rename(columns={'index': 'series_name'})
        
        # Reorder columns for better readability
        desired_columns = ['series_name', 'precision', 'recall', 'f1', 'f1_best', 'auc_pr', 'processing_time', 'time_length', 'n_observations', 'csv_path']
        # Only include columns that exist in the DataFrame
        columns = [col for col in desired_columns if col in metrics_df.columns]
        metrics_df = metrics_df[columns]
        
        # Round numeric columns for better display
        numeric_columns = ['precision', 'recall', 'f1', 'processing_time', 'time_length']
        for col in numeric_columns:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].round(4)
        
        # Use temporary directory to create CSV file with proper name
        with TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "metrics_table.csv")
            metrics_df.to_csv(csv_path, index=False)
            
            # Log CSV as artifact (this will overwrite previous version)
            mlflow.log_artifact(csv_path, run_id=self.run_id)

    def _update_overall_metrics(self, overall_metrics):
        metrics_df = pd.DataFrame.from_dict(overall_metrics, orient='index')
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        stats_df = pd.DataFrame([metrics_df[numeric_cols].mean()]).round(3)
        
        mlflow.log_metrics(stats_df.iloc[0].to_dict(), step=self.step, run_id=self.run_id)
        self.step += 1
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import f1_score
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult


class TSPulseDetector(BaseDetector):
    """
    IBM-Granite TSPulse-r1 anomaly detection model with mode triangulation.

    Implements anomaly detection using IBM Granite Time Series Pulse (TSPulse) model,
    which is based on transformer architecture for time series reconstruction and anomaly scoring.
    Uses mode triangulation to select the best prediction mode based on f1-score.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = self.params.get("_model")
        self._best_mode = None
        # self._initialize_model()

    def _initialize_model(self):
        """Initialize the TSPulse model and pipeline."""
    
        model_path = "ibm-granite/granite-timeseries-tspulse-r1"
        self._model = TSPulseForReconstruction.from_pretrained(
            model_path,
            num_input_channels=1,
            revision="main",
            mask_type="user",
        )

    def _create_pipeline(self, prediction_mode):
        """
        Create a pipeline with specific prediction mode.
        
        Use only after you have run _prepare_dataframe function!
        """
        return TimeSeriesAnomalyDetectionPipeline(
            self._model,
            timestamp_column="timestamp",
            target_columns=["value"],
            prediction_mode=prediction_mode if isinstance(prediction_mode, list) else [prediction_mode],
            aggregation_length=self.params.get("aggregation_length", 64),
            aggr_function=self.params.get("aggr_function", "max"),
            smoothing_length=self.params.get("smoothing_length", 8),
            least_significant_scale=self.params.get("least_significant_scale", 0.01),
            least_significant_score=self.params.get("least_significant_score", 0.1),
        )

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "threshold": 0.3,  # Anomaly score threshold (0-1 range from TSPulse)
            "aggregation_length": 64,
            "aggr_function": "max",
            "smoothing_length": 8,
            "least_significant_scale": 0.01,
            "least_significant_score": 0.1,
            "batch_size": 256,
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if not (0 <= params["threshold"] <= 1):
            raise ValueError("threshold must be between 0 and 1")
        if params["aggregation_length"] <= 0:
            raise ValueError("aggregation_length must be > 0")
        if params["smoothing_length"] <= 0:
            raise ValueError("smoothing_length must be > 0")
        if params["batch_size"] <= 0:
            raise ValueError("batch_size must be > 0")

    def _prepare_dataframe(self, time_series: TimeSeriesWrapper) -> pd.DataFrame:
        """
        Prepare DataFrame in the format expected by TSPulse pipeline.

        Args:
            time_series: Time series data

        Returns:
            DataFrame with timestamp and value columns
        """
        df = time_series.time_series_pd.copy()

        # TSPulse expects a timestamp column
        df = df.reset_index()
        df = df.rename(columns={"index": "timestamp", "value_0": "value", "is_anomaly": "y"})

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        return df

    def _check_insufficient_samples(self, time_series: TimeSeriesWrapper) -> bool:
        """
        Check if there are insufficient samples for TSPulse processing.

        TSPulse requires at least 3-4 windows of context (1536-2048 samples minimum)
        but we'll be conservative and require at least 1536 samples (512??)

        Args:
            time_series: Time series data

        Returns:
            True if samples are insufficient, False otherwise
        """
        n_samples = time_series.time_series_pd.shape[0]
        min_samples_required = 1536  
        return n_samples < min_samples_required

    def _get_fallback_result_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Return fallback result with zero scores for univariate time series with insufficient samples.

        Args:
            time_series: Time series data

        Returns:
            ModelResult with zero anomaly scores
        """
        n_samples = time_series.time_series_pd.shape[0]
        anomaly_scores = np.zeros(n_samples)
        expected_value = np.array(time_series.time_series_pd["value_0"])
        expected_bounds = np.column_stack(
            (
                expected_value - self.params["threshold"],
                expected_value + self.params["threshold"],
            )
        )
        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=(anomaly_scores > self.params["threshold"]),
            expected_value=expected_value,
            expected_bounds=expected_bounds,
            used_fallback=True,
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        """
        Perform anomaly detection on univariate time series using TSPulse with mode triangulation.
        Args:
            time_series: Time series data

        Returns:
            ModelResult with zero anomaly scores
        """
        if self._check_insufficient_samples(time_series):
            return self._get_fallback_result_univariate(time_series)

        # Preprocess dataframe
        data = self._prepare_dataframe(time_series)
        
        # Check for NaNs in input and handle them (though TimeSeriesWrapper should handle most)
        if data["value"].isnull().any():
            data["value"] = data["value"].interpolate(method="linear").ffill().bfill()
        
        if self._best_mode is None:
            # Modes to choose from as per instructions
            base_modes = [
                AnomalyScoreMethods.PREDICTIVE.value,
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ]
            
            # Possible sublists to test for triangulation
            prediction_modes = [
                [base_modes[0]],
                [base_modes[1]],
                [base_modes[2]],
                [base_modes[0], base_modes[1]],
                [base_modes[0], base_modes[2]],
                [base_modes[1], base_modes[2]],
                base_modes # All three
            ]
            
            f_scores = []

            # Use a portion for validation
            val_size = min(len(data) // 2, 1024) 
            val_data = data.iloc[:val_size]
            
            # Use 'is_anomaly' for ground truth as per task description
            y_true_col = "is_anomaly" if "is_anomaly" in val_data.columns else "y"

            # If no labels are available, default to all modes
            if y_true_col not in val_data.columns or len(np.unique(val_data[y_true_col].dropna())) < 2:
                self._best_mode = base_modes
            else:
                for mode in prediction_modes:
                    try:
                        pipeline = self._create_pipeline(mode)
                        result = pipeline(
                            val_data,
                            batch_size=self.params["batch_size"],
                            predictive_score_smoothing=True
                        )
                        scores = result["anomaly_score"].values
                        # Handle potential NaNs in scores
                        scores = np.nan_to_num(scores, nan=0.0)
                        
                        y_pred = (scores > self.params["threshold"]).astype(int)
                        y_true = val_data["is_anomaly"].values
                        
                        # Handle NaNs in y_true for f1_score
                        mask = ~np.isnan(y_true)
                        if np.sum(mask) > 0:
                            f_score = f1_score(y_true[mask], y_pred[mask], average="binary")
                        else:
                            f_score = 0.0
                            
                        f_scores.append(f_score)
                    except Exception:
                        f_scores.append(-1.0)
            
            self._best_mode = prediction_modes[np.argmax(f_scores)]

        # Run final pipeline with best mode
        final_pipeline = self._create_pipeline(self._best_mode)
        result = final_pipeline(
            data,
            batch_size=self.params["batch_size"],
            predictive_score_smoothing=True
        )

        anomaly_scores = result["anomaly_score"].values
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0)
        is_anomaly = anomaly_scores > self.params["threshold"]

        expected_value = np.array(time_series.time_series_pd["value_0"])
        
        # Calculate robust std for bounds if possible, else use threshold
        residual = data["value"].values - anomaly_scores # Heuristic for bounds
        try:
            std = self.calculate_std(residual)
        except Exception:
            std = self.params["threshold"]

        expected_bounds = np.column_stack(
            (
                expected_value - std * self.params["threshold"],
                expected_value + std * self.params["threshold"],
            )
        )

        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

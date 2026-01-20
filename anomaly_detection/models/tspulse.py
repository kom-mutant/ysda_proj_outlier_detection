import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction
from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import TimeSeriesAnomalyDetectionPipeline
from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult


class TSPulseDetector(BaseDetector):
    """
    IBM-Granite TSPulse-r1 anomaly detection model.

    Implements anomaly detection using IBM Granite Time Series Pulse (TSPulse) model,
    which is based on transformer architecture for time series reconstruction and anomaly scoring.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = self.params["_model"]
        self._pipeline = self.params["_pipeline"]
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

        # Create pipeline with default parameters
        self._pipeline = TimeSeriesAnomalyDetectionPipeline(
            self._model,
            timestamp_column="timestamp",
            target_columns=["value"],
            prediction_mode=[
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ],
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
        df = df.rename(columns={"index": "timestamp", "value_0": "value"})

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
        )

    def _detect_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
        if self._check_insufficient_samples(time_series):
            return self._get_fallback_result_univariate(time_series)

        # Preprocess dataframe
        data = self._prepare_dataframe(time_series)

        result = self._pipeline(
            data,
            batch_size=self.params["batch_size"],
            predictive_score_smoothing=True
        )

        anomaly_scores = result["anomaly_score"].values
        is_anomaly = anomaly_scores > self.params["threshold"]

        # For expected_value and expected_bounds, we use the original values
        # since TSPulse is a reconstruction-based method and doesn't provide explicit predictions

        expected_value = np.array(time_series.time_series_pd["value_0"])
        expected_bounds = np.column_stack(
            (
                expected_value - self.params["threshold"],
                expected_value + self.params["threshold"],
            )
        )

        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

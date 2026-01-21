
import numpy as np
import torch
import pandas as pd
from typing import Dict, Any
from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult
from tsfm_public.models.flowstate.modeling_flowstate import FlowStateForPrediction

class FlowStateDetector(BaseDetector):
    """
    IBM-Granite FlowState-r1 anomaly detection model.

    Uses FlowState forecasting model for anomaly detection by predicting next values
    and comparing with actual values using residual analysis with z-score thresholding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = self.params["_model"]
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._initialize_model()

    def _initialize_model(self):
        """Initialize the FlowState model."""

        model_path = "ibm-granite/granite-timeseries-flowstate-r1"
        self._model = FlowStateForPrediction.from_pretrained(model_path).to(self._device)

    def get_default_params(self) -> Dict[str, Any]:
        return {
            "context_length": 512,  # Minimum context length for FlowState
            "prediction_length": 1,  # Predict next single value for anomaly detection
            "threshold": 3.0,  # Z-score threshold for anomaly detection
            "scale_factor": 1.0,  # Sampling rate adjustment factor
            "batch_size": 32,  # Batch size for GPU processing
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["context_length"] <= 0:
            raise ValueError("context_length must be > 0")
        if params["prediction_length"] <= 0:
            raise ValueError("prediction_length must be > 0")
        if params["threshold"] < 0:
            raise ValueError("threshold must be >= 0")
        if params["scale_factor"] <= 0:
            raise ValueError("scale_factor must be > 0")

    def _check_insufficient_samples(self, time_series: TimeSeriesWrapper) -> bool:
        """
        Check if there are insufficient samples for FlowState processing.

        Args:
            time_series: Time series data

        Returns:
            True if samples are insufficient, False otherwise
        """
        n_samples = time_series.time_series_pd.shape[0]
        min_samples_required = self.params["context_length"] + self.params["prediction_length"]
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

        # Get the time series values as numpy array
        values = np.array(time_series.values)
        n_samples = len(values)

        context_length = self.params["context_length"]
        prediction_length = self.params["prediction_length"]

        # Initialize arrays for predictions and residuals
        predictions = np.full(n_samples, np.nan)
        residuals = np.full(n_samples, np.nan)

        # Process each prediction position individually (following FlowState notebook pattern)
        for i in range(context_length, n_samples - prediction_length + 1):
            # Get context window for this prediction
            context = values[i - context_length:i]

            # Convert to tensor for FlowState
            context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self._device)

            with torch.no_grad():
                forecast = self._model(
                    past_values=context_tensor,
                    prediction_length=prediction_length,
                    batch_first=True,
                    scale_factor=self.params["scale_factor"]
                )

                # Get point prediction (scalar)
                pred_value = forecast.prediction_outputs[0, 0, 0].item()

                # Get actual value (scalar)
                actual_value = values[i + prediction_length - 1]

                # Store prediction and calculate residual
                predictions[i + prediction_length - 1] = pred_value
                residuals[i + prediction_length - 1] = actual_value - pred_value
                
        # Handle positions where we couldn't make predictions
        valid_mask = np.isfinite(predictions)

        if np.any(valid_mask):
            first_valid_idx = np.where(valid_mask)[0][0]
            predictions[:first_valid_idx] = values[:first_valid_idx]
            residuals[:first_valid_idx] = 0.0
            predictions = pd.Series(predictions).fillna(method='ffill').values
            residuals = pd.Series(residuals).fillna(method='ffill').values

        # Calculate anomaly scores
        residual_std = self.calculate_std(residuals[valid_mask])

        if residual_std == 0:
            anomaly_scores = np.zeros(n_samples)
        else:
            anomaly_scores = np.abs(residuals / residual_std)

        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0)
        is_anomaly = anomaly_scores > self.params["threshold"]

        # Expected values and bounds
        expected_value = np.where(np.isnan(predictions), values, predictions)
        expected_bounds = np.column_stack(
            (
                expected_value - residual_std * self.params["threshold"],
                expected_value + residual_std * self.params["threshold"],
            )
        )

        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            expected_value=expected_value,
            expected_bounds=expected_bounds,
        )

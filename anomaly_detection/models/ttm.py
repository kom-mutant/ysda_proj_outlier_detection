import numpy as np
import torch
import pandas as pd
from typing import Dict, Any, Optional
from ..core import TimeSeriesWrapper
from .base import BaseDetector, ModelResult
from tsfm_public.models.tinytimemixer.modeling_tinytimemixer import TinyTimeMixerForPrediction

class TTMDetector(BaseDetector):
    """
    IBM-Granite TinyTimeMixer (TTM) anomaly detection model.

    Uses TTM-r2 forecasting model for anomaly detection. 
    It can work in two modes:
    1. Residual analysis (similar to AR): compares forecast with actual value.
    2. Probabilistic bounds: uses the model's ability to estimate uncertainty/quantiles.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = self.params.get("_model")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the TTM model."""
        model_path = "ibm-granite/granite-timeseries-ttm-r2"
        self._model = TinyTimeMixerForPrediction.from_pretrained(model_path).to(self._device)
        self._model.eval()

    def get_default_params(self) -> Dict[str, Any]:
         # TTM-r2 default parametres
        return {
            "context_length": 512,
            "prediction_length": 96,
            "threshold": 3.0,
            "use_probabilistic": True,
            "confidence_level": 0.95,
            "batch_size": 32,
        }

    def validate_params(self, params: Dict[str, Any]) -> None:
        if params["context_length"] <= 0:
            raise ValueError("context_length must be > 0")
        if params["prediction_length"] <= 0:
            raise ValueError("prediction_length must be > 0")
        if params["threshold"] < 0:
            raise ValueError("threshold must be >= 0")

    def _check_insufficient_samples(self, time_series: TimeSeriesWrapper) -> bool:
        n_samples = time_series.time_series_pd.shape[0]
        return n_samples < self.params["context_length"]

    def _get_fallback_result_univariate(self, time_series: TimeSeriesWrapper) -> ModelResult:
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
        if self._check_insufficient_samples(time_series):
            return self._get_fallback_result_univariate(time_series)

        values = np.array(time_series.values)
        n_samples = len(values)
        context_length = self.params["context_length"]
        prediction_length = self.params["prediction_length"]

        # Prepare for sliding window inference
        # In a real production setting, we might optimize this by batching
        
        all_predictions = np.full(n_samples, np.nan)
        all_upper_bounds = np.full(n_samples, np.nan)
        all_lower_bounds = np.full(n_samples, np.nan)
        
        # We start predicting from the first point we have enough context for
        for i in range(context_length, n_samples, prediction_length):
            end_idx = min(i + prediction_length, n_samples)
            current_prediction_length = end_idx - i
            
            context = values[i - context_length:i]
            context_tensor = torch.tensor(context, dtype=torch.float32).view(1, context_length, 1).to(self._device)
            
            with torch.no_grad():
                # TTM-r2 forward pass
                output = self._model(past_values=context_tensor)
                
                # Get mean prediction
                # output.prediction_outputs shape: [batch, pred_len, channels]
                pred_chunk = output.prediction_outputs[0, :current_prediction_length, 0].cpu().numpy()
                all_predictions[i:end_idx] = pred_chunk
                
                # Estimate bounds if model supports it or via heuristic
                # Based on ttm_conformal_anomaly_detection.ipynb idea:
                # We can use the scale (std) provided by the model if available
                if hasattr(output, "scale") and output.scale is not None:
                    scale = output.scale[0, 0, 0].item()
                else:
                    # Fallback to historical residual std
                    scale = 1.0 # Placeholder
                
                # Heuristic: 1.96 for 95% confidence if Gaussian
                margin = 1.96 * scale * (self.params["threshold"] / 3.0)
                all_upper_bounds[i:end_idx] = pred_chunk + margin
                all_lower_bounds[i:end_idx] = pred_chunk - margin

        # Fill leading NaNs with actual values for metrics consistency
        valid_mask = ~np.isnan(all_predictions)
        if not np.any(valid_mask):
             return self._get_fallback_result_univariate(time_series)
             
        first_valid_idx = np.where(valid_mask)[0][0]
        all_predictions[:first_valid_idx] = values[:first_valid_idx]
        
        # Calculate residuals and anomaly scores
        residuals = values - all_predictions
        # Handle NaNs in residuals for std calculation
        clean_residuals = residuals[valid_mask]
        residual_std = self.calculate_std(clean_residuals) if len(clean_residuals) > 0 else 1.0
        
        if residual_std == 0: 
            residual_std = 1e-8
        
        anomaly_scores = np.abs(residuals / residual_std)
        anomaly_scores = np.nan_to_num(anomaly_scores, nan=0.0)
        
        is_anomaly = anomaly_scores > self.params["threshold"]
        
        # Construct final expected bounds
        expected_bounds = np.column_stack((
            np.where(np.isnan(all_lower_bounds), all_predictions - residual_std * self.params["threshold"], all_lower_bounds),
            np.where(np.isnan(all_upper_bounds), all_predictions + residual_std * self.params["threshold"], all_upper_bounds)
        ))

        return ModelResult(
            anomaly_scores=anomaly_scores,
            is_anomaly=is_anomaly,
            expected_value=all_predictions,
            expected_bounds=expected_bounds,
        )

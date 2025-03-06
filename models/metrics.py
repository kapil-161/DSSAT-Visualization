"""
Model performance metrics calculation
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculate model performance metrics."""
    
    @staticmethod
    def d_stat(measured, simulated):
        """Calculate Willmott's index of agreement (d-stat)."""
        try:
            M = np.array(measured)
            S = np.array(simulated)
            M_mean = np.mean(M)
            
            numerator = np.sum((M - S) ** 2)
            denominator = np.sum((np.abs(M - M_mean) + np.abs(S - M_mean)) ** 2)
            
            return 1 - (numerator / denominator) if denominator != 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating d-stat: {e}")
            return None
            
    @staticmethod
    def rmse(obs_values: np.ndarray, sim_values: np.ndarray) -> float:
        """Calculate Root Mean Square Error."""
        return np.sqrt(np.mean((obs_values - sim_values) ** 2))
            
    @staticmethod
    def calculate_metrics(sim_values: np.ndarray, obs_values: np.ndarray, 
                         treatment_number: str) -> dict:
        """Calculate multiple performance metrics."""
        try:
            sim_values = np.asarray(sim_values, dtype=float)
            obs_values = np.asarray(obs_values, dtype=float)
            
            if len(sim_values) == 0 or len(obs_values) == 0:
                logger.warning("Empty input arrays")
                return None
                
            # Prepare data
            min_length = min(len(sim_values), len(obs_values))
            mask = ~np.isnan(sim_values[:min_length]) & ~np.isnan(obs_values[:min_length])
            sim_values = sim_values[:min_length][mask]
            obs_values = obs_values[:min_length][mask]
            
            if len(sim_values) == 0:
                logger.warning("No valid pairs after filtering")
                return None
                
            # Calculate metrics
            mean_obs = np.mean(obs_values)
            n = len(sim_values)
            
            rmse_value = MetricsCalculator.rmse(obs_values, sim_values)
            nrmse = (rmse_value / mean_obs) * 100 if mean_obs != 0 else None
            d_stat = MetricsCalculator.d_stat(obs_values, sim_values)
            
            return {
                "TRT": treatment_number,
                "n": n,
                "RMSE": rmse_value,
                "NRMSE": nrmse,
                "Willmott's d-stat": d_stat,
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return None

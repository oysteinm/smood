"""
Variance-Aware Kalman Smoothing for Survey Data

This module provides functions for smoothing time series survey data that accounts
for varying sample sizes and measurement uncertainty across time periods.

The approach uses Kalman filtering with time-varying measurement variance to
automatically adapt smoothing based on data quality.

Author: [Your Team]
Date: 2025
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Optional, Dict


def calculate_weighted_statistics(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    weight_col: str
) -> pd.DataFrame:
    """
    Calculate weighted monthly statistics from individual survey responses.
    
    For each time period, computes:
    - Weighted mean
    - Effective sample size (accounting for unequal weights)
    - Weighted variance
    - Measurement variance (standard error of the mean)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing individual survey responses
    date_col : str
        Name of the date/period column
    value_col : str
        Name of the survey response value column
    weight_col : str
        Name of the survey weight column
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: Time period
        - n: Sample size
        - n_eff: Effective sample size
        - value_mean: Weighted mean
        - value_se: Standard error of the mean
        - var_mean: Measurement variance (H_t)
    """
    # Filter to valid observations
    valid = (
        df[value_col].notna() & 
        df[weight_col].notna() & 
        (df[weight_col] > 0)
    )
    df_clean = df.loc[valid].copy()
    
    results = []
    
    for date, group in df_clean.groupby(date_col):
        w = group[weight_col].to_numpy()
        x = group[value_col].to_numpy()
        n = len(x)
        
        # Weight sums
        w_sum = w.sum()
        w2_sum = (w ** 2).sum()
        
        # Effective sample size
        n_eff = (w_sum ** 2) / max(w2_sum, 1e-12)
        
        # Weighted mean
        x_mean = (w * x).sum() / w_sum
        
        # Bessel-corrected weighted variance
        denom = max(w_sum - (w2_sum / w_sum), 1e-8)
        s2_w = (w * (x - x_mean) ** 2).sum() / denom
        
        # Variance of the mean (measurement variance)
        var_mean = s2_w / max(n_eff, 1e-8)
        se_mean = np.sqrt(max(var_mean, 1e-12))
        
        results.append({
            'date': date,
            'n': n,
            'n_eff': n_eff,
            'value_mean': x_mean,
            'value_se': se_mean,
            'var_mean': var_mean
        })
    
    return pd.DataFrame(results).sort_values('date').reset_index(drop=True)


def prepare_measurement_variance(H: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Ensure measurement variance values are positive and finite.
    
    Applies guards to handle:
    - Missing or infinite values
    - Extremely small variances
    
    Parameters
    ----------
    H : np.ndarray
        Array of measurement variances
    y : np.ndarray
        Array of observations (used for fallback variance)
        
    Returns
    -------
    np.ndarray
        Cleaned measurement variance array
    """
    H = H.copy()
    
    # Replace invalid values with median or data-based fallback
    good = np.isfinite(H) & (H > 0)
    fallback = np.nanmedian(H[good]) if np.any(good) else (np.nanvar(y) * 0.05)
    if not np.isfinite(fallback):
        fallback = np.nanvar(y) * 0.05
    H[~good] = fallback
    
    # Apply floor to prevent extremely small variances
    q05 = np.nanquantile(H, 0.05)
    if np.isfinite(q05) and q05 > 0:
        floor_val = 0.1 * q05
    else:
        floor_val = 0.1 * fallback
    
    H = np.maximum(H, floor_val)
    
    return H


def kalman_loglikelihood(
    y: np.ndarray, 
    H: np.ndarray, 
    Q: float, 
    P0: float = 1e6
) -> float:
    """
    Compute log-likelihood for local level model using Kalman filter.
    
    Model:
        y_t = mu_t + eps_t,  eps_t ~ N(0, H_t)
        mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, Q)
    
    Parameters
    ----------
    y : np.ndarray
        Observations (can contain NaN for missing values)
    H : np.ndarray
        Time-varying measurement variance
    Q : float
        Process variance
    P0 : float, optional
        Initial state variance
        
    Returns
    -------
    float
        Log-likelihood value
    """
    T = len(y)
    mu_pred = 0.0
    P_pred = P0 + Q
    loglik = 0.0
    
    for t in range(T):
        yt = y[t]
        Ht = H[t]
        
        if np.isfinite(yt):
            # Innovation and its variance
            S = P_pred + Ht
            S = max(S, 1e-8)
            v = yt - mu_pred
            
            # Log-likelihood contribution
            loglik -= 0.5 * (np.log(2 * np.pi) + np.log(S) + (v ** 2) / S)
            
            # Kalman update
            K = P_pred / S
            mu_filt = mu_pred + K * v
            P_filt = (1 - K) * P_pred
        else:
            # Skip update for missing observations
            mu_filt = mu_pred
            P_filt = P_pred
        
        # Predict next step
        mu_pred = mu_filt
        P_pred = P_filt + Q
    
    return loglik


def estimate_process_variance(
    y: np.ndarray, 
    H: np.ndarray, 
    q0: Optional[float] = None
) -> Tuple[float, Dict]:
    """
    Estimate process variance Q via maximum likelihood estimation.
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    H : np.ndarray
        Measurement variances
    q0 : float, optional
        Initial guess for Q
        
    Returns
    -------
    Q_hat : float
        Estimated process variance
    info : dict
        Dictionary containing optimization information:
        - success: Whether optimization converged
        - loglik: Log-likelihood at optimum
        - iterations: Number of iterations
    """
    # Initial guess
    if q0 is None:
        q0 = np.nanmean(H) * 0.05
        if not np.isfinite(q0) or q0 <= 0:
            q0 = np.nanvar(y) * 0.05
    
    # Optimize in log-space to ensure Q > 0
    def negative_loglik(log_Q):
        Q = np.exp(log_Q)
        return -kalman_loglikelihood(y, H, Q)
    
    result = minimize(negative_loglik, x0=np.log(q0), method='L-BFGS-B')
    
    Q_hat = float(np.exp(result.x[0]))
    
    info = {
        'success': result.success,
        'loglik': -result.fun,
        'iterations': result.nit,
        'message': result.message
    }
    
    return Q_hat, info


def kalman_filter(
    y: np.ndarray, 
    H: np.ndarray, 
    Q: float, 
    P0: float = 1e6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Kalman filter forward pass.
    
    Parameters
    ----------
    y : np.ndarray
        Observations
    H : np.ndarray
        Measurement variances
    Q : float
        Process variance
    P0 : float, optional
        Initial state variance
        
    Returns
    -------
    mu_pred : np.ndarray
        Predicted state means
    P_pred : np.ndarray
        Predicted state variances
    mu_filt : np.ndarray
        Filtered state means
    P_filt : np.ndarray
        Filtered state variances
    """
    T = len(y)
    mu_pred = np.zeros(T)
    P_pred = np.zeros(T)
    mu_filt = np.zeros(T)
    P_filt = np.zeros(T)
    
    # Initial prediction
    mu_pred[0] = 0.0
    P_pred[0] = P0 + Q
    
    # Filter first observation
    if np.isfinite(y[0]):
        S = P_pred[0] + H[0]
        S = max(S, 1e-8)
        K = P_pred[0] / S
        v = y[0] - mu_pred[0]
        mu_filt[0] = mu_pred[0] + K * v
        P_filt[0] = (1 - K) * P_pred[0]
    else:
        mu_filt[0] = mu_pred[0]
        P_filt[0] = P_pred[0]
    
    # Filter remaining observations
    for t in range(1, T):
        mu_pred[t] = mu_filt[t - 1]
        P_pred[t] = P_filt[t - 1] + Q
        
        if np.isfinite(y[t]):
            S = P_pred[t] + H[t]
            S = max(S, 1e-8)
            K = P_pred[t] / S
            v = y[t] - mu_pred[t]
            mu_filt[t] = mu_pred[t] + K * v
            P_filt[t] = (1 - K) * P_pred[t]
        else:
            mu_filt[t] = mu_pred[t]
            P_filt[t] = P_pred[t]
    
    return mu_pred, P_pred, mu_filt, P_filt


def rts_smoother(
    mu_pred: np.ndarray,
    P_pred: np.ndarray,
    mu_filt: np.ndarray,
    P_filt: np.ndarray,
    Q: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Rauch-Tung-Striebel (RTS) smoother backward pass.
    
    Parameters
    ----------
    mu_pred : np.ndarray
        Predicted state means from filter
    P_pred : np.ndarray
        Predicted state variances from filter
    mu_filt : np.ndarray
        Filtered state means
    P_filt : np.ndarray
        Filtered state variances
    Q : float
        Process variance
        
    Returns
    -------
    mu_smooth : np.ndarray
        Smoothed state means
    P_smooth : np.ndarray
        Smoothed state variances
    """
    T = len(mu_filt)
    mu_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    
    # Initialize with filtered values at last time point
    mu_smooth[-1] = mu_filt[-1]
    P_smooth[-1] = P_filt[-1]
    
    # Backward pass
    for t in range(T - 2, -1, -1):
        denom = max(P_pred[t + 1], 1e-8)
        J = P_filt[t] / denom
        mu_smooth[t] = mu_filt[t] + J * (mu_smooth[t + 1] - mu_pred[t + 1])
        P_smooth[t] = P_filt[t] + (J ** 2) * (P_smooth[t + 1] - P_pred[t + 1])
    
    return mu_smooth, P_smooth


def smooth_survey_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    weight_col: str,
    q_initial: Optional[float] = None,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Apply variance-aware Kalman smoothing to survey time series data.
    
    This is the main function that orchestrates the entire smoothing process:
    1. Calculate weighted statistics and measurement variance for each period
    2. Estimate optimal process variance Q via MLE
    3. Run Kalman filter and RTS smoother
    4. Return smoothed values with confidence intervals
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with individual survey responses
    date_col : str
        Name of the date/period column
    value_col : str
        Name of the survey response value column
    weight_col : str
        Name of the survey weight column
    q_initial : float, optional
        Initial guess for process variance Q
    confidence_level : float, optional
        Confidence level for intervals (default: 0.95)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - date: Time period
        - n: Sample size
        - n_eff: Effective sample size
        - raw_mean: Raw weighted mean
        - raw_se: Standard error of raw mean
        - smooth_mean: Smoothed estimate
        - smooth_se: Standard error of smoothed estimate
        - smooth_lower: Lower confidence bound
        - smooth_upper: Upper confidence bound
        - Q_estimated: Estimated process variance (constant for all rows)
        
    Example
    -------
    >>> df = pd.DataFrame({
    ...     'date': pd.date_range('2020-01-01', periods=100, freq='M'),
    ...     'response': np.random.normal(50, 10, 100),
    ...     'weight': np.random.uniform(0.5, 1.5, 100)
    ... })
    >>> result = smooth_survey_timeseries(df, 'date', 'response', 'weight')
    """
    # Step 1: Calculate weighted statistics
    stats = calculate_weighted_statistics(df, date_col, value_col, weight_col)
    
    # Step 2: Prepare measurement variance
    H = prepare_measurement_variance(
        stats['var_mean'].to_numpy(),
        stats['value_mean'].to_numpy()
    )
    y = stats['value_mean'].to_numpy()
    
    # Step 3: Estimate process variance Q
    Q_hat, opt_info = estimate_process_variance(y, H, q_initial)
    
    if not opt_info['success']:
        raise RuntimeError(
            f"Optimization failed: {opt_info['message']}. "
            "Try providing a different initial guess for q_initial."
        )
    
    # Step 4: Run Kalman filter
    mu_pred, P_pred, mu_filt, P_filt = kalman_filter(y, H, Q_hat)
    
    # Step 5: Run RTS smoother
    mu_smooth, P_smooth = rts_smoother(mu_pred, P_pred, mu_filt, P_filt, Q_hat)
    
    # Step 6: Calculate confidence intervals
    se_smooth = np.sqrt(np.maximum(P_smooth, 0.0))
    z_score = np.abs(np.percentile(np.random.standard_normal(10000), 
                                   [(1 - confidence_level) / 2 * 100,
                                    (1 + confidence_level) / 2 * 100]))[-1]
    
    # Step 7: Compile results
    results = stats.copy()
    results['raw_mean'] = results['value_mean']
    results['raw_se'] = results['value_se']
    results['smooth_mean'] = mu_smooth
    results['smooth_se'] = se_smooth
    results['smooth_lower'] = mu_smooth - z_score * se_smooth
    results['smooth_upper'] = mu_smooth + z_score * se_smooth
    results['Q_estimated'] = Q_hat
    
    # Drop intermediate columns
    results = results.drop(columns=['value_mean', 'value_se', 'var_mean'])
    
    return results


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic survey data
    np.random.seed(42)
    n_months = 60
    dates = pd.date_range('2020-01-01', periods=n_months, freq='MS')
    
    # Generate individual responses
    data = []
    for date in dates:
        # Varying sample sizes
        n_respondents = np.random.randint(500, 1500)
        
        # True underlying value with slow drift
        true_value = 60 + 10 * np.sin(2 * np.pi * len(data) / 12)
        
        # Generate responses with noise
        responses = np.random.normal(true_value, 15, n_respondents)
        weights = np.random.uniform(0.5, 1.5, n_respondents)
        
        for response, weight in zip(responses, weights):
            data.append({
                'date': date,
                'response': response,
                'weight': weight
            })
    
    df = pd.DataFrame(data)
    
    # Apply smoothing
    print("Running variance-aware Kalman smoothing...")
    result = smooth_survey_timeseries(df, 'date', 'response', 'weight')
    
    print(f"\nEstimated process variance (Q): {result['Q_estimated'].iloc[0]:.6f}")
    print(f"\nSmoothed results (first 10 months):")
    print(result[['date', 'n', 'n_eff', 'raw_mean', 'smooth_mean', 
                  'smooth_lower', 'smooth_upper']].head(10))
    
    # Calculate noise reduction
    noise_reduction = (1 - result['smooth_mean'].std() / result['raw_mean'].std()) * 100
    print(f"\nNoise reduction: {noise_reduction:.1f}%")

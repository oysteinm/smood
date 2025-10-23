# Variance-Aware Kalman Smoothing - Implementation Guide

## Overview

This module provides production-ready functions for smoothing survey time series data while accounting for varying measurement uncertainty across time periods.

## Key Features

- **Automatic adaptation to data quality**: More smoothing when sample sizes are small, less when they're large
- **Time-varying measurement variance**: Accounts for effective sample size variations
- **Maximum Likelihood Estimation**: Data-driven approach to estimate optimal smoothing parameters
- **Uncertainty quantification**: Provides confidence intervals for smoothed estimates
- **Missing data handling**: Robust to missing observations

## Quick Start

### Simple Usage

```python
from variance_aware_smoothing import smooth_survey_timeseries
import pandas as pd

# Your survey data with individual responses
df = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-01', '2024-02-01', ...],
    'nps_score': [9, 7, 8, ...],
    'survey_weight': [1.0, 1.2, 0.8, ...]
})

# Apply smoothing
result = smooth_survey_timeseries(
    df=df,
    date_col='date',
    value_col='nps_score',
    weight_col='survey_weight'
)

# Result contains:
# - raw_mean: Original weighted monthly means
# - smooth_mean: Smoothed values
# - smooth_lower/upper: 95% confidence bounds
print(result[['date', 'raw_mean', 'smooth_mean', 'smooth_lower', 'smooth_upper']])
```

## API Reference

### Main Function: `smooth_survey_timeseries`

Complete end-to-end smoothing pipeline.

**Parameters:**
- `df` (DataFrame): Individual survey responses with date, value, and weight columns
- `date_col` (str): Name of the date/period column
- `value_col` (str): Name of the survey response value column
- `weight_col` (str): Name of the survey weight column
- `q_initial` (float, optional): Initial guess for process variance (auto-detected if None)
- `confidence_level` (float, optional): Confidence level for intervals (default: 0.95)

**Returns:**
DataFrame with columns:
- `date`: Time period
- `n`: Sample size
- `n_eff`: Effective sample size
- `raw_mean`: Raw weighted mean
- `raw_se`: Standard error of raw mean
- `smooth_mean`: Smoothed estimate
- `smooth_se`: Standard error of smoothed estimate
- `smooth_lower`: Lower confidence bound
- `smooth_upper`: Upper confidence bound
- `Q_estimated`: Estimated process variance

### Supporting Functions

If you need more control, you can use the individual functions:

#### 1. `calculate_weighted_statistics`
Calculate weighted monthly statistics from individual responses.

```python
from variance_aware_smoothing import calculate_weighted_statistics

stats = calculate_weighted_statistics(df, 'date', 'value', 'weight')
# Returns: date, n, n_eff, value_mean, value_se, var_mean
```

#### 2. `estimate_process_variance`
Estimate Q using maximum likelihood.

```python
from variance_aware_smoothing import estimate_process_variance

Q_hat, info = estimate_process_variance(y, H)
print(f"Q = {Q_hat}, converged = {info['success']}")
```

#### 3. `kalman_filter` and `rts_smoother`
Low-level Kalman filtering functions.

```python
from variance_aware_smoothing import kalman_filter, rts_smoother

# Forward pass
mu_pred, P_pred, mu_filt, P_filt = kalman_filter(y, H, Q)

# Backward pass
mu_smooth, P_smooth = rts_smoother(mu_pred, P_pred, mu_filt, P_filt, Q)
```

## Use Cases

### 1. NPS Score Smoothing
```python
# Monthly NPS with varying sample sizes
nps_smoothed = smooth_survey_timeseries(
    df=nps_responses,
    date_col='survey_month',
    value_col='nps_score',
    weight_col='response_weight'
)
```

### 2. Customer Satisfaction (CSAT)
```python
# Weekly CSAT scores
csat_smoothed = smooth_survey_timeseries(
    df=csat_data,
    date_col='week',
    value_col='satisfaction_score',
    weight_col='weight'
)
```

### 3. Any Survey KPI
Works with any metric:
- Employee engagement scores
- Brand perception metrics
- Customer effort scores (CES)
- Custom satisfaction indices

## Integration Guide for SaaS Platform

### Database Schema Requirements

Your data should be structured as individual survey responses:

```sql
CREATE TABLE survey_responses (
    id BIGINT PRIMARY KEY,
    user_id BIGINT,
    survey_date DATE,
    metric_value FLOAT,
    response_weight FLOAT DEFAULT 1.0,
    metric_type VARCHAR(50)  -- e.g., 'NPS', 'CSAT', 'CES'
);
```

### API Endpoint Example

```python
from flask import Flask, jsonify, request
from variance_aware_smoothing import smooth_survey_timeseries
import pandas as pd

app = Flask(__name__)

@app.route('/api/smooth-metric', methods=['POST'])
def smooth_metric():
    data = request.get_json()
    
    # Fetch data from database
    query = f"""
        SELECT survey_date as date, 
               metric_value as value,
               response_weight as weight
        FROM survey_responses
        WHERE metric_type = %s
          AND survey_date BETWEEN %s AND %s
    """
    df = pd.read_sql(query, conn, params=[
        data['metric_type'],
        data['start_date'],
        data['end_date']
    ])
    
    # Apply smoothing
    result = smooth_survey_timeseries(df, 'date', 'value', 'weight')
    
    # Return as JSON
    return jsonify({
        'data': result.to_dict('records'),
        'Q_estimated': float(result['Q_estimated'].iloc[0])
    })
```

### Caching Strategy

Since smoothing is computationally light, you can:
1. **Real-time**: Compute on-demand for recent data (recommended for < 100 time periods)
2. **Pre-computed**: Cache results and update daily/hourly
3. **Hybrid**: Cache old data, compute recent periods on-demand

```python
def get_smoothed_metric(metric_type, start_date, end_date):
    # Check cache
    cached = redis.get(f'smooth:{metric_type}:{start_date}:{end_date}')
    if cached:
        return json.loads(cached)
    
    # Compute if not cached
    df = fetch_survey_data(metric_type, start_date, end_date)
    result = smooth_survey_timeseries(df, 'date', 'value', 'weight')
    
    # Cache for 1 hour
    redis.setex(
        f'smooth:{metric_type}:{start_date}:{end_date}',
        3600,
        result.to_json()
    )
    
    return result
```

## Performance Characteristics

- **Time complexity**: O(T) where T is number of time periods (not number of responses)
- **Typical performance**: 
  - 100 time periods: < 10ms
  - 1,000 time periods: < 100ms
  - 10,000 time periods: < 1s
- **Memory**: O(T) - minimal memory footprint

## Mathematical Model

The approach uses a **local level state space model**:

**Observation equation:**
```
y_t = μ_t + ε_t,  where ε_t ~ N(0, H_t)
```

**State equation:**
```
μ_t = μ_{t-1} + η_t,  where η_t ~ N(0, Q)
```

Where:
- `y_t`: Observed weighted mean at time t
- `μ_t`: True underlying metric (latent state)
- `H_t`: Measurement variance (from sample size)
- `Q`: Process variance (how much true metric can change)

## Parameter Interpretation

### Process Variance (Q)
- **Small Q** (< 0.1): Very smooth, assumes metric changes slowly
- **Medium Q** (0.1 - 1.0): Moderate smoothing, typical for monthly metrics
- **Large Q** (> 1.0): Light smoothing, allows rapid changes

The algorithm estimates Q automatically, but you can override if needed:

```python
result = smooth_survey_timeseries(
    df, 'date', 'value', 'weight',
    q_initial=0.5  # Force specific starting point
)
```

## Troubleshooting

### Issue: Optimization doesn't converge

**Solution**: Provide a different initial guess
```python
result = smooth_survey_timeseries(
    df, 'date', 'value', 'weight',
    q_initial=1.0  # Try different values: 0.1, 1.0, 10.0
)
```

### Issue: Not enough smoothing / Too much smoothing

**Diagnosis**: Check the estimated Q value
- If Q is very large, there's minimal smoothing
- If Q is very small, there's heavy smoothing

**Solution**: This usually means the data has these characteristics. But if you want to override:
```python
from variance_aware_smoothing import kalman_filter, rts_smoother, calculate_weighted_statistics, prepare_measurement_variance

# Manual control
stats = calculate_weighted_statistics(df, 'date', 'value', 'weight')
H = prepare_measurement_variance(stats['var_mean'].values, stats['value_mean'].values)
y = stats['value_mean'].values

# Set Q manually
Q = 0.5  # Your desired value
mu_pred, P_pred, mu_filt, P_filt = kalman_filter(y, H, Q)
mu_smooth, P_smooth = rts_smoother(mu_pred, P_pred, mu_filt, P_filt, Q)
```

### Issue: Too few time periods

The method works best with at least 10-12 time periods. For shorter series, consider:
- Using simpler smoothing (moving average)
- Waiting for more data
- Combining with other regularization

## Testing

The module includes built-in tests. Run:

```bash
python variance_aware_smoothing.py
```

This generates synthetic data and verifies all functions work correctly.

## Dependencies

```
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
```

## Support

For questions or issues, contact the data science team or refer to the full notebook documentation.

---

**Version**: 1.0  
**Last Updated**: 2025  
**License**: Internal Use

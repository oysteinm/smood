# Variance-Aware Kalman Smoothing for Survey Data

A production-ready Python implementation for smoothing time series survey data that automatically adapts to varying measurement uncertainty across time periods.

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Survey-based KPIs (NPS, CSAT, engagement scores) are inherently noisy because each time period has finite sample size. This noise:
- Creates unstable trends and difficult comparisons
- Adds variance without adding information
- Makes dashboards harder to interpret

**Key insight**: Not all observations are equally reliable. A month with 50 respondents should be smoothed more than a month with 500 respondents.

This repository provides a **variance-aware Kalman smoother** that:
- ✅ Automatically adapts smoothing based on sample size
- ✅ Provides optimal estimates with uncertainty quantification
- ✅ Uses data-driven parameter estimation (MLE)
- ✅ Handles missing data and irregular time series
- ✅ Works with any weighted survey data

## The Problem

Consider a monthly NPS survey:
- **January**: n=50 respondents → Raw mean = 45.2 ± 4.1
- **February**: n=500 respondents → Raw mean = 47.8 ± 1.3

The January estimate is much noisier (wider uncertainty). Traditional smoothing methods (moving averages, exponential smoothing) treat both observations equally, which is suboptimal.

## Our Solution

We use a **state space model** with time-varying measurement variance:

```
Observation equation:  y_t = μ_t + ε_t,  where ε_t ~ N(0, H_t)
State equation:        μ_t = μ_{t-1} + η_t,  where η_t ~ N(0, Q)
```

Where:
- **y_t**: Observed weighted mean at time t
- **μ_t**: True underlying metric (what we want to estimate)
- **H_t**: Measurement variance (computed from effective sample size)
- **Q**: Process variance (estimated via maximum likelihood)

The Kalman filter automatically gives more weight to observations with smaller H_t (larger sample sizes) and provides optimal smoothed estimates.

## Repository Contents

### 📓 `ics_smoothing_final.ipynb`
**Interactive Tutorial Notebook**

A complete Jupyter notebook demonstrating the methodology using real data from the University of Michigan Index of Consumer Sentiment (ICS).

**What's inside:**
- Mathematical framework and motivation
- Step-by-step walkthrough of the algorithm
- Data loading and preprocessing
- Weighted statistics calculation
- Kalman filtering and smoothing
- Visualization and diagnostics

**Best for:** Understanding the methodology, learning how it works, experimenting with parameters.

### 🐍 `variance_aware_smoothing.py`
**Production-Ready Python Module**

A clean, tested, production-ready implementation with no visualization dependencies.

**Core functions:**
- `smooth_survey_timeseries()` - One-line solution for end-to-end smoothing
- `calculate_weighted_statistics()` - Compute weighted means and measurement variance
- `estimate_process_variance()` - MLE estimation of optimal Q parameter
- `kalman_filter()` - Forward pass state estimation
- `rts_smoother()` - Backward pass for optimal smoothing

**Best for:** Integrating into production systems, APIs, data pipelines.

**Quick example:**
```python
from variance_aware_smoothing import smooth_survey_timeseries

result = smooth_survey_timeseries(
    df=survey_data,
    date_col='month',
    value_col='nps_score',
    weight_col='response_weight'
)

# Result includes raw_mean, smooth_mean, confidence bounds
```

### 📖 `IMPLEMENTATION_GUIDE.md`
**Comprehensive Integration Guide**

Detailed documentation for SaaS development teams.

**Contents:**
- API reference for all functions
- Integration patterns (Flask/Django examples)
- Database schema recommendations
- Caching strategies
- Performance characteristics
- Troubleshooting guide
- Parameter interpretation

**Best for:** Development teams integrating the smoothing into their platform.

### 📦 `requirements.txt`
**Dependencies**

Minimal dependencies:
```
numpy>=1.19.0
pandas>=1.1.0
scipy>=1.5.0
```

## Methodology

### Step 1: Calculate Weighted Statistics

For each time period t, from individual microdata (y_it, w_it):

**Weighted mean:**
```
ŷ_t = Σ(w_it × y_it) / Σ(w_it)
```

**Effective sample size** (accounts for unequal weights):
```
n_eff,t = (Σw_it)² / Σ(w_it²)
```

**Measurement variance** (standard error of the mean):
```
H_t = s²_w,t / n_eff,t
```

This gives us period-specific uncertainty: small sample → large H_t → more smoothing.

### Step 2: Estimate Process Variance

We estimate Q (how much the true metric can change period-to-period) using **maximum likelihood estimation**:

```
Q̂ = argmax_Q { log L(y | Q, H) }
```

This is data-driven: the algorithm finds the Q that best explains the observed patterns given the known measurement variances.

### Step 3: Kalman Filter (Forward Pass)

For each time period, we predict and update:

**Predict:**
```
μ̂_t|t-1 = μ̂_t-1|t-1
P_t|t-1 = P_t-1|t-1 + Q
```

**Update:**
```
K_t = P_t|t-1 / (P_t|t-1 + H_t)  (Kalman gain)
μ̂_t|t = μ̂_t|t-1 + K_t(y_t - μ̂_t|t-1)
P_t|t = (1 - K_t)P_t|t-1
```

The Kalman gain K_t automatically adapts: when H_t is large (small sample), K_t is small (trust the model more than the data).

### Step 4: RTS Smoother (Backward Pass)

We run a backward pass using all available data:

```
J_t = P_t|t / P_t+1|t
μ̂_t|T = μ̂_t|t + J_t(μ̂_t+1|T - μ̂_t+1|t)
P_t|T = P_t|t + J_t²(P_t+1|T - P_t+1|t)
```

This gives us the optimal estimate at each time point using all observations (past and future).

### Step 5: Confidence Intervals

We compute 95% confidence bounds:
```
[μ̂_t - 1.96√P_t, μ̂_t + 1.96√P_t]
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/variance-aware-smoothing.git
cd variance-aware-smoothing

# Install dependencies
pip install -r requirements.txt

# Test the module
python variance_aware_smoothing.py
```

## Quick Start

### Using the Python Module

```python
import pandas as pd
from variance_aware_smoothing import smooth_survey_timeseries

# Load your survey data (individual responses)
df = pd.DataFrame({
    'date': ['2024-01', '2024-01', '2024-02', ...],
    'score': [8, 9, 7, ...],
    'weight': [1.0, 1.2, 0.9, ...]
})

# Apply smoothing
result = smooth_survey_timeseries(
    df=df,
    date_col='date',
    value_col='score',
    weight_col='weight'
)

# View results
print(result[['date', 'raw_mean', 'smooth_mean', 'smooth_lower', 'smooth_upper']])
```

### Using the Jupyter Notebook

```bash
jupyter notebook ics_smoothing_final.ipynb
```

Run all cells to see the complete methodology applied to real ICS data.

## Use Cases

- **NPS (Net Promoter Score)** tracking with varying monthly sample sizes
- **CSAT (Customer Satisfaction)** with weekly or daily measurements
- **Employee engagement** surveys with quarterly or monthly cadence
- **Brand perception** metrics from market research
- **Any weighted survey KPI** where sample size varies over time

## Performance

- **Time complexity:** O(T) where T = number of time periods
- **Typical performance:** 
  - 100 periods: <10ms
  - 1,000 periods: <100ms
  - 10,000 periods: <1s
- **Memory:** O(T) - minimal footprint

## Advantages Over Alternatives

| Method | Adapts to Sample Size | Uncertainty Quantification | Parameter Estimation |
|--------|----------------------|----------------------------|---------------------|
| Moving Average | ❌ No | ❌ No | ⚠️ Manual window size |
| Exponential Smoothing | ❌ No | ⚠️ Limited | ⚠️ Manual alpha |
| LOESS | ❌ No | ⚠️ Bootstrap needed | ⚠️ Manual bandwidth |
| **This Method** | ✅ Yes | ✅ Yes | ✅ Automatic (MLE) |

## Mathematical Background

This implementation is based on:

1. **Kalman Filter** - Optimal recursive estimator for linear dynamic systems (Kalman, 1960)
2. **RTS Smoother** - Fixed-interval smoothing algorithm (Rauch, Tung, Striebel, 1965)
3. **Maximum Likelihood Estimation** - Data-driven parameter selection
4. **Weighted Survey Statistics** - Proper handling of survey weights and effective sample size

For theoretical details, see the notebook or these references:
- Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods*
- Särkkä, S. (2013). *Bayesian Filtering and Smoothing*

## Real-World Example: Index of Consumer Sentiment

The notebook demonstrates the method on the University of Michigan Index of Consumer Sentiment:

- **Data:** 47 years of monthly surveys (1978-2025)
- **Sample sizes:** 500-1,500 respondents per month
- **Result:** Smooth trends that preserve genuine changes while removing sampling noise
- **Validation:** High correlation with published index

## API Integration Example

```python
from flask import Flask, request, jsonify
from variance_aware_smoothing import smooth_survey_timeseries
import pandas as pd

app = Flask(__name__)

@app.route('/api/smooth', methods=['POST'])
def smooth_metric():
    data = request.json
    
    # Fetch from your database
    df = get_survey_data(
        metric=data['metric'],
        start=data['start_date'],
        end=data['end_date']
    )
    
    # Apply smoothing
    result = smooth_survey_timeseries(
        df=df,
        date_col='date',
        value_col='value',
        weight_col='weight'
    )
    
    return jsonify(result.to_dict('records'))
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Add tests for new functionality
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Open a Pull Request

## Testing

Run the built-in test suite:

```bash
python variance_aware_smoothing.py
```

This generates synthetic data and validates all functions.

## License

MIT License.

## Citation

If you use this code in your research or product, please cite:

```bibtex
@software{variance_aware_smoothing,
  title={Variance-Aware Kalman Smoothing for Survey Data},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/variance-aware-smoothing}
}
```

## Support

- 📖 **Documentation:** See `IMPLEMENTATION_GUIDE.md`
- 🐛 **Issues:** Open an issue on GitHub
- 💬 **Questions:** Start a discussion in the repository

## Acknowledgments

- Data: University of Michigan Surveys of Consumers
- Inspired by state space modeling literature and best practices in survey methodology
- Built for production use in SaaS platforms

---

**Made with ❤️ for data teams who care about statistical rigor**

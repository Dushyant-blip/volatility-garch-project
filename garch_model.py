import pandas as pd
import numpy as np
from arch import arch_model
import streamlit as st

def run_garch_model(returns, p=1, q=1, mean_model='Constant', vol_model='GARCH'):
    """
    Run a GARCH model on the provided returns data
    
    Parameters:
    -----------
    returns : Series
        Series containing returns data
    p : int
        GARCH p parameter
    q : int
        GARCH q parameter
    mean_model : str
        Mean model specification
    vol_model : str
        Volatility model specification
        
    Returns:
    --------
    model_result : ARCHModelResult
        Results from the GARCH model
    forecasts : Series
        Forecasted volatility
    volatility : Series
        Historical conditional volatility
    """
    # Handle mean model
    if mean_model == 'Zero':
        mean = 'Zero'
        lags = 0
    elif mean_model == 'Constant':
        mean = 'Constant'
        lags = 0
    elif mean_model == 'AR':
        mean = 'AR'
        lags = 1  # Default to AR(1)
    else:  # ARX
        mean = 'ARX'
        lags = 1  # Default to ARX(1)
    
    # Create and fit the model
    model = arch_model(
        returns,
        p=p,
        q=q,
        mean=mean,
        lags=lags,
        vol=vol_model
    )
    
    # Fit the model with robust standard errors
    model_result = model.fit(disp='off', update_freq=0, cov_type='robust')
    
    # Get conditional volatility
    volatility = model_result.conditional_volatility
    
    # Generate forecasts - we'll forecast 10 periods ahead
    forecasts = model_result.forecast(horizon=10)
    forecast_variance = forecasts.variance.iloc[-1].values
    
    # Create a series for the forecasted volatility
    last_date = returns.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=10,
        freq=pd.infer_freq(returns.index)
    )
    
    forecasted_vol = pd.Series(
        np.sqrt(forecast_variance),
        index=forecast_dates
    )
    
    return model_result, forecasted_vol, volatility

def simulate_paths(model_result, n_days=30, n_paths=100):
    """
    Simulate future returns paths based on GARCH model
    
    Parameters:
    -----------
    model_result : ARCHModelResult
        Results from GARCH model
    n_days : int
        Number of days to simulate
    n_paths : int
        Number of paths to simulate
        
    Returns:
    --------
    paths : ndarray
        Array of simulated paths
    """
    # Extract model parameters
    params = model_result.params
    
    # Initialize paths array
    paths = np.zeros((n_days, n_paths))
    
    # Get last variance
    last_var = model_result.conditional_volatility.iloc[-1]**2
    
    # Get GARCH parameters
    omega = params['omega']
    alpha = params['alpha[1]']
    beta = params['beta[1]']
    
    # Get mean model parameters
    if 'mu' in params:
        mu = params['mu']
    else:
        mu = 0
    
    # Simulation loop
    for i in range(n_days):
        if i == 0:
            var = omega + alpha * model_result.resid.iloc[-1]**2 + beta * last_var
        else:
            var = omega + alpha * paths[i-1, :]**2 + beta * var
        
        vol = np.sqrt(var)
        shocks = np.random.normal(0, 1, n_paths)
        paths[i, :] = mu + vol * shocks
    
    return paths

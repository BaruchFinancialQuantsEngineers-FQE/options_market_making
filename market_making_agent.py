#!/usr/bin/env python
# coding: utf-8

# In[ ]:


config = {
    # Underlying stock
    "initial_stock_price": 100.0,
    "annual_drift": 0.0,
    "annual_volatility": 0.20,

    # Call option
    "strike": 100.0,
    "time_to_expiry_years": 30 / 365,
    "risk_free_rate": 0.0,
    "contract_multiplier": 100,
    "as_gamma": 0.1,

    # Simulation clock
    "num_steps": 390,
    "step_minutes": 1,
    "trading_days_per_year": 252,

    # Starting account and limits
    "initial_cash": 10000,
    "initial_inventory": 0,
    "max_inventory": 10,
    "quote_size": 1,
    "half_spread": 0.10,
    "base_fill_rate": 0.5,
    "fill_sensitivity": 5.0,

    # Other
    "risk-free rate": 0.0413,
    "seed": 42,
}

def pnl(cash, mult, inventory, option_value, initial_cash):
    return (cash + mult*inventory*option_value) - initial_cash

def black_scholes(S_t, K, tau, r, sig):
    if tau <= 0:
        C = max(S_t - K, 0.0)
        P = max(K - S_t, 0.0)
        return C, P

    d_1 = (math.log(S_t/K) + (r + sig**2/2)*tau ) / (sig*math.sqrt(tau))
    d_2 = d_1 - sig*math.sqrt(tau)

    FV_C = S_t * norm.cdf(d_1) - K * math.exp(-r*tau) * norm.cdf(d_2)
    FV_P = K * math.exp(-r*tau) * norm.cdf(-d_2) - S_t * norm.cdf(-d_1)

    return FV_C, FV_P


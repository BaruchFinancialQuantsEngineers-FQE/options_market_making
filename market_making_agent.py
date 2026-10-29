#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
from scipy.stats import norm

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

def delta(option_0, option_f, stock_0, stock_f):
    return (option_f - option_0)/(stock_f - stock_0)

def simulate_stock_price(S_t, config, rng):
    mu = config["annual_drift"] 
    sig = config["annual_volatility"]
    dt = config["step_minutes"] / (390 * config["trading_days_per_year"])
    Z = rng.standard_normal()

    exp = (mu - (sig**2 / 2)) * dt + sig * math.sqrt(dt) * Z

    return S_t * math.exp(exp)

def strategy(C, config):
    hf = config["half_spread"]


    bid = max(C - hf, 0.0)
    ask = C + hf

    return bid, ask

def fill_simulator(C, bid, ask, config, rng):
    dt = config["step_minutes"]
    A = config["base_fill_rate"]
    k = config["fill_sensitivity"]

    distance_b = C - bid
    distance_a = ask - C

    lambda_b = A*math.exp(-k*distance_b)
    lambda_a = A*math.exp(-k*distance_a)

    p_bid = 1 - math.exp(-lambda_b*dt)
    p_ask = 1 - math.exp(-lambda_a*dt)

    bid_fill = rng.random() < p_bid
    ask_fill = rng.random() < p_ask

    return bid_fill, ask_fill

def execute_order(cash, inventory, bid, ask, bid_fill, ask_fill, config):
    size = config["quote_size"]
    mult = config["contract_multiplier"]
    lim = config["max_inventory"]
    executed = []

    if bid_fill and inventory + size <= lim and cash >= mult*size*bid:
        cash -= mult*size*bid
        inventory += size
        executed.append({"side": "buy", "price": bid,"quantity": size})


    if ask_fill and inventory - size >= -lim:
        cash += mult*size*ask
        inventory -= size
        executed.append({"side": "sell", "price": ask, "quantity": size})


    return cash, inventory, executed

def avellaneda_stoikov(time, S_t, C, inventory, tau, config):
    k = config["fill_sensitivity"]
    vol = config["annual_volatility"]
    r = config["risk_free_rate"]
    K = config["strike"]
    gamma = config["as_gamma"]

    d_1 = (math.log(S_t/K) + (r + vol**2/2)*tau) / (vol*math.sqrt(tau))
    delta = norm.cdf(d_1)

    sig = abs(delta) * S_t * vol

    session_time = config["num_steps"] * config["step_minutes"]
    remaining = max(session_time - time, 0.0)
    H = remaining / (390*config["trading_days_per_year"])

    r = C - inventory * gamma * sig**2 * H
    s = gamma * sig**2 * H + (2/gamma) * math.log1p(gamma/k)

    bid = max(r - s/2, 0.0)
    ask = max(r + s/2, 0.0)

    return bid, ask


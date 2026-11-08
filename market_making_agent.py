#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt


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

class InformedTraders:
    def __init__(self, V, P):
        # market value at end of day
        self.V = V            

        # current price
        self.price = P

    def _get_signal(self):
        sig = 0.4
        error = norm.rvs(loc=0, scale=0.4)
        S = self.V + error

        return S


    def fill_order(self, bid, ask, beta, max_size=10):

        # checks
        if bid > ask:
            raise ValueError("bid must be less than or equal to ask")
        if beta < 0 or max_size < 1:
            raise ValueError("beta must be nonnegative; max_size must be positive")

        S = self._get_signal()


        if S > ask:
            position = "buy"
            price = ask
            edge = S - ask

        elif S < bid:
            position = "sell"
            price = bid
            edge = bid - S

        else:
            return None

        quantity = min(int(beta * edge), max_size)

        if quantity == 0:
            return None
        else:
            return {
                "position": position,
                "price": price,
                "quantity": quantity,
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



def simulate_day(config):

    # Market
    stock_price = config["initial_stock_price"]
    tau = config["time_to_expiry_years"]

    # Account
    cash = config["initial_cash"]
    inventory = config["initial_inventory"]

    # Records
    history = []
    trades = []

    # Random shock
    rng = np.random.default_rng(config["seed"])
    mature_step = (config["step_minutes"]/390) / 365

    initial_pv = cash

    for step in range(config["num_steps"] + 1):
        call, put = black_scholes(stock_price, config["strike"], tau, config["risk_free_rate"], config["annual_volatility"])
        # bid, ask = strategy(call, config)
        bid, ask = avellaneda_stoikov(step*config["step_minutes"], stock_price, call, inventory, tau, config)

        pv = cash + config["contract_multiplier"] * inventory * call
        pnl = pv - initial_pv


        history.append({
            "minute": step * config["step_minutes"],
            "stock_price": stock_price,
            "tau": tau,
            "call_value": call,
            "put_value": put,
            "bid": bid,
            "ask": ask,
            "cash": cash,
            "inventory": inventory,
            "portfolio_value": pv,
            "P&L": pnl,
        })

        if step == config["num_steps"]:
            break

        bid_fill, ask_fill = fill_simulator(call, bid, ask, config, rng)
        cash, inventory, executed = execute_order(cash, inventory, bid, ask, bid_fill, ask_fill, config)

        for trade in executed:
            trade["minute"] = step * config["step_minutes"]
            trades.append(trade)

        stock_price = simulate_stock_price(stock_price, config, rng)
        tau = max(tau - mature_step, 0.0)



    return {
        "history": history,
        "trades": trades,
        "Data": {
            "PnL": history[-1]["P&L"],
            "Inventory": inventory,
            "Total Fills": len(trades),
        },
    }

result = simulate_day(config)
history = result["history"]
trades = result["trades"]

print(result["Data"])
def plot_it():
    minutes = [row["minute"] for row in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    for key, label in [
        ("call_value", "Call value"),
        ("bid", "Bid"),
        ("ask", "Ask"),
    ]:
        axes[0].plot(
            minutes, [row[key] for row in history], label=label
        )

    axes[0].set_ylabel("Price ($ per share)")
    axes[0].legend()

    axes[1].step(
        minutes,
        [row["inventory"] for row in history],
        where="post",
    )
    axes[1].axhline(config["max_inventory"], color="red", linestyle="--")
    axes[1].axhline(-config["max_inventory"], color="red", linestyle="--")
    axes[1].set_ylabel("Inventory (contracts)")

    axes[2].plot(minutes, [row["P&L"] for row in history])
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set_ylabel("P&L ($)")
    axes[2].set_xlabel("Minutes since open")

    plt.tight_layout()
    plt.show()
plot_it()


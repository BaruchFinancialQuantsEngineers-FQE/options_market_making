#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

    # informed traders parameters
    "arrival_rate": 0.05, #random.random(),
    "signal": 0.10,
    "kyle_beta": 10.0,
    "max_size": 1.0,

    # Other
    "risk-free rate": 0.0413,
    "customer_arrival": 0.45, #random.random(),
    "seed": 42,
}

class InformedTraders:
    def __init__(self, V, rng, sig=0.4):
        # market value at end of day
        self.V = V 
        self.sig = sig
        self.rng = rng


    def _get_signal(self):
        error = self.rng.normal(0, self.sig)
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

def execute_special_order(cash, inventory, order, config):
    if order is None:
        return cash, inventory, []

    mult = config["contract_multiplier"]
    limit = config["max_inventory"]
    price = order["price"]

    quantity = min(order["quantity"], config["quote_size"])

    if order["position"] == "buy":
        quantity = min(quantity, limit + inventory)

        if quantity <= 0:
            return cash, inventory, []

        cash += mult * quantity * price
        inventory -= quantity
        bot_side = "sell"

    else:

        quantity = min(quantity, limit - inventory)

        if price > 0:
            affordable = math.floor(cash / (mult * price))
            quantity = min(quantity, affordable)

        if quantity <= 0:
            return cash, inventory, []

        cash -= mult * quantity * price
        inventory += quantity
        bot_side = "buy"

    executed = [{
        "side": bot_side,
        "price": price,
        "quantity": quantity,
    }]

    return cash, inventory, executed

def glosten_milgrom(P_vh, mu, v_h, v_l):
    P_buy_vh = mu + (1-mu)*0.5
    P_buy_vl = (1-mu)*0.5

    num = P_buy_vh * P_vh
    denom = num + P_buy_vl * (1-P_vh)
    P_vh_buy = num / denom

    p_b = P_vh_buy

    ask = p_b*v_h + (1-p_b)*v_l

    P_sell_vh = 1 - P_buy_vh
    P_sell_vl = 1 - P_buy_vl

    num_sell = P_sell_vh * P_vh
    den_sell = num_sell + P_sell_vl * (1-P_vh)
    P_vh_sell = num_sell/den_sell

    p_s = P_vh_sell

    bid = (1-p_s)*v_l + p_s*v_h

    return bid, ask, P_vh_buy, P_vh_sell

def generate_customer_order(true_value, bid, ask, mu, rng):

    if rng.random() < mu:
        if true_value > ask:
            return "buy"
        elif true_value < bid:
            return "sell"
        return None

    return rng.choice(["buy", "sell"])  

def _get_terminal_values(n, stock_price, terminal_tau, rng, config):
    """
    initializes bot's initial beliefs for each terminal value of the option
    each value is equal
    """

    H = (config["num_steps"] * config["step_minutes"] / (390 * config["trading_days_per_year"]))

    Z = rng.standard_normal(n)

    possible_prices = stock_price * np.exp((config["annual_drift"] - 0.5 * config["annual_volatility"]**2) * H + config["annual_volatility"] * np.sqrt(H) * Z)

    possible_values = np.array([
        black_scholes(
            s,
            config["strike"],
            terminal_tau,
            config["risk_free_rate"],
            config["annual_volatility"],
        )[0]
        for s in possible_prices
    ])

    probabilities = np.full(n, 1.0 / n)

    return possible_values, probabilities

def _get_likelihoods(possible_values, bid, ask, mu, config):
    """
    gets likelihoods of the respective possible value in the possible values vector
    """
    possible_values = np.asarray(possible_values, dtype=float)

    beta = config["kyle_beta"]
    signal = config["signal"]

    if not 0 <= mu <= 1:
        raise ValueError("mu must be between 0 and 1")

    if bid > ask:
        raise ValueError("bid cannot exceed ask")

    if beta < 0 or signal < 0:
        raise ValueError("beta and signal must be nonnegative")   


    if beta == 0:
        p_buy = np.zeros_like(possible_values, dtype=float)
        p_sell = np.zeros_like(possible_values, dtype=float)
    else:
        buy_threshold = ask + 1 / beta
        sell_threshold = bid - 1 / beta

    if signal == 0:
        p_buy = (possible_values >= buy_threshold).astype(float)
        p_sell = (possible_values <= sell_threshold).astype(float)
    else:
        p_buy = norm.sf(
            buy_threshold, loc=possible_values, scale=signal
        )
        p_sell = norm.cdf(
            sell_threshold, loc=possible_values, scale=signal
        )

    P_buy_v = mu*p_buy + (1-mu)*0.5
    P_sell_v = mu*p_sell + (1-mu)*0.5

    return P_buy_v, P_sell_v

def _get_gm_quotes(possible_values, probabilities,
                   P_buy_v, P_sell_v, config):

    possible_values = np.asarray(possible_values)
    probabilities = np.asarray(probabilities)

    buy_weights = probabilities * P_buy_v
    sell_weights = probabilities * P_sell_v

    buy_denom = np.sum(buy_weights)
    sell_denom = np.sum(sell_weights)

    if buy_denom <= 0 or sell_denom <= 0:
        raise ValueError("Cannot condition on an impossible event")

    buy_probs = buy_weights / buy_denom
    sell_probs = sell_weights / sell_denom

    ask = np.dot(possible_values, buy_probs)
    bid = np.dot(possible_values, sell_probs)

    return bid, ask, buy_probs, sell_probs

def arrival_types(rng, config):
    informed = config["arrival_rate"]
    uninformed = config["customer_arrival"]
    steps = config["step_minutes"]

    lmbda = informed + uninformed
    informed_ratio = informed / lmbda

    n_customers = rng.poisson(lam=lmbda*steps)

    customers = []

    for n in range(n_customers):
        if rng.random() > informed_ratio:
            customers.append("uninformed")
        else:
            customers.append("informed")

    return customers

def update_beliefs(probabilities, likelihood):

    weights = probabilities * likelihood
    denom = np.sum(weights)

    if denom <= 0:
        raise ValueError("Observed an impossible event")

    return weights / denom

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
    rng = np.random.default_rng(seed=config["seed"])
    mature_step = (config["step_minutes"]/390) / 365

    stock_path = [stock_price]

    for hingadingadergen in range(config["num_steps"]):
        stock_path.append(
            simulate_stock_price(stock_path[-1], config, rng)
        )

    terminal_tau = max(
        tau - config["num_steps"] * mature_step,
        0.0,
    )

    mu = config["arrival_rate"] / (config["arrival_rate"] + config["customer_arrival"])

    terminal_call, _ = black_scholes(
        stock_path[-1],
        config["strike"],
        terminal_tau,
        config["risk_free_rate"],
        config["annual_volatility"],
    )

    informed = InformedTraders(
        V=terminal_call,
        rng=rng,
        sig=config["signal"],
    )

    initial_ov = black_scholes(
        stock_price,
        config["strike"],
        tau,
        config["risk_free_rate"],
        config["annual_volatility"],
    )[0]

    initial_pv = (cash + config["contract_multiplier"] * inventory * initial_ov)

    possible_values, probabilities = _get_terminal_values(100, stock_price, terminal_tau, rng, config)
    bid, ask = strategy(initial_ov, config)

    for step in range(config["num_steps"] + 1):
        call, put = black_scholes(stock_price, config["strike"], tau, config["risk_free_rate"], config["annual_volatility"])
        #bid, ask = strategy(call, config)
        #bid, ask = avellaneda_stoikov(step*config["step_minutes"], stock_price, call, inventory, tau, config)

        p_buy, p_sell = _get_likelihoods(possible_values, bid, ask, mu, config)
        bid, ask, buy_probs, sell_probs = _get_gm_quotes(possible_values, probabilities, p_buy, p_sell, config)

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
            "spread": ask - bid,
            "cash": cash,
            "inventory": inventory,
            "portfolio_value": pv,
            "P&L": pnl,
        })

        if step == config["num_steps"]:
            break

        customers = arrival_types(rng, config)
        for customer in customers:
            if customer == "uninformed":
                if rng.random() <= 0.5:
                    price = ask
                    quantity = config["quote_size"]
                    cash, inventory, executed = execute_order(cash, inventory, bid, ask, False, True, config)
                else:
                    price = bid
                    quantity = config["quote_size"]
                    cash, inventory, executed = execute_order(cash, inventory, bid, ask, True, False, config)

            else:
                order = informed.fill_order(bid, ask, config["kyle_beta"], config["max_size"])
                cash, inventory, executed = execute_special_order(cash, inventory, order, config)
            for trade in executed:
                p_buy, p_sell = _get_likelihoods(possible_values, bid, ask, mu, config)

                if trade["side"] == "sell":
                    probabilities = update_beliefs(probabilities, p_buy)

                elif trade["side"] == "buy":
                    probabilities = update_beliefs(probabilities, p_sell)

                trade["minute"] = step * config["step_minutes"]
                trade["source"] = customer
                trades.append(trade)

        stock_price = stock_path[step + 1]
        tau = max(tau - mature_step, 0.0)


    informed_trades = [
        trade for trade in trades
        if trade["source"] == "informed"
    ]


    return {
        "history": history,
        "trades": trades,
        "Data": {
            "PnL": history[-1]["P&L"],
            "Inventory": inventory,
            "Total Fills": len(trades),
            "Informed executions:": len(informed_trades),
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

terminal_value = result["history"][-1]["call_value"]

informed_markout = sum(
    (terminal_value - trade["price"])
    * (1 if trade["side"] == "buy" else -1)
    * trade["quantity"]
    * config["contract_multiplier"]
    for trade in result["trades"]
    if trade["source"] == "informed"
)

print(f"Bot terminal markout from informed trades: ${informed_markout:.2f}")
plot_it()


#!/usr/bin/env python
# coding: utf-8

# In[11]:


from market_making_agent import simulate_day, config
import matplotlib.pyplot as plt
import numpy as np

def evaluate_informed(config, num_days=100, arrival=0.0):
    if arrival == 0.0:
        config["arrival_rate"] = 0.0
    else:
        config["arrival_rate"] = arrival

    results = []
    fig, ax = plt.subplots(figsize=(11, 6))
    pnl_paths = []

    for seed in range(num_days):
        day_config = config.copy()
        day_config["seed"] = seed

        result = simulate_day(day_config)
        results.append(result["Data"])

        history = result["history"]

        minutes = [row["minute"] for row in history]
        pnls = [row["P&L"] for row in history]

        pnl_paths.append(pnls)

        ax.plot(minutes, pnls, alpha=0.25, linewidth=1)

    mean_pnl = np.mean(pnl_paths, axis=0)

    ax.plot(minutes, mean_pnl,color="black", linewidth=3,label="Mean P&L", zorder=10)

    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Minutes since open")
    ax.set_ylabel("P&L ($)")
    ax.set_title(f"AS with Informed Trades P&L for {num_days} simulated days")

    plt.tight_layout()
    plt.show()

    return results
results = evaluate_informed(config, 100, 0.5)

pnls = np.array([day["PnL"] for day in results])

print(f"Mean P&L: ${pnls.mean():.2f}")
print(f"P&L standard deviation: ${pnls.std(ddof=1):.2f}")
print(f"Worst day: ${pnls.min():.2f}")
print(f"Best day: ${pnls.max():.2f}")
print(f"Profitable days: {(pnls > 0).mean()}")


# In[ ]:





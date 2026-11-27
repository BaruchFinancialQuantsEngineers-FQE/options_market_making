#!/usr/bin/env python
# coding: utf-8

# In[11]:


from market_making_agent import _get_terminal_values, _get_public_likelihoods, _update_public_beliefs, _get_likelihoods, update_beliefs, config
import numpy as np

rng = np.random.default_rng(42)

terminal_tau = (
    config["time_to_expiry_years"]
    - config["num_steps"] * config["step_minutes"]
    / (390 * 365)
)

prices, values, probs = _get_terminal_values(
    5000, 100, terminal_tau, rng, config
)

old = _get_public_likelihoods(
    prices, 100, 0, config
)

new = _get_public_likelihoods(
    prices, 101, 100, config
)

new_probs = _update_public_beliefs(
    probs, old, new
)

old_fair_value = np.dot(values, probs)
new_fair_value = np.dot(values, new_probs)

print("Old fair value:", old_fair_value)
print("New fair value:", new_fair_value)

assert new_fair_value > old_fair_value
assert np.isclose(new_probs.sum(), 1)

print("TEST 1 PASSED")


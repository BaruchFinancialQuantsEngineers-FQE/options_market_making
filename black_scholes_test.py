#!/usr/bin/env python
# coding: utf-8

# In[1]:


import market_making_agent
from market_making_agent import black_scholes

def prints(call, put):
    print(f"   Call Option: {call}")
    print(f"   Put Option: {put}")

# At the money
call, put = black_scholes(100, 100, 30 / 365, 0, 0.20)
print("Test 1: ATM Call & ATM Put")
prints(call, put)

# In the Money and Out of the Money
call, put = black_scholes(120, 100, 60 / 365, 0.05, 0.25)
print("Test 2: ITM Call & OTM Put")
prints(call, put)

# Out of the Money and In the Money
call, put = black_scholes(80, 100, 60 / 365, 0.05, 0.25)
print("Test 3: OTM Call & ITM Put")
prints(call, put)

# Long Expiry
call, put = black_scholes(100, 100, 1.0, 0.05, 0.30)
print("Test 4: Long Expiration Date")
prints(call, put)

# High Volatility 
call, put = black_scholes(100, 100, 30 / 365, 0.02, 0.80)
print("Test 5: High Volatility")
prints(call, put)


# In[ ]:





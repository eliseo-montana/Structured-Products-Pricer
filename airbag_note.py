import numpy as np
import matplotlib.pyplot as plt
from math import exp

# ==========================
# 0. INPUTS (TO FILL)
# ==========================

N = 100_000        # initial investment of the client
S0 = 494.65       # stock price on 9 Dec 2025 (from MarketWatch)
r = 0.036          # risk-free rate (continuous), 
T = 402/365          # maturity in years
margin_target = 0.04   # bank's margin (4% of N)

# --- Option data from MarketWatch ---

K_put = 400 # strike of the put (K)
P0    = 19  # price of that put (per share)

K_call =  500  # strike of the call (K*)
C0     = 54  # price of that call (per share)


# ==========================
# 1. WE COMPUTE PARTICIPATION RATE p
# ==========================
# Formula for p with target margin m:
# p = [1 - m - e^{-rT} + P0 / K_put] * (K_call / C0)

p = (1.0 - margin_target - exp(-r * T) + P0 / K_put) * (K_call / C0)

# We don't want negative participation (just in case)
p = max(p, 0.0)

print("=== PRODUCT PARAMETERS ===")
print(f"Put strike K       = {K_put:.2f}")
print(f"Call strike K*     = {K_call:.2f}")
print(f"Participation p    = {p:.2%}")


# ==========================
# 2. CHECK REALIZED MARGIN
# ==========================
# Hedging cost for 1 note of notional N:
# cost = N e^{-rT} + (pN/K_call)*C0 - (N/K_put)*P0

bond_today   = N * exp(-r * T)
calls_today  = (p * N / K_call) * C0
puts_premium = (N / K_put) * P0    # we RECEIVE this, so we subtract it in the cost

cost = bond_today + calls_today - puts_premium
realized_margin = 1.0 - cost / N

print("\n*COST & MARGIN CHECK *")
print(f"Cost of hedge today   = {cost:,.2f} €")
print(f"Target margin          = {margin_target:.2%}")
print(f"Realized margin        = {realized_margin:.2%}")


# ==========================
# 3. PAYOFF FUNCTIONS
# ==========================

def airbag_payoff(ST, N, K_put, K_call, p):
    """
    Payoff at maturity of the airbag note using the option decomposition:
    Payoff = N - (N/K_put)*(K_put - ST)^+ + (pN/K_call)*(ST - K_call)^+
    """
    ST = np.array(ST)
    term_put  = (N / K_put) * np.maximum(K_put - ST, 0.0)
    term_call = (p * N / K_call) * np.maximum(ST - K_call, 0.0)
    return N - term_put + term_call

def classical_payoff(ST, N, S0):
    """
    Classical buy-and-hold: invest N in the stock at S0 and hold to maturity.
    """
    ST = np.array(ST)
    return N * (ST / S0)


# ==========================
# 4. PLOT PAYOFFS
# ==========================

ST_grid = np.linspace(0.4 * S0, 1.6 * S0, 400)

payoff_airbag  = airbag_payoff(ST_grid, N, K_put, K_call, p)
payoff_stock   = classical_payoff(ST_grid, N, S0)

plt.figure(figsize=(9, 6))

plt.plot(ST_grid, payoff_airbag, label="Airbag note", linewidth=2)
plt.plot(ST_grid, payoff_stock, "--", label="Classical stock investment")

plt.axvline(K_put,  color="grey",  linestyle=":", label="K (put strike)")
plt.axvline(K_call, color="black", linestyle=":", label="K* (call strike)")

plt.title("Airbag Note vs Stock Only")
plt.xlabel("Underlying price at maturity $S_T$")
plt.ylabel("Payoff (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

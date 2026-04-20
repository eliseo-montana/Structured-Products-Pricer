import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========= 0. INPUTS =========
# Market data (from MarketWatch)
S0 = 494.65        # stock price on 9 Dec 2025

# strikes + ask prices
# ===== option chain from MarketWatch ( maturity = 15/12/2027) =====

# Put strikes + ask prices
# put strikes + ask prices
puts_data = [
    #  K1,   ask_put
    (370.0, 10.90),
    (380.0, 12.30),
    (390.0, 14.00),
    (400.0, 19.00),
    (410.0, 20.00),
    (420.0, 22.00),
    (430.0, 21.20),
    (440.0, 25.50),
    (450.0, 30.00),
    (460.0, 33.00),
    (470.0, 38.00),
    (480.0, 39.80),
    (490.0, 42.60),
]

# Call strikes + ask prices
calls_data = [
    #  K2,   ask_call
    (500.0, 54.00),
    (510.0, 48.00),
    (520.0, 43.80),
    (530.0, 40.00),
    (540.0, 35.50),
    (550.0, 33.00),
    (560.0, 29.50),
    (570.0, 27.00),
    (580.0, 23.60),
    (590.0, 22.00),
]



puts  = pd.DataFrame(puts_data,  columns=["K_put",  "ask_put"])
calls = pd.DataFrame(calls_data, columns=["K_call", "ask_call"])

# Product / bank parameters 
N          = 100_000   # client investment
g          = 0.90      # principal guarantee 90%
margin_pct = 0.04 # bank margin: 3% of N
r          = 0.036  # risk-free rate 
T_days     = 402    # days to maturity
# =============================


# ===== PV of guaranteed amount =====
T = T_days / 365.0
PV = g * N / math.exp(r * T)

print(f"S0                 = {S0:.2f}")
print(f"PV of 0.9 N        = {PV:,.2f}")

# ===== Bank margin & option budget per share =====
bank_margin_abs = margin_pct * N
leftover_today  = N - PV - bank_margin_abs

budget_per_share = leftover_today * S0 / N   
print(f"Bank margin (abs)  = {bank_margin_abs:,.2f}")
print(f"Option budget/share= {budget_per_share:.2f}\n")


# ===== We try all put–call combinations, check inequality =====
rows = []

for _, p in puts.iterrows():
    K1 = p["K_put"]
    ask_put = p["ask_put"]
    if K1 >= S0:
        continue  # put should be below S0

    for _, c in calls.iterrows():
        K2 = c["K_call"]
        ask_call = c["ask_call"]
        if K2 <= S0:
            continue  # call should be above S0

        total_premium_per_share = ask_put + ask_call

        
        # [N - PV - BankMargin] * S0 / N >= askPut + askCall
        feasible = total_premium_per_share <= budget_per_share + 1e-8
        if not feasible:
            continue

        spread = K2 - K1
        rows.append({
            "K1_put": K1,
            "ask_put": ask_put,
            "K2_call": K2,
            "ask_call": ask_call,
            "spread": spread,
            "total_premium_per_share": total_premium_per_share
        })

feasible_pairs = pd.DataFrame(rows)

if feasible_pairs.empty:
    print("No feasible put–call combinations with current parameters.")
else:
    # ===== We pick the combination with smallest spread =====
    feasible_pairs = feasible_pairs.sort_values(
        ["spread", "total_premium_per_share"],
        ascending=[True, True]
    ).reset_index(drop=True)

    print("Feasible put–call combinations (sorted by smallest spread):")
    print(feasible_pairs.to_string(index=False))

    best = feasible_pairs.iloc[0]
    K1 = best["K1_put"]
    K2 = best["K2_call"]
    ask_put  = best["ask_put"]
    ask_call = best["ask_call"]

    print("\nChosen combination with smallest spread:")
    print(best)

    # ===== Number of options=====
    n = N / S0
    print(f"\nNumber of underlying shares (N/S0) = {n:.2f}")
    print("→ This is the quantity of each option to buy.\n")

    # ---------- Payoff plot using short PPPN formula ----------
    floor = g * N

    S_T = np.linspace(0.5 * S0, 1.5 * S0, 300)

    PPPN_payoff = np.where(
        S_T < K1,
        floor + (K1 - S_T) * n,             # downside region (put)
        np.where(
            S_T > K2,
            floor + (S_T - K2) * n,         # upside region (call)
            floor                           # flat region
        )
    )

    Stock_only = (S_T / S0) * N
    Risk_free  = np.full_like(S_T, floor)

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(S_T, PPPN_payoff, label="Product payoff (PPPN)", linewidth=2)
    plt.plot(S_T, Stock_only, label="Stock only", linestyle="--", linewidth=2)
    plt.plot(S_T, Risk_free, label=f"Risk-free {int(g*100)}% N", linestyle="--", linewidth=2)

    plt.xlabel("Stock price at maturity $S_T$")
    plt.ylabel("Payoff at maturity")
    plt.title("PPPN vs Stock-only ")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

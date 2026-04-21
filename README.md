
# Structured-Products-Pricer

Pricing and structuring of equity-linked products in Python, from the bank's perspective. Each product is built up from vanilla replication, includes the self-financing / margin constraints used on a structuring desk, and is illustrated on a real underlying with market option data.

**Reference trade setup** (used across both products currently published):

| Parameter | Value |
|---|---|
| Underlying | S&P Global Inc. (SPGI) |
| Issue date | 9 December 2025 |
| Maturity | 15 January 2027 (402 days) |
| Spot at issue (S₀) | $494.95 |
| Risk-free rate (1Y UST, cont. comp.) | 3.60% |
| Target bank margin | 4% |
| Notional example (N) | $100,000 |

---

## Products

### 1. Partially Principal Protected Note — `ppp_note.py`

A capital-at-risk note guaranteeing **90% of the initial notional** at maturity, with additional upside if the underlying ends either well below a lower strike K₁ or well above an upper strike K₂. The investor is exposed at maturity to a **long put K₁ + long call K₂** position on top of a zero-coupon bond locking in the 90% floor.

**Decomposition.** The bank replicates the payoff using three legs:

- **Zero-coupon bond:** PV = 0.9N · e^(−rT), locking in the 90% capital protection.
- **Long European put** with strike K₁, quantity N/S₀.
- **Long European call** with strike K₂, quantity N/S₀.

**Budget constraint (per share):**

$$\frac{\big(N - \text{PV}(0.9N) - \text{bank margin}\big)\cdot S_0}{N} \geq \text{ask}_{\text{Put}} + \text{ask}_{\text{Call}}$$

**What the pricer does.** Given the investment amount, protection level, bank margin, and an option chain, the program scans **all feasible put–call combinations**, filters those that respect the budget-per-share constraint, and sorts them by smallest spread (K₂ − K₁). The tightest spread maximises the client's probability of earning the enhanced payoff.

**Illustrative trade on SPGI (N = $100,000, 90% protection, 4% margin):** the PV of the protected amount is $86,501.38, leaving a budget-per-share of $47.01. A feasible combination is **put K₁ = $410 (ask $20) and call K₂ = $580 (ask $23.60)**, with 202 option units purchased per leg. Payoff profile versus a direct stock investment is in the output plots.

---

### 2. Airbag Note — `airbag_note.py`

A structured note targeted at **moderately bullish, capital-preservation-oriented investors**. The client keeps full nominal as long as the stock holds within a tolerance band around spot, absorbs cushioned losses on a sharp fall, and participates at rate *p* in the upside beyond the participation strike.

**Payoff at maturity:**

$$
\text{Payoff}(S_T) = 
\begin{cases}
N \cdot S_T / K & \text{if } S_T < K \quad \text{(cushioned downside)} \\
N & \text{if } K \leq S_T < K + c \quad \text{(protection plateau)} \\
N + p \cdot N \cdot \dfrac{S_T - (K+c)}{K+c} & \text{if } S_T \geq K + c \quad \text{(participation)}
\end{cases}
$$

where K = 0.8 · S₀ is the airbag level and K + c ≈ S₀ is the participation strike.

**Replication (bank side).** The note is hedged with:

- **Zero-coupon bond:** N · e^(−rT)
- **Short European put** struck at K, quantity N/K
- **Long European call** struck at K + c, quantity pN/(K + c)

**Margin-adjusted self-financing condition.** Given ask prices C₀ (call) and P₀ (put), the participation rate *p* is solved from:

$$N e^{-rT} + \frac{pN}{K+c} \cdot C_0 - \frac{N}{K} \cdot P_0 = N(1 - m)$$

so that the bank locks in the target margin *m* up-front.

**Illustrative trade on SPGI (K = $400, P₀ = $19, K + c = $500, C₀ = $54):** the pricer returns a **participation rate of 42.94%** while delivering a 4% up-front margin to the desk. The structure is fully covered down to ST = $400.

---

## Roadmap (work in progress)

The following products are under active development :


- **Variance Swap pricer — three complementary approaches.** The first leg, **static replication via the Carr–Madan formula** (strip of European options replicating realised variance), is already drafted (variance_swap_pricer_part1.py). Still to come:
  - **Monte Carlo under Heston stochastic volatility**, handling the Feller condition and the Heston Trap via the Lord–Kahl rotation count fix.
  - **VIX-style model-free calculation** (CBOE methodology), to cross-validate the two model-based prices.


- **Autocallable Note.** Path-dependent note with periodic observation dates, coupon memory, and early redemption trigger. Pricing via Monte Carlo under Black–Scholes and Heston dynamics.



## Conventions

- Time to maturity in years, with T = (days to maturity) / 365.
- Risk-free rate continuously compounded.
- Ask prices used on long legs, bid on short legs (where relevant), to remain on the conservative side of the bid–ask spread from the bank's point of view.
- Arbitrage-free, frictionless market assumed for all examples.



## Data sources (reference trades)

- Option chain: [MarketWatch — SPGI options](https://www.marketwatch.com/investing/stock/spgi/options)
- Risk-free rate: [Bloomberg — US government bonds](https://www.bloomberg.com/markets/rates-bonds/government-bonds/us)
- Spot and fundamentals: [Yahoo Finance](https://finance.yahoo.com/quote/SPGI/) / [Stock Analysis](https://stockanalysis.com/stocks/spgi/)



## Author

**Eliseo Montana** — MSc Actuarial Science & Financial Engineering, KU Leuven.  
[LinkedIn](https://www.linkedin.com/in/eliseo-montana/)

# Variance Swap : Analytical Appproximation
# Pricing vanilla calls under the Heston model, usinf FFT and the Carr-Madan formula

import numpy as np
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
S0 = 100
r  = 0.05
q  = 0.0
T  = 4/12 
def heston_charfct (u, T, S0, r, q, kappa, eta_var, theta, rho, v0):


    
    #Heston characteristic function of log(S_t), exactly as in the course slide.
    
    #Parameters
    #----------
    # u        : complex array — evaluation points (can be complex, as needed by Carr-Madan)
    # t        : float         — maturity
    # S0       : float         — spot
    # r, q     : float         — risk-free rate, dividend yield
    # kappa    : float         — speed of mean reversion
    # eta_var  : float         — long-run variance level (η in the course)
    # theta    : float         — vol-of-vol
    # rho      : float         — correlation
    #v0       : float         — initial variance (σ₀² in the course)
    
    
    i = 1j
    d = np.sqrt(
        (rho * theta * i * u - kappa)**2
        - theta**2 * (-i * u - u**2)
    )
    g = (kappa - rho * theta * i * u - d) / \
        (kappa - rho * theta * i * u + d)
    exp_neg_dT = np.exp(-d * T)
    
    term1 = i * u * (np.log(S0) + (r - q) * T)
    term2 = (eta_var * kappa / theta**2) * (
        (kappa - rho * theta * i * u - d) * T
        - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g))
    )
    term3 = (v0 / theta**2) * \
        (kappa - rho * theta * i * u - d) * \
        (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)
    
    return np.exp(term1 + term2 + term3)




# CARR-MADAN FORMULA + FFT PRICER

# 
#
# GOAL: compute call prices C(K, T) for many strikes at once.
#
# WHY WE NEED THIS:
#   The Heston model has no simple closed-form for call prices
#   (unlike Black-Scholes). But it DOES have a closed-form
#   characteristic function (Step 1 above).
#
#   Carr & Madan (1999) showed that the call price can be written as
#   a Fourier integral of the characteristic function, and that integral
#   can be evaluated using the Fast Fourier Transform (FFT).
#
# The Math (in 4 lines):
#
#   1) Start from:  C(K) = exp(-rT)/pi * integral of something with phi
#      Problem: this integral doesn't converge for ITM options.
#
#   2) Carr-Madan fix: multiply C(K) by exp(alpha*k) where k=log(K)
#      and alpha>0 is a "dampening" parameter. This makes the integral
#      converge for ALL strikes. We undo the multiplication at the end.
#
#   3) The modified (damped) call price has a Fourier representation:
#        exp(alpha*k)*C(k) = (1/pi) * integral_0^inf exp(-i*v*k) * psi(v) dv
#
#      where psi(v) involves the characteristic function phi.
#
#   4) We discretize this integral and use FFT to evaluate it at N
#      log-strikes simultaneously in O(N*log(N)) time.
#
# =============================================================

def carr_madan_fft(S0, r, q, T, kappa, eta_var, theta, rho, v0,
                   N=4096, alpha=1.5, eta_grid=0.25):
    """
    Price European calls at many strikes simultaneously using Carr-Madan + FFT.

    Parameters
    ----------
    S0, r, q, T                          : contract parameters
    kappa, eta_var, theta, rho, v0       : Heston model parameters
    N          : int    — number of FFT points (power of 2 for FFT speed)
    alpha      : float  — dampening parameter (typically 1.0 to 2.0)
    eta_grid   : float  — spacing of the integration grid in v-space
                          (NOT the Heston parameter eta_var!)

    Returns
    -------
    strikes : np.array of shape (N,) — strike prices
    calls   : np.array of shape (N,) — corresponding European call prices
    """
    i = 1j

    # =============================================================
    # PART A: BUILD THE TWO GRIDS
    # =============================================================
    #
    # The FFT connects two grids:
    #   - v-grid (integration / "frequency" space): where we evaluate psi
    #   - k-grid (log-strike space): where we read off call prices
    #
    # These grids are linked by the FFT constraint:
    #   lambda * eta_grid = 2*pi / N
    # Smaller eta_grid = finer v-grid = wider k-grid (and vice versa).

    # --- v-grid (integration grid) ---
    # v_j = eta_grid * j   for j = 0, 1, 2, ..., N-1
    # These are the points where we'll evaluate our integrand.
    # v_0 = 0,  v_1 = 0.25,  v_2 = 0.50,  ...  v_{N-1} = 0.25*(N-1)

    j_values = np.arange(N)          # array [0, 1, 2, ..., 4095]
    v = eta_grid * j_values          # array [0.0, 0.25, 0.50, ..., 1023.75]

    # --- lambda: the log-strike spacing (determined by the FFT constraint) ---
    # lambda = 2*pi / (N * eta_grid)
    # With N=4096, eta_grid=0.25:  lambda = 2*pi / 1024 ≈ 0.00614
    # This means our strikes will be very closely spaced (good for interpolation).

    lam = 2 * np.pi / (N * eta_grid)

    # --- k-grid (log-strike grid) ---
    # k_n = -b + lambda*n   for n = 0, 1, 2, ..., N-1
    # where b = lambda*N/2 centers the grid around k=0 (i.e. around K=S0=100).
    #
    # With our numbers: b = 0.00614 * 2048 ≈ 12.57
    # So k ranges from about -12.57 to +12.57
    # which means K = exp(k) ranges from exp(-12.57)≈0.000003 to exp(12.57)≈287,000
    # The interesting part is near k=0, i.e. K near 100.

    b = lam * N / 2                  # ≈ 12.57 (half-width of log-strike grid)
    k = -b + lam * j_values          # array of N log-strike values

    # =============================================================
    # PART B: THE CARR-MADAN INTEGRAND  psi(v)
    # =============================================================
    #
    # The Carr-Madan formula says:
    #
    #   C(k) = exp(-alpha*k) / pi * Re[ integral_0^inf  exp(-i*v*k) * psi(v) dv ]
    #
    # where psi(v) is:
    #
    #            exp(-r*T) * phi( v - (alpha+1)*i )
    #   psi(v) = -------------------------------------------
    #            alpha^2 + alpha - v^2 + i*(2*alpha+1)*v
    #
    # Let's build this piece by piece.

    # --- Numerator: exp(-r*T) * phi( v - (alpha+1)*i ) ---
    #
    # We evaluate the characteristic function at a COMPLEX argument:
    #   u_shifted = v - (alpha+1)*i
    #
    # Why complex? The "alpha dampening" effectively shifts the evaluation
    # point into the complex plane. For v=0: u_shifted = -(alpha+1)*i = -2.5i
    # For v=1: u_shifted = 1 - 2.5i.  And so on.
    #
    # The exp(-r*T) is the standard risk-neutral discount factor.

    u_shifted = v - (alpha + 1) * i     # array of complex values

    cf_values = heston_charfct(        # evaluate phi at each shifted point
        u_shifted, T, S0, r, q,
        kappa, eta_var, theta, rho, v0
    )

    numerator = np.exp(-r * T) * cf_values

    # --- Denominator: alpha^2 + alpha - v^2 + i*(2*alpha+1)*v ---
    #
    # This comes from the analytical Fourier transform of the damped payoff
    # function exp(alpha*k) * max(exp(k) - 1, 0).
    #
    # With alpha=1.5:
    #   alpha^2 + alpha = 2.25 + 1.5 = 3.75
    #   2*alpha + 1 = 4.0
    # So: denominator = 3.75 - v^2 + 4*i*v
    #
    # At v=0: denominator = 3.75 (real, nonzero — no singularity)
    # The denominator never hits zero for alpha > 0, which is the whole
    # reason the dampening trick works.

    denominator = alpha**2 + alpha - v**2 + i * (2 * alpha + 1) * v

    # --- Full integrand ---
    psi = numerator / denominator

    # =============================================================
    # PART C: SIMPSON'S RULE WEIGHTS
    # =============================================================
    #
    # We approximate the integral using Simpson's 1/3 rule instead of
    # the simple rectangle rule. Simpson fits a parabola through every
    # 3 consecutive points, giving much higher accuracy.
    #
    # The weight pattern is:
    #   j=0:             1/3     (first point: half-weight)
    #   j=1,3,5,...:     4/3     (odd points: high weight)
    #   j=2,4,6,...:     2/3     (even points > 0: medium weight)
    #
    # Compare to rectangle rule where all weights = 1.
    # Simpson typically gives 4th-order accuracy vs 1st-order for rectangles.

    simpson_w = np.zeros(N)
    simpson_w[0] = 1.0 / 3.0           # first point gets 1/3
    simpson_w[1::2] = 4.0 / 3.0        # odd indices (1,3,5,...) get 4/3
    simpson_w[2::2] = 2.0 / 3.0        # even indices (2,4,6,...) get 2/3

    # =============================================================
    # PART D: ASSEMBLE THE FFT INPUT VECTOR
    # =============================================================
    #
    # We want to compute, for each n = 0,...,N-1:
    #
    #   C(k_n) = exp(-alpha*k_n)/pi * Re[ SUM_{j=0}^{N-1}  exp(-i*v_j*k_n) * psi(v_j) * eta_grid * w_j ]
    #
    # Now substitute k_n = -b + lambda*n  and  v_j = eta_grid*j :
    #
    #   exp(-i*v_j*k_n) = exp(-i * eta_grid*j * (-b + lambda*n))
    #                    = exp(i*j*eta_grid*b) * exp(-i * eta_grid * lambda * j * n)
    #
    # Since eta_grid * lambda = 2*pi/N, the second factor becomes:
    #                    = exp(i*j*eta_grid*b) * exp(-i * 2*pi * j * n / N)
    #
    # The exp(-i*2*pi*j*n/N) part is EXACTLY what numpy's FFT computes!
    #
    # So we define:
    #   x_j = exp(i * v_j * b) * psi(v_j) * eta_grid * simpson_w_j
    #
    # Then:  FFT(x)[n] = SUM_j x_j * exp(-i*2*pi*j*n/N)
    #                   = SUM_j exp(i*v_j*b) * exp(-i*2*pi*j*n/N) * psi(v_j) * eta_grid * w_j
    #
    # which is exactly our integral sum!

    x = np.exp(i * b * v) * psi * eta_grid * simpson_w

    # Let's break down what each factor in x_j does:
    #   exp(i*b*v_j)  — phase shift that accounts for the log-strike grid
    #                    starting at -b instead of 0
    #   psi(v_j)      — the Carr-Madan integrand (contains the char. function)
    #   eta_grid      — the spacing dv for the numerical integration
    #   simpson_w_j   — Simpson weight for higher accuracy

    # =============================================================
    # PART E: RUN THE FFT
    # =============================================================
    #
    # numpy.fft.fft computes:
    #   Y[n] = SUM_{j=0}^{N-1}  x[j] * exp(-i * 2*pi * j * n / N)
    #
    # for n = 0, 1, ..., N-1.   This is O(N*log(N)) instead of O(N^2).

    fft_result = np.fft.fft(x)        # complex array of length N

    # =============================================================
    # PART F: EXTRACT CALL PRICES
    # =============================================================
    #
    # From the Carr-Madan formula:
    #   C(k_n) = exp(-alpha * k_n) / pi * Re[ FFT(x)[n] ]
    #
    # - exp(-alpha*k_n): undoes the dampening we introduced. Recall we
    #   multiplied C(K) by exp(alpha*k) to make the integral converge;
    #   now we divide it back out.
    #
    # - 1/pi: comes from the Fourier inversion theorem (the inverse
    #   Fourier transform has a 1/(2*pi) factor, and we only integrate
    #   over [0, inf) instead of (-inf, inf), giving 2/(2*pi) = 1/pi).
    #
    # - Re[...]: the call price is a real number. Any imaginary part
    #   in the FFT output is just numerical noise from discretization.

    call_prices = (np.exp(-alpha * k) / np.pi) * fft_result.real

    # =============================================================
    # PART G: CONVERT LOG-STRIKES TO ACTUAL STRIKES
    # =============================================================
    # k_n = log(K_n)  =>  K_n = exp(k_n)

    strikes = np.exp(k)

    return strikes, call_prices

data = pd.read_excel(r"C:\Users\elise\OneDrive\Bureau\Pricer_Variance_Swap\OptionsData2FE.xlsx")
market_strikes    = data["Strikes"].values      # shape (720,)
market_maturities = data["Maturities"].values   # shape (720,)
market_prices     = data["Prices"].values        # shape (720,)


def objective(params):
    kappa, eta_var, theta, rho, v0 = params

    # Nelder-Mead doesn't support bounds natively, so we enforce
    # parameter constraints manually: return a large penalty value
    # whenever the optimizer tries physically impossible parameters.
    if kappa <= 0 or eta_var <= 0 or theta <= 0 or v0 <= 0:
        return 1e10
    if rho <= -1 or rho >= 1:
        return 1e10

# We get the 12 unique maturities. The idea: instead of running 720 separate FFTs,
# we run one FFT per maturity (12 total).Each FFT gives us prices at all 60 strikes for that maturity (much faster)

    unique_T = np.unique(market_maturities)
    model_prices = np.zeros_like(market_prices)
    
# We create an empty array of size 720 to store the model prices that we'll compute (for each of the 720 options

    for T_val in unique_T:
        mask = market_maturities == T_val
        K_for_this_T = market_strikes[mask]
        strikes_fft, calls_fft = carr_madan_fft(
            S0, r, q, T_val, kappa, eta_var, theta, rho, v0
        )
        valid = (strikes_fft > 1) & (calls_fft > 0) & (strikes_fft < 5 * S0)
        model_prices[mask] = np.interp(
            K_for_this_T, strikes_fft[valid], calls_fft[valid]
        )

    rmse = np.sqrt(np.mean((model_prices - market_prices)**2))
    return rmse

x0 = [0.5, 0.04, 0.3, -0.75, 0.04]

bounds = [(1e-4, 10), (1e-4, 1), (1e-4, 2), (-0.999, 0.999), (1e-4, 1)]

result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})


optimal_params = result.x
kappa_opt, eta_opt, theta_opt, rho_opt, v0_opt = optimal_params


print(f"Optimal parameters:")
print(f"  kappa = {kappa_opt:.4f}")
print(f"  eta   = {eta_opt:.4f}")
print(f"  theta = {theta_opt:.4f}")
print(f"  rho   = {rho_opt:.4f}")
print(f"  v0    = {v0_opt:.4f}")
print(f"  RMSE  = {result.fun:.6f}")

# where result.fun is the final objection function value (our best RMSE),
#  The name .fun is just scipy's shorthand for "function value at the optimum."


K_var = ((1 - np.exp(-kappa_opt * T)) / (kappa_opt * T)) * (v0_opt - eta_opt) + eta_opt
print(f"Fair variance strike (Approach 1): K_var = {K_var:.6f}")
print(f"Implied volatility: {np.sqrt(K_var)*100:.2f}%")



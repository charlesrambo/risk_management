# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm 
from scipy.optimize import fsolve

# Define function to get d_1 and d_2
def get_d(V, T, K, r, sigma_V):
    """
    Obtain the inputs of the normal CDF, d_1 and d_2, within the Black-Scholes formula.
    Formula on page 604 of "Options, Futures, and Other Derivatives" 9th ed. by Hull.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    """
    
    try:
        # Define d_1 and d_2 within the Black-Scholes formula
        d_1 = (np.log(V/K) + (r + 0.5 * sigma_V**2) * T)/(sigma_V * np.sqrt(T))
        d_2 = d_1 - sigma_V * np.sqrt(T)
        
    except:
        # If simga_V is 0, then results deterministic; d_1 = 3 close enough to deterministic
        d_1 = 3
        d_2 = d_1  - sigma_V * np.sqrt(T)
    
    return d_1, d_2

# Create function to get the Merton model distance to default
def distance(V, T, K, r, sigma_V):
    """
    The Merton model distance to default. Corresponds to d_2 in the Black-Scholes framework.
    """
    
    # The Merton model distance to default is d_2
    _, distance = get_d(V, T, K, r, sigma_V)
        
    return distance 

# Define function to get default probability
def prob_default(V, T, K, r, sigma_V):
    """
    The probability of default using the Merton model. Corresponds to N(-d_2) in the Black-Scholes framework.
    """
    
    # Get d_2
    _, d_2 = get_d(V, T, K, r, sigma_V)
    
    # Get probability
    prob = norm.cdf(-d_2)
    
    return prob

# Define function to get call price
def call(V, T, K, r, sigma_V):
    """
    Value of a call option under the Black-Scholes framework. Corresponds to the value of equity under the Merton model.
    Formula on page 604 of "Options, Futures, and Other Derivatives" 9th ed. by Hull.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    """
    
    # Get d_1 and d_2
    d_1, d_2 = get_d(V, T, K, r, sigma_V)
    
    # Black-Scholes formula for call
    C = V * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    
    return C


# Define function to get put price
def put(V, T, K, r, sigma_V, phi = 1):
    """
    Value of a put option under the Black-Scholes framework. 
    Corresponds to value of insurance required to make debt risk free.
    Formula on page 604 of "Options, Futures, and Other Derivatives" 9th ed. by Hull.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    """
    
    # Get d_1 and d_2
    d_1, d_2 = get_d(V, T, K, r, sigma_V)
    
    # Use the Black-Scholes put formula
    P = K * np.exp(-r * T) * norm.cdf(-d_2) - phi * V * norm.cdf(-d_1) 
    
    return P


# Up-and-in put option
def put_ui(V, T, K, H, r, sigma_V, phi = 1):
    """
    Value of an up-and-in put option under the Black-Scholes framework. 
    Formula on page 605 of "Options, Futures, and Other Derivatives" 9th ed. by Hull.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    H: Value of the firm which activates the up-and-in option.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    """ 
    
    λ = (r + 0.5 * sigma_V**2)/sigma_V**2
    
    y = np.log(H**2/(V * K))/(sigma_V * np.sqrt(T)) + λ * sigma_V * np.sqrt(T)
    
    put_ui = -phi * V * (H/V)**(2 * λ) * norm.cdf(-y) + K * np.exp(-r * T) * (H/V)**(2 * λ - 2) * norm.cdf(-y + sigma_V * np.sqrt(T))
    
    return put_ui


# Up-and-out put option
def put_uo(V, T, K, H, r, sigma_V, phi = 1):
    """
    Value of an up-and-out put option under the Black-Scholes framework. 
    Formula on page 605 of "Options, Futures, and Other Derivatives" 9th ed. by Hull.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    H: Value of the firm which deactivates the up-and-out option.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    """  
      
    # put = put_uo + put_ui
    put_uo = put(V, T, K, r, sigma_V, phi) - put_ui(V, T, K, H, r, sigma_V, phi)
    
    return put_uo


# Define function to get the probability of default if it's a barrier option
def prob_default_mc(V, T, K, r, sigma_V, trails = 10000, steps = None):
    """
    Probability of default if possible for some values of t prior to maturity. 
    Default occurs if the value of the firm is below K at time Δt, 2Δt, ..., or T = steps · Δt.
    If T is measured in years, steps = floor(12 * T + 1) corresponds to possibility of default at each monthly payment.
    Probability obtained using Monte Carlo simulation.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    trails: Monte Carlo simulations used to obtain approximation; antithetic values also considered.
    steps: Number of times the firm can default within each simulation.
    """  
    
    # If steps is None, set it equal to floor(12 * T + 1)
    if steps is None:
        
        # Corresponds to monthly payments if T in years
        steps = int(12 * T + 1)
        
    # Use the number of steps to obtain Δt
    delta_t = T/steps
    
    # Set the random seed
    np.random.seed(0)
    
    # Simulate the Brownian motions
    bm = norm.rvs(loc = 0, scale = np.sqrt(delta_t), size = (trails, steps))
    
    # Consider antithetic cases
    bm = np.concatenate([bm, -bm], axis = 0)
    
    # Create an array of values
    values = np.repeat(V, repeats = 2 * trails * (steps + 1))
    
    # Reshape; one simulation each row and one step each columns
    values = np.reshape(values, newshape = (2 * trails, steps + 1))
    
    # Initialize V_old
    V_old = values[:, 0]
    
    # Loop over the steps but NOT simulations
    for i in range(1, steps + 1):
        
        # Record the first result
        V_new = V_old + V_old * (r - 0.5 * sigma_V**2) * delta_t + sigma_V * V_old * bm[:, i - 1]
        
        # Add it to the array
        values[:, i] = V_new 
        
        # The new becomes the old
        V_old = V_new
    
    # Calculate the fraction of observations where the lower barrier was breached
    prob = np.mean(np.sum(values < K, axis = 1) > 0) + 1e-5
    
    return prob


# Define function to get spread
def spread(V, T, K, r, sigma_V, phi = 1):
    """
    CDS spread given default only possible at T. Uses the Black-Scholes framework.
    Value tends to be below market CDS spreads.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    """    
    
    # Get G, the value of insurance
    G_val = put(V, T, K, r, sigma_V, phi)
    
    return -1/T * np.log((K * np.exp(-r * T) - G_val)/(K * np.exp(-r * T)))


# Define function to get spread for barrier option
def spread_mc(V, T, K, r, sigma_V, phi = 1, trails = 10000, steps = None):
    """
    CDS spread if default possible for some values of t prior to maturity. 
    Default occurs if the value of the firm is below K at time Δt, 2Δt, ..., or T = steps · Δt.
    If T is measured in years, steps floor(12 * T + 1) corresponds to monthly payments.
    In the case of default, partial payment assumed to be made at time T.
    Probability obtained using Monte Carlo simulation.
    
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    trails: Monte Carlo simulations used to obtain approximation; antithetic values also considered.
    steps: Number of times the firm can default within each simulation.
    """  
    
    # If steps is None, set it equal to floor(12 * T + 1)
    if steps is None:
        
        # Corresponds to monthly payments if T in years
        steps = int(12 * T + 1)
    
    # Use the number of steps to obtain Δt
    delta_t = T/steps
    
    # Simulate the Brownian motions
    bm = norm.rvs(loc = 0, scale = np.sqrt(delta_t), size = (trails, steps))
    
    # Consider antithetic cases
    bm = np.concatenate([bm, -bm], axis = 0)
    
    # Create an array of values
    values = np.repeat(V, repeats = 2 * trails * (steps + 1))
    
    # Reshape; one simulation each row and one step each columns
    values = np.reshape(values, newshape = (2 * trails, steps + 1))
    
    # Initialize V_old
    V_old = values[:, 0]
    
    # Loop over the steps but NOT the simulations
    for i in range(1, steps + 1):
        
        # Record the first result
        V_new = V_old + V_old * (r - 0.5 * sigma_V**2) * delta_t + sigma_V * V_old * bm[:, i - 1]
        
        # Add it to the array
        values[:, i] = V_new 
        
        # The new becomes the old
        V_old = V_new
    
    # Create function to get the PV of cash flows under each trail
    def get_PV(row):
        
        try:
            
            # Get the first column where the firm value dropped below K
            col = np.where(row < K)[0][0]
        
            # Record the value
            value = row[col]
        
            # Fraction of value retained given default, times value given default, times discount factor
            PV = phi * value * np.exp(-r * T)
            
        except:
            
            # Never crossed below K so simply discount back
            PV = K * np.exp(-r * T)
          
        return PV

    # Apply the get_PV function to each row
    debt_vals = np.apply_along_axis(get_PV, 1, values)  
    
    try:
        
        # Compute the spread
        return -1/T * np.log(np.mean(debt_vals)/(K * np.exp(-r * T)))
    
    except:
        
        return np.nan

# Create function to construct spread given rachet up of capital structure
def spread_das(V, T, K_1, K_2, H, r, sigma_V, phi = 1):
    """
    CDS spread if default possible only at time T, but firm increases debt if its value goes above barrior. 
    Inspired by the Journal of Banking & Finance article "Credit spreads with dynamic debt" by Das and Kim.
    
    V: Value of the firm.
    T: Time until maturity.
    K_1: Strike price if firm does not increase its debt level.
    K_2: Strike price if firm increases its debt level.
    H: Value of the firm which triggers it to increase its debt level from K_1 to K_2.
    r: Risk-free rate.
    sigma_V: Firm volatility. 
    phi: Fraction of firm's value retained in the case of default.
    """ 
      
    G_val = put_uo(V, T, K_1, H, r, sigma_V, phi) + K_1/K_2 * put_ui(V, T, K_2, H, r, sigma_V, phi)
    
    if G_val < K_1 * np.exp(-r * T):
        
        return -1/T * np.log((K_1 * np.exp(-r * T) - G_val)/(K_1 * np.exp(-r * T)))
    
    else:
        
        return np.nan
    

# Create the objective function that we would like to minimize to obtain sigma_V
def obj(sigma_V, sigma_E, V, T, K, r):
    """
    Objective function to be minimized. Used to obtain the volatility of the firm's value.
    
    sigma_V: Firm volatility.
    sigma_E: Equity volatility.
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    """
    
    # Get d_1 for the calculation
    d_1, _ = get_d(V, T, K, r, sigma_V)
    
    # Calculate value of equity
    E = call(V, T, K, r, sigma_V)
    
    return E * sigma_E - V * sigma_V * norm.cdf(d_1)


# Create function to obtain sigma_V
def get_sigma_V(sigma_E, V, T, K, r):
    """
    Obtain the volatility of the firm's value. Uses the Black-Scholes framework and fsolve.
    
    sigma_E: Equity volatility.
    V: Value of the firm.
    T: Time until maturity.
    K: Strike price; related to the firm's debt level.
    r: Risk-free rate.
    """     
    
    # Use solver to find sigma_V
    result = fsolve(obj, x0 = 0.5 * sigma_E, args = (sigma_E, V, T, K, r))
    
    if np.abs(obj(result, sigma_E, V, T, K, r)) < 10**-3:    
        
        return result   
    
    else:       
        
        # Assume vol of assets same as vol of market equity
        E = call(V, T, K, r, sigma_E)
        
        # Assume Delta = 0.80
        Delta = 0.80
        
        return sigma_E * E/(Delta * V)


# Create function to obtain suggested K value
def get_K(CL, NCL, T = 1, r = 0):
    """
    If the value of the firm drops below K at maturity, then the firm is considered to be in default under the Merton model.
    Formula overweights current liabilities because they must be paid within the year or one business cycle.
    
    CL: Current liabilities.
    NCL: Noncurrent liabilities
    T: Time until maturity.
    r: Risk-free rate.
    """ 
    
    # Over-weight current liability
    K = (CL + 0.5 * NCL) * np.exp(r * (T - 1))
    
    return K

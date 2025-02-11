import numpy as np
from scipy.stats import norm
import pandas as pd

class BS_transac_costs():
    
    def __init__(self, S, K, T, sigma, r, costs, q=0, option_type='Call'):
        self.S = S
        self.K = K
        self.T = T
        self.vol = sigma 
        self.rate = r
        self.div = q
        self.option_type = option_type
        self.costs = costs
    
    def pricing_BS(self, St, t):
        
        St = np.asarray(St)
        t = np.asarray(t)

        price = np.zeros_like(St, dtype=float)
        
        # At maturity
        at_maturity = t >= self.T
        if self.option_type.lower() == 'call':
            price[at_maturity] = np.maximum(St[at_maturity] - self.K, 0)
        elif self.option_type.lower() == 'put':
            price[at_maturity] = np.maximum(self.K - St[at_maturity], 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Before maturity
        before_maturity = ~at_maturity
        if self.option_type.lower() == 'call':
            d1 = (np.log(St[before_maturity] / self.K) + (self.rate - self.div + 0.5 * self.vol ** 2) * (self.T - t[before_maturity])) / (self.vol * np.sqrt(self.T - t[before_maturity]))
            d2 = d1 - self.vol * np.sqrt(self.T - t[before_maturity])
            price[before_maturity] = (St[before_maturity] * np.exp(-self.div * (self.T - t[before_maturity])) * norm.cdf(d1) -
                                       self.K * np.exp(-self.rate * (self.T - t[before_maturity])) * norm.cdf(d2))
        elif self.option_type.lower() == 'put':
            d1 = (np.log(St[before_maturity] / self.K) + (self.rate - self.div + 0.5 * self.vol ** 2) * (self.T - t[before_maturity])) / (self.vol * np.sqrt(self.T - t[before_maturity]))
            d2 = d1 - self.vol * np.sqrt(self.T - t[before_maturity])
            price[before_maturity] = (self.K * np.exp(-self.rate * (self.T - t[before_maturity])) * norm.cdf(-d2) -
                                       St[before_maturity] * np.exp(-self.div * (self.T - t[before_maturity])) * norm.cdf(-d1))
        
        return price 
    
    def Delta(self, St, t):

        St = np.asarray(St)
        t = np.asarray(t)

        delta = np.zeros_like(St, dtype=float)

        # Identify where t >= T (at maturity)
        at_maturity = t >= self.T

        if self.option_type.lower() == 'call':
            delta_maturity = np.where(St > self.K, 1.0,
                               np.where(St < self.K, 0.0, 0.5))
            
            valid = t < self.T
            d1 = (np.log(St[valid] / self.K) + (self.rate - self.div + 0.5 * self.vol ** 2) * (self.T - t[valid])) / (self.vol * np.sqrt(self.T - t[valid]))
            delta_before = np.exp(-self.div * (self.T - t[valid])) * norm.cdf(d1)

            delta[at_maturity] = delta_maturity[at_maturity]
            delta[valid] = delta_before

        elif self.option_type.lower() == 'put':
            delta_maturity = np.where(St < self.K, -1.0,
                               np.where(St > self.K, 0.0, -0.5))

            valid = t < self.T
            d1 = (np.log(St[valid] / self.K) + (self.rate - self.div + 0.5 * self.vol ** 2) * (self.T - t[valid])) / (self.vol * np.sqrt(self.T - t[valid]))
            delta_before = np.exp(-self.div * (self.T - t[valid])) * (norm.cdf(d1) - 1)

            delta[at_maturity] = delta_maturity[at_maturity]
            delta[valid] = delta_before
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return delta   
    
    def price_trajectory(self, frequence):
        frequence = frequence.lower()
        if frequence == 'minute':
            steps = int(self.T * 252 * 6.5 * 60)
        elif frequence == 'hourly':
            steps = int(self.T * 252 * 6)
        elif frequence == 'daily':
            steps = int(self.T * 252)
        elif frequence == 'weekly':
            steps = int(self.T * 50)
        else: 
            raise ValueError("frequence must be 'minute', 'hourly', 'daily' or 'weekly'")  
        
        dt = self.T / steps
        Z = np.random.standard_normal(steps)
        log_returns = (self.rate - 0.5 * self.vol ** 2) * dt + self.vol * np.sqrt(dt) * Z
        log_S = np.log(self.S) + np.cumsum(log_returns)
        S = np.exp(log_S)
        S = np.insert(S, 0, self.S)
        return S
    
    def Hedging(self, frequence):
        frequence = frequence.lower()
        if frequence == 'minute':
            steps = int(self.T * 252 * 6.5 * 60)
        elif frequence == 'hourly':
            steps = int(self.T * 252 * 6)
        elif frequence == 'daily':
            steps = int(self.T * 252)
        elif frequence == 'weekly':
            steps = int(self.T * 50)
        else: 
            raise ValueError("frequence must be 'minute', 'hourly', 'daily' or 'weekly'")
        
        S = self.price_trajectory(frequence)
        dt = self.T / steps 

        Delta = np.zeros(steps + 1)
        Buy = np.zeros(steps + 1)
        Depenses = np.zeros(steps + 1)
        Portfolio = np.zeros(steps + 1)
        Value = np.zeros(steps + 1)

        Delta[0] = self.Delta(self.S, 0)
        Buy[0] = Delta[0]
        Depenses[0] = self.pricing_BS(self.S, 0) - (1+self.costs) * Buy[0] * S[0]
        Portfolio[0] = Delta[0] * S[0]
        Value[0] = Depenses[0] + Portfolio[0]

        times = np.linspace(0, self.T, steps + 1)
        Delta[1:] = self.Delta(S[1:], times[1:])
        
        for s in range(1, steps + 1):
            # Apply interest based on the sign of Depenses
            if Depenses[s-1] >= 0:
                Depenses[s] = Depenses[s-1] * np.exp(self.rate * dt)
            else:
                Depenses[s] = Depenses[s-1] * np.exp(-self.rate * dt)
            

            Buy[s] = Delta[s] - Delta[s-1]
            Depenses[s] -= (1+self.costs) * Buy[s] * S[s]
            Portfolio[s] = Delta[s] * S[s]
            Value[s] = Depenses[s] + Portfolio[s]
        
        # Handle option payoff at maturity
        if self.option_type.lower() == 'call':
            payoff = max(S[-1] - self.K, 0)
        elif self.option_type.lower() == 'put':
            payoff = max(self.K - S[-1], 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Adjust the final portfolio value by subtracting the option payoff
        Value[-1] -= payoff

        hedge = pd.DataFrame({
            'Price': S,
            'Delta': Delta,
            'Buy': Buy,
            'Depenses': Depenses,
            'Portfolio': Portfolio,
            'Value': Value
        })
        
        return hedge
    
    def MonteCarlo(self, frequence, number):
        Sims = []
        for _ in range(number):
            hedge = self.Hedging(frequence)
            Sims.append(hedge['Value'].values)
        return Sims

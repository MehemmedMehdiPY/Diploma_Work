import numpy as np
import sys
sys.path.append("../")
from constants.constants import Data

class KineticModels(Data):
    """
    Class to store methods for kinetic modeling. 
    At the moment, only Rase (1977) is available for use.
    """
    def __init__(self, model: str = "rase"):
        super().__init__()
        if model == "rase":
            self.kinetic_model = self.Rase
        else:
            raise NameError("No model named {} available. Choose one of these: rase".format(model))

        self.func = lambda T, P, X: (self.kinetic_model(T, P, X)) # / 1000 / 2.3 * np.exp(-X)
        
        self.Tc = np.array([
            self.Tc["CO"], self.Tc["H2O"], self.Tc["CO2"], self.Tc["H2"],
            self.Tc["Ar"], self.Tc["N2"], self.Tc["CH4"]
        ])
        self.Pc = np.array([
            self.Pc["CO"], self.Pc["H2O"], self.Pc["CO2"], self.Pc["H2"],
            self.Pc["Ar"], self.Pc["N2"], self.Pc["CH4"]
        ])
        self.w = np.array([
            self.w["CO"], self.w["H2O"], self.w["CO2"], self.w["H2"],
            self.w["Ar"], self.w["N2"], self.w["CH4"]
        ])
        self.F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])

        self.F_total = self.F.sum()
        self.f = self.F / self.F_total

    def __call__(self, T, P, X):
        return self.func(T, P, X)

    def get_final_f(self, X, F_A0, F_B0, stoich_coef):
        return F_B0 + stoich_coef * X * F_A0

    def Rase(self, T, P, X):
        """
        Reference: Howard F. Rase, "Chemical Reactor Design for Process Plants", Vol 2, 1977.
        
        Rase proposed an empirical model developed based on the industrial data, on page 47. The model is 
        applicable for low/high-temperature water gas shift reactor.
        """
        if type(X) is float or type(X) is int:
            X = np.array([X])
        if type(T) in (float, int):
            T = np.array([T])
            
        T = T / (5 / 9)

        bulk_density = self.bulk_density

        k = np.exp(12.88 - 3340 / T) # for copper-zinc catalyst
        
        K = np.zeros(T.size)
        K[T <= 1060] = np.exp(-4.72 + 8640 / T[T <= 1060])
        K[T > 1060] = np.exp(-4.33 + 8240 / T[T > 1060])
        A = 4.33
        f_final_CO = self.get_final_f(X, self.F[0], self.F[0], -1) / self.F_total
        f_final_H2O = self.get_final_f(X, self.F[0], self.F[1], -1) / self.F_total
        f_final_CO2 = self.get_final_f(X, self.F[0], self.F[2], +1) / self.F_total
        f_final_H2 = self.get_final_f(X, self.F[0], self.F[3], +1) / self.F_total
        
        rate = A * k * (f_final_CO * f_final_H2O - f_final_CO2 * f_final_H2 / K) / (379 * bulk_density)
        
        return rate


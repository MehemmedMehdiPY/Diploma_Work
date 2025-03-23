from integration_approximation import Trapezoidal, Simpson
from models import R_eq_1, LH_1_3, LH_7_3, get_eq_constant, Power_Law, Temkin
import matplotlib.pyplot as plt
import numpy as np

class F:
    def __init__(self, rate_function, params: dict, eps: float = 1e-16):
        self.rate_function = rate_function
        self.params = params
        self.eps = eps

    def __call__(self, X):
        """
        The function used in integral(fx * dx) to compute area with integration approximation methods.

        Returns:
            value of weight at each range of X
        """

        if type(X) is float or type(X) is int:
            X = np.array([X])
        
        rate = self.rate_function(X, self.params)
        # rate[rate == 0] = np.inf
        return (
            - 1 / rate
        )

# kmol / h
# FA0 = 100807.8964

# kmol / hr
# F0_CO = 266 
# F0_CO2 = 1239
# F0_H2 = 4726
# F0_H2O = 2712

# From first semester material balance.
F0_CO = 289.64
F0_CO2 = 1844.69
F0_H2 = 7503.02
F0_H2O = 4082.96

F0_total = (F0_CO + F0_CO2 + F0_H2 + F0_H2O)

# mol / s
F0_CO = F0_CO * 1000 / 3600
F0_CO2 = F0_CO2 * 1000 / 3600
F0_H2 = F0_H2 * 1000 / 3600
F0_H2O = F0_H2O * 1000 / 3600
F0_total = F0_total * 1000 / 3600

# mol / s
FA0 = F0_CO

# T = 501
T = 478
R = 8.3145
eq_constant = get_eq_constant(T)

# kPa
# P_total = 34.6 * 10**5
P_total = 34.6 * 10**5

f_CO = F0_CO / F0_total
f_CO2 = F0_CO2 / F0_total
f_H2O = F0_H2O / F0_total
f_H2 = F0_H2 / F0_total

print(f_CO, f_CO2, f_H2O, f_H2)
print(f_CO + f_CO2 + f_H2O + f_H2)

X_final = 0.90

params = {
    # "R": 8.3145,
    "T": T,
    "P_total": P_total,
    "fractions": {
        "f_CO": f_CO,
        "f_CO2": f_CO2,
        "f_H2O": f_H2O,
        "f_H2": f_H2
    },
    "theta": {
        "theta_CO": F0_CO / F0_CO,
        "theta_CO2": F0_CO2 / F0_CO,
        "theta_H2O": F0_H2O / F0_CO,
        "theta_H2": F0_H2 / F0_CO
    },
    "eq_constant": eq_constant,
}

bulk_density = 1173.1

f = F(rate_function=LH_7_3, params=params)
weight = Simpson(f=f, a=0, b=0.85, n=25000) * FA0
volume = weight / bulk_density
print(weight, volume)

Xs = np.linspace(0, 0.8, num=10000)
plt.plot(Xs, 1 / LH_7_3(Xs, params), color="blue", label="rate")
plt.plot([0, 1], [0, 0], linestyle="--", color="red", label="0 line")
plt.legend()
plt.show()

# plt.plot(Xs, f(Xs), color="blue", label="rate")
# plt.legend()
# plt.show()
import sys
sys.path.append("../")
from models import OutletTemperature, KineticModels

import numpy as np
from models import OutletTemperatureProfile, Enthalpy, KineticModels
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import simpson
from constants import Data
from optimization import Solution

P = 10.0e5
T_init = 455.4792
X_objective = 0.9
n_samples = 1000
ld_ratio = 3.0

solver = Solution()
results = solver(T=T_init, P=P, X_objective=X_objective, ld_ratio=ld_ratio, n_samples=n_samples)

conversion = results["X_profile"]
temperature = results["temperature_profile"]
temperature_change = results["temperature_change_profile"]
rate = results["rate_profile"]
enthalpy = results["enthalpy_profile"]

enthalpy_change = enthalpy.copy()
enthalpy_change[1:] = (
    enthalpy[1:] - enthalpy[:-1]
)

data = np.zeros((n_samples, 6))

data[:, 0] = conversion
data[:, 1] = temperature
data[:, 2] = temperature_change
data[:, 3] = enthalpy
data[:, 4] = enthalpy_change
data[:, 5] = rate

df = pd.DataFrame(data, columns=["conversion_profile", "temperature_profile", "instant_temperature_change_profile", 
                                 "enthalpy_profile", "instant_enthalpy_profile", "reaction_rate_profile"])

df.to_csv("profiles.csv", index=False)

print(df)
import sys
sys.path.append("../")
from constants import Data
import numpy as np
from models import OutletTemperatureProfile, KineticModels
import matplotlib.pyplot as plt

# X = np.linspace(0, 0.9, 100)
self = Data

F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])
stoichs = np.array([-1, -1, +1, +1, 0, 0, 0])

T = 469.902
P = 10.0e5
ld_ratio = 5.0

model = OutletTemperatureProfile()
Xs, Ts, Hs, dTs = model(T, P, X_objective=0.9)


plt.plot(Xs, Ts)
plt.show()

plt.plot(Xs, Hs.cumsum())
plt.show()
print(Ts[-1])

X = Xs

print(X[49], X[50])
print(F.tolist())

F_new = F + stoichs * F[0] * X[49]
print(F_new.tolist())

F_new = F + stoichs * F[0] * X[50]
print(F_new.tolist())

print(Ts[49], Ts[50])
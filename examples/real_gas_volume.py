import sys
sys.path.append("../")
from constants import Data
import numpy as np
from models.models import PengRobinson
import matplotlib.pyplot as plt

# X = np.linspace(0, 0.9, 100)
self = Data

F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])
stoichs = np.array([-1, -1, +1, +1, 0, 0, 0])


P = 10.0e5
ld_ratio = 3.0

model = PengRobinson()
# indexes=[0, 2, 3, 4, 5, 6]
T = 480.83
# T = 455.53
indexes = None

volume = model(T=T, P=P, v=0.001, optimize=True, indexes=indexes)
print(volume)


self = Data
self.F = np.array([
    self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
    self.F["Ar"], self.F["N2"], self.F["CH4"]
])
Mw = np.array([
            self.Mw["CO"], self.Mw["H2O"], self.Mw["CO2"], self.Mw["H2"],
            self.Mw["Ar"], self.Mw["N2"], self.Mw["CH4"]
        ])

self.F_total = self.F.sum()

self.f = self.F / self.F_total

# g/mol or kg/kmol
self.Mw_mixture = (self.f * Mw).sum()

print(self.Mw_mixture / volume / 1000)
print(self.F_total)

print(self.F)
print(self.F.sum())
self.F = self.F + self.F[0] * np.array([-1, -1, 1, 1, 0, 0, 0])
print(self.F)
print(self.F.sum())
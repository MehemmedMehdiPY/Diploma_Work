import sys
sys.path.append("../")
from constants import Data
import numpy as np
from models import PengRobinson
import matplotlib.pyplot as plt

# X = np.linspace(0, 0.9, 100)
self = Data

F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])
stoichs = np.array([-1, -1, +1, +1, 0, 0, 0])

T = 470.831
P = 10.0e5
ld_ratio = 5.0

model = PengRobinson()
volume = model(T=T, P=P, v=0.001, optimize=True, indexes=[0, 2, 3, 4, 5, 6])
print(volume)

print(model(T, P, indexes=[0]))
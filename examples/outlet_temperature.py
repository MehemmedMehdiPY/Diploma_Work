import sys
sys.path.append("../")
from models import OutletTemperatureProfile
import matplotlib.pyplot as plt

# Optimized initial temperature
T_init = 455.4792

# Optimized pressure
P = 10.0e5

model = OutletTemperatureProfile()

conversion_profile, temperature_profile, enthalpy_profile, temperature_change_profile = model(T=T_init, P=P, X_objective=0.9, n_samples=100)

print("Final temperature: {} K".format(temperature_profile[-1]))

plt.plot(conversion_profile, temperature_profile)
plt.show()
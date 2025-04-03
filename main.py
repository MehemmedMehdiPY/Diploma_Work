from scipy.integrate import simpson

class Solution(Data):
    def __init__(self):
        super().__init__()
        self.flow_rate = self.F["CO"] * 1000 / 453.592

        """The model to provide temperature/enthalpy/instantaneous temperature change profile versus conversion across reactor"""
        self.model_temperature = OutletTemperatureProfile()

        """The model to provide pressure profile across reactor
        (will be removed in the future)
        """
        self.model_pressure_drop = PressureDrop()

        """The model to provide reaction rate profile across reactor"""
        self.kinetic_model = KineticModels(model="rase")
    
    def __call__(self, T, P, X_objective, ld_ratio = 0.67):
        designer = MechanicalDesign()

        X_profile, T_profile, enthalpy_profile, dTs = self.model_temperature(T=T, P=P, X_objective=X_objective, n_samples=100)
        
        rate_profile = self.kinetic_model(T=T_profile, P=P, X=X_profile)
        rate_profile[rate_profile < 1e-6] = 1e-7
        volume, weight = self.get_volume_weight(X_profile, rate_profile)
        diameter = (volume * 4 / np.pi / ld_ratio) ** (1 / 3)
        height = diameter * ld_ratio
        

        mechanical_design_results = designer(T=T, P=P, volume=volume, ld_ratio=ld_ratio)
        
        # A_cross = np.pi / 4 * diameter ** 2
        # pressure_drop_profile = self.model_pressure_drop(T0=T, P0=P, T=T_profile, weight=weight, A_cross=A_cross)
        
        results = {
            "volume": volume,
            "weight": weight,
            "diameter": diameter,
            "height": height,
            "X_profile": X_profile,
            "temperature_profile": T_profile,
            "temperature_change_profile": dTs,
            "rate_profile": rate_profile,
            "enthalpy_profile": enthalpy_profile.cumsum(),
            # "pressure_drop_profile": pressure_drop_profile,
            "pressure": P,
            "initial_temperature": T,
            "final_temperature": T_profile[-1],
            "final_enthalpy": enthalpy_profile[-1],
            "final_rate": rate_profile[-1],
            # "pressure_drop": pressure_drop_profile[-1],
            "mechanical_design": mechanical_design_results
        }
        return results
    
    def get_volume_weight(self, X, rates):
        # First, we convert flow rate to lb-mol since kinetic model expects units like this
        flow_rate = self.F["CO"] * 1000 / 453.592

        # Flow rate profile across reactor with respect to conversion
        flow_rates = np.zeros(X.size)
        flow_rates = flow_rate - 1 * flow_rate * X

        # This function should be integrated to compute weight of catalyst
        f = 1 / rates * flow_rates
        weight = simpson(y=f, x=X)

        # And volume of reactor. self.bulk_density comes from Data class
        volume = weight / self.bulk_density
        volume = volume / 3.28084 ** 3
        weight = weight / 2.20462
        return volume, weight


from models import OutletTemperatureProfile, KineticModels, Data
# from models_try import OutletTemperatureProfile, KineticModels, Data
import matplotlib.pyplot as plt
import numpy as np

T = 478
P = 32.4e5
X_objective = 0.9

model = OutletTemperatureProfile()
X, T_profile, enthalpy_profile, dTs = model(T=T, P=P, X_objective=X_objective, n_samples=100)
model = KineticModels()
rates = model(T=T_profile, P=P, X=X_objective)

plt.plot(X, T_profile)
plt.show()

plt.plot(X, rates)
plt.show()
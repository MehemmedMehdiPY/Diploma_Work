import numpy as np
from models import PressureDrop, KineticModels, OutletTemperatureProfile
from scipy.integrate import simpson
import sys
sys.path.append("../")
from constants.constants import Data

class MechanicalDesign:
    """The method to provide design parameters, such as weight, volume, diameters, wall thicknesses, ellipsoidal heads, etc."""
    def __init__(self):
        self.stress_max = 18 * 68.9476 * 10 ** 5
        self.joint_efficiency = 1
        self.wall_thickness_min = 0.002
        self.Cv = 1.08
        self.density = 7850

    def __call__(self, T, P, volume, ld_ratio = 0.67):
        self.ld_ratio = ld_ratio

        P_design = P *1.1
        # T_design = T + 10

        diameter_internal, height = self.get_internal_dimensions(volume)
        
        wall_thickness_vessel = self.get_wall_thickness_vessel(P_design, diameter_internal)
        wall_thickness_head = self.get_wall_thickness_head(P_design, diameter_internal)
        hemispherical_head = self.get_hemispherical_head(P_design, diameter_internal)

        # SF
        # straight_flange = 2.0 * hemispherical_head

        # Internal Dish Diameter (IDD)
        dish_depth = diameter_internal / 2
        # dish_height_total = straight_flange + dish_depth + wall_thickness_head

        # Reactor
        diameter_mean = diameter_internal + wall_thickness_vessel

        volume_vessel, weight_vessel = self.get_vessel(diameter_mean, height, wall_thickness_vessel)
        volume_head, weight_head = self.get_head(hemispherical_head, dish_depth)

        # Total
        weight_total = weight_vessel + 2 * weight_head
        volume_total = volume_vessel + volume_head

        results = {
            "volume_vessel": volume_vessel,
            "volume_head": volume_head,
            "volume_total": volume_total,
            "weight_vessel": weight_vessel,
            "weight_head": weight_head,
            "weight_total": weight_total,
            "wall_thickness_vessel": wall_thickness_vessel,
            "wall_thickness_head": wall_thickness_head,
            "hemispherical_head": hemispherical_head,
        }
        return results
    
    def get_internal_dimensions(self, volume):
        diameter_internal = (volume / (np.pi * self.ld_ratio / 4)) ** (1 / 3)
        height = diameter_internal * self.ld_ratio
        return diameter_internal, height
    
    def get_wall_thickness_vessel(self, P_design, diameter_internal):
        wall_thickness_vessel = P_design * diameter_internal / (4 * self.stress_max * self.joint_efficiency + 0.8 * P_design)
        wall_thickness_vessel = wall_thickness_vessel + self.wall_thickness_min
        return wall_thickness_vessel

    def get_wall_thickness_head(self, P_design, diameter_internal):
        wall_thickness_head = P_design * diameter_internal / (4 * self.stress_max * self.joint_efficiency - 0.4 * P_design)
        wall_thickness_head = wall_thickness_head + self.wall_thickness_min
        return wall_thickness_head

    def get_hemispherical_head(self, P_design, diameter_internal):
        hemispherical_head = P_design * diameter_internal / (4 * self.stress_max * self.joint_efficiency - 0.4 * P_design)
        return hemispherical_head
    
    def get_vessel(self, diameter_mean, length, wall_thickness_vessel):
        volume_vessel = np.pi * diameter_mean ** 2 / 4 * length
        weight_vessel = self.density * self.Cv * np.pi * diameter_mean * (length + 0.8 * diameter_mean) * wall_thickness_vessel
        return volume_vessel, weight_vessel
    
    def get_head(self, hemispherical_head, dish_depth):
        volume_head = (np.pi * hemispherical_head * dish_depth ** 2)
        weight_head = self.density * volume_head * 2
        return volume_head, weight_head
    
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
    
    def __call__(self, T, P, X_objective, ld_ratio = 0.67, n_samples = 100):
        designer = MechanicalDesign()

        X_profile, T_profile, enthalpy_profile, dTs = self.model_temperature(T=T, P=P, X_objective=X_objective, n_samples=n_samples)
        
        rate_profile = self.kinetic_model(T=T_profile, P=P, X=X_profile)
        rate_profile[rate_profile < 1e-6] = 1e-6
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

if __name__ == "__main__":
    solver = Solution()
    results = solver(T=455.4792, P=10e5, X_objective=0.9, ld_ratio=3, n_samples=100)
    print(results.keys())
    print(results["weight"])
    print(results["volume"])
    print(results["final_enthalpy"])
    print(results["final_temperature"])
    print(results["final_rate"])
import numpy as np
import pandas as pd
from models import OutletTemperatureProfile, KineticModels, PressureDrop
from constants import Data
from scipy.integrate import simpson
from scipy.optimize import minimize, Bounds
import numdifftools as nd

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
    
class Cost(Data):
    def __init__(self, X_objective):
        self.X_objective = X_objective
        self.solver = Solution()
        self.catalyst_unit_price = 20
        self.callback = []

    def __call__(self, args):
        return self.cost(args)
    
    def cost(self, args):
        T, P, ld_ratio = args
        
        results = self.solver(T=T, P=P, X_objective=0.9, ld_ratio=ld_ratio)
        volume = results["volume"]
        weight = results["weight"]
        weight_equipment = results["mechanical_design"]["weight_total"]

        cost_equipment = self.equipment_cost(weight_equipment=weight_equipment)
        cost_catalyst = self.catalyst_cost(weight)
        cost_total = cost_equipment + cost_catalyst
        results = {
            "equipment_cost": cost_equipment,
            "catalyst_cost": cost_catalyst,
            "total_cost": cost_total,
            "solution": results
        }
        return results

    def equipment_cost(self, weight_equipment):
        a = 17400
        b = 79
        n = 0.85

        escalation_ratio = 1.6
        f_er = 0.3
        f_p = 0.8
        f_i = 0.3
        f_el = 0.2
        f_c = 0.3
        f_s = 0.2
        f_l = 0.1
        f_m = 1
        f_total = f_m * (1 + f_p) + (f_er + f_i + f_el + f_c + f_s + f_l) 
        cost = a + b * weight_equipment ** n
        cost = f_total * cost
        cost = cost * escalation_ratio
        return cost

    def catalyst_cost(self, catalyst_weight):
        return catalyst_weight * self.catalyst_unit_price

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
    
    def update_callback(self, *args):
        self.callback.append(args)

    def save(self, *args):
        pd.DataFrame(self.callback,
        columns=args
        ).to_csv("./callback/callback.csv", index=False)

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

X_OBJECTIVE = 0.9
class Optimization:
    def __init__(self):
        self.cost_model = Cost(X_objective=X_OBJECTIVE)
        self.F = np.array([
            Data.F["CO"], Data.F["H2O"], Data.F["CO2"], Data.F["H2"],
            Data.F["Ar"], Data.F["N2"], Data.F["CH4"]
        ])
        self.Mw = np.array([
            Data.Mw["CO"], Data.Mw["H2O"], Data.Mw["CO2"], Data.Mw["H2"],
            Data.Mw["Ar"], Data.Mw["N2"], Data.Mw["CH4"]
        ])

        # Initiailizing jacobian matrix solvers
        self.obj_func_jac_ = nd.Gradient(self.obj_func)
        self.rate_constrain_jac_ = nd.Gradient(self.rate_constrain)

    def __call__(self, T, P, ld_ratio):
        T, P, ld_ratio = self.scale(T, P, ld_ratio)
        print(T, P, ld_ratio)
        x0 = np.array([T, P, ld_ratio])

        lb = (0.0, 0.0, 0.0)
        ub = (1.0, 1.0, 1.0)
        bounds = Bounds(lb=lb, ub=ub)
        
        cons = {
            "type": "ineq",
            "fun": self.rate_constrain,
            "jac": self.rate_constrain_jac
        }        
        
        print("Optimization starts")
        results = minimize(
            fun=self.obj_func,
            x0=x0,
            jac=self.obj_func_jac,
            method="SLSQP",
            options={"disp": False},
            bounds=bounds,
            constraints=cons
        )
        print("Optimization ended")

        # Saving dataframe of callbacks.
        self.cost_model.save(
            "Pressure", "Initial Temperature", "Length-to-diameter Ratio", "Final Temperature", "Final Enthalpy", "Reaction Rate",
            #"Pressure Drop", 
            "Catalyst Weight","Equipment Volume", "Equipment Weight", "Equipment Cost", "Catalyst Cost",
            "Total Cost", "Objective Function"
                    )
        
        return results

    def scale(self, T, P, ld_ratio):
        T_min = 298.0
        T_max = 500.0
        P_min = 10.0e5
        P_max = 40.0e5
        ld_ratio_min = 0.4
        ld_ratio_max = 3.0
        # F_N2_min = 0.001
        # F_N2_max = 3000.0
        # F_CH4_min = 0.001
        # F_CH4_max = 3000.0

        T = (T - T_min) / (T_max - T_min)
        P = (P - P_min) / (P_max - P_min)
        ld_ratio = (ld_ratio - ld_ratio_min) / (ld_ratio_max - ld_ratio_min)
        
        return (T, P, ld_ratio)
    
    def unscale(self, T, P, ld_ratio):
        T_min = 298.0
        T_max = 500
        P_min = 10.0e5
        P_max = 40.0e5
        ld_ratio_min = 0.4
        ld_ratio_max = 3.0
        # F_N2_min = 0.001
        # F_N2_max = 3000.0
        # F_CH4_min = 0.001
        # F_CH4_max = 3000.0

        T = T * (T_max - T_min) + T_min
        P = P * (P_max - P_min) + P_min
        ld_ratio = ld_ratio * (ld_ratio_max - ld_ratio_min) + ld_ratio_min
        
        return (T, P, ld_ratio)
    
    def reset(self, F_N2, F_CH4):
        """F is the array of flowrates of 7 components, however, we only manipulate the last two variables 
            to see their influence on the results. Those variables correspond to flowrates of N2 and CH4.
        """
        F = self.F

        F[-2] = F_N2
        F[-1] = F_CH4
        
        F_total = F.sum()
        f = F / F_total

        # The below will be removed in the future. Currently, does not have influence on the results.
        self.cost_model.solver.model_pressure_drop.F = F
        self.cost_model.solver.model_pressure_drop.F_total = F_total
        self.cost_model.solver.model_pressure_drop.f = f
        self.cost_model.solver.model_pressure_drop.Mw_mixture = (f * self.Mw).sum()
        self.cost_model.solver.model_pressure_drop.PR_model.F = F
         
        # Updating flow rates
        self.cost_model.solver.model_temperature.F_start = F
        self.cost_model.solver.model_temperature.F_init = F

        self.cost_model.solver.kinetic_model.F = F
        self.cost_model.solver.kinetic_model.F_total = F_total
        self.cost_model.solver.kinetic_model.f = f

    def obj_func(self, x):
        # Scaling parameters (experimentally selected)
        division_scale = 17000939 / 2
        w1 = 0.64
        w2 = 0.36

        # Scaled input parameters
        T, P, ld_ratio = x
        print(T, P, ld_ratio)
        T, P, ld_ratio = self.unscale(T, P, ld_ratio)
        
        # Setting up the updated flow rates.
        # self.reset(F_N2, F_CH4)

        results = self.cost_model([T, P, ld_ratio])

        # Balancing cost contributions from equipment and catalyst with scaling
        cost = (results["equipment_cost"] * w1 + results["catalyst_cost"] * w2) / division_scale
        
        # Saving history
        self.cost_model.update_callback(
            P,
            results["solution"]["initial_temperature"],
            ld_ratio,
            results["solution"]["final_temperature"],
            results["solution"]["final_enthalpy"],
            results["solution"]["final_rate"],
            # results["solution"]["pressure_drop"],
            results["solution"]["weight"],
            results["solution"]["mechanical_design"]["volume_total"],
            results["solution"]["mechanical_design"]["weight_total"],
            results["equipment_cost"],
            results["catalyst_cost"],
            results["total_cost"],
            cost
        )
        return cost
    
    def obj_func_jac(self, x):
        jac_mat = self.obj_func_jac_(x)
        print("obj jac:", jac_mat)
        return jac_mat
    
    def rate_constrain(self, x):
        """Constrain function for reaction rate. Instended to achieve at least 10^-6 of rate at X=0.9
        If final (outlet) rate is lower than 0, it indicates that reaction finished earlier.
        """
        T, P, ld_ratio = x
        T, P, ld_ratio = self.unscale(T, P, ld_ratio)

        # Setting up the updated flow rates.
        # self.reset(F_N2, F_CH4)
        
        results = self.cost_model([T, P, ld_ratio])
        final_rate = results["solution"]["final_rate"]
        constrain = final_rate - 1e-6
        print("rate constrain:", constrain)
        return constrain

    def rate_constrain_jac(self, x):
        """The jacobian matrix of constrain function"""
        jac_mat = self.rate_constrain_jac_(x)
        print("rate jac:", jac_mat)
        return jac_mat
        

if __name__ == "__main__":
    # Optimization results:
    T = 455.4792
    P = 10.0e5
    ld_ratio = 3
    # T = 300
    # P = 20.0 * 10 ** 5
    # ld_ratio = 1.01
    # print(T, P, ld_ratio)

    # optimizer = Optimization()
    # results = optimizer(T, P, ld_ratio)
    # print(results)


    # import matplotlib.pyplot as plt

    # T = 470.8312
    # P = 28.2 * 10 ** 5

    solver = Solution()
    results = solver(T=T, P=P, X_objective=0.9, ld_ratio=3.0)

    # print(results["volume"], results["final_rate"], results["final_temperature"])

    # plt.plot(results["X_profile"], results["temperature_profile"])
    # plt.show()

    # plt.plot(results["X_profile"], results["enthalpy_profile"])
    # plt.show()

    # plt.plot(results["X_profile"], results["rate_profile"])
    # plt.show()
    print(results["mechanical_design"]["weight_vessel"])
    print(results["mechanical_design"]["weight_head"])
    print(results["mechanical_design"]["weight_total"])
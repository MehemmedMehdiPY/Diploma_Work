import numpy as np
import sys
from models import SaturationTemperature, PengRobinson
sys.path.append("../")
from constants.constants import Data

class EnthalpyTemperature(Data):
    def __init__(self):
        self.cp_constants_gas_arr = np.array(
            [self.cp_constants_gas["CO"],
            self.cp_constants_gas["H2O"],
            self.cp_constants_gas["CO2"],
            self.cp_constants_gas["H2"]]
        ).reshape(4, 5)

        self.cp_constants_liq_arr = np.array(
            self.cp_constants_liq["H2O"]
        ).reshape(1, 5)

        self.saturation_estimator = SaturationTemperature()
        self.vap_enthalpy = VaporizationEnthalpy()

    def __call__(self, T, indexes = None, is_H2O = False):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return self.get_enthalpy_temperature(T, is_H2O, indexes=indexes)
    
    def get_abcde_m(self, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        constants = self.cp_constants_gas[indexes]
        constants_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        for i in range(len(indexes)):
            constants_mat[i] = constants * constants[i]
            f_mat[i] = f * f[i]

        constants_m = np.sum(
            np.sqrt(constants_mat) * f_mat
            )
        return constants_m


    def get_enthalpy_temperature(self, T, is_H2O = False, indexes = None):
        if indexes is None and not is_H2O:
            indexes = np.arange(self.F.size)
        
        T_ref = 298
        if not is_H2O:
            enthalpy_temperature_corrected = (
                self.cp_constants_gas_arr[indexes, 0] / 1 * (T ** 1 - T_ref ** 1) + 
                self.cp_constants_gas_arr[indexes, 1] / 2 * (T ** 2 - T_ref ** 2) + 
                self.cp_constants_gas_arr[indexes, 2] / 3 * (T ** 3 - T_ref ** 3) + 
                self.cp_constants_gas_arr[indexes, 3] / 4 * (T ** 4 - T_ref ** 4) + 
                self.cp_constants_gas_arr[indexes, 4] / 5 * (T ** 5 - T_ref ** 5)
            )
        else:
            T_sat = 100 + 273.15
            
            enthalpy_liquid = (
                self.cp_constants_liq_arr[0, 0] / 1 * (T_sat ** 1 - T_ref ** 1) + 
                self.cp_constants_liq_arr[0, 1] / 2 * (T_sat ** 2 - T_ref ** 2) + 
                self.cp_constants_liq_arr[0, 2] / 3 * (T_sat ** 3 - T_ref ** 3) + 
                self.cp_constants_liq_arr[0, 3] / 4 * (T_sat ** 4 - T_ref ** 4)
            )

            enthalpy_vaporization = self.vap_enthalpy()

            enthalpy_gas = (
                self.cp_constants_gas_arr[1, 0] / 1 * (T ** 1 - T_sat ** 1) + 
                self.cp_constants_gas_arr[1, 1] / 2 * (T ** 2 - T_sat ** 2) + 
                self.cp_constants_gas_arr[1, 2] / 3 * (T ** 3 - T_sat ** 3) + 
                self.cp_constants_gas_arr[1, 3] / 4 * (T ** 4 - T_sat ** 4) + 
                self.cp_constants_gas_arr[1, 4] / 5 * (T ** 5 - T_sat ** 5)
            )

            enthalpy_temperature_corrected = (
                enthalpy_liquid + enthalpy_vaporization + enthalpy_gas
                )
        return enthalpy_temperature_corrected
    
    def get_capacity_temperature(self, T, P, is_H2O = False, indexes = None):
        if indexes is None and not is_H2O:
            indexes = np.arange(self.F.size)
        if not is_H2O:
            heat_capacity = (
                    self.cp_constants_gas_arr[indexes, 0] * T ** 0 + 
                    self.cp_constants_gas_arr[indexes, 1] * T ** 1 + 
                    self.cp_constants_gas_arr[indexes, 2] * T ** 2 + 
                    self.cp_constants_gas_arr[indexes, 3] * T ** 3 + 
                    self.cp_constants_gas_arr[indexes, 4] * T ** 4
                )
        else:
            T_sat = self.saturation_estimator(P=P, optimize=True)
            if T < T_sat:
                heat_capacity = (
                        self.cp_constants_liq_arr[0, 0] * T ** 0 + 
                        self.cp_constants_liq_arr[0, 1] * T ** 1 + 
                        self.cp_constants_liq_arr[0, 2] * T ** 2 + 
                        self.cp_constants_liq_arr[0, 3] * T ** 3
                    )
            else:
                heat_capacity = (
                    self.cp_constants_gas_arr[1, 0] * T ** 0 + 
                    self.cp_constants_gas_arr[1, 1] * T ** 1 + 
                    self.cp_constants_gas_arr[1, 2] * T ** 2 + 
                    self.cp_constants_gas_arr[1, 3] * T ** 3 + 
                    self.cp_constants_gas_arr[1, 4] * T ** 4
                )
        return heat_capacity
    
class EnthalpyPressure(PengRobinson):
    def __init__(self):
        super().__init__()
        self.R = 8.3145

    def __call__(self, T, P, indexes = None, is_H2O=False):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return self.get_enthalpy_pressure(T, P, indexes=indexes, is_H2O=is_H2O)
    
    def get_enthalpy_pressure(self, T, P, is_H2O = True, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)

        if is_H2O:
            """Water is negligbly compressible and pressure influence is insignificant"""
            return 0
        
        R = self.R
        a = self.get_am(T, indexes=indexes)
        b = self.get_bm(indexes=indexes)
        
        v_real = self.approximate_v(T, P, a, b)
        Z = P * v_real / (R * T)
        B = b * P / (R * T)
        da_dT = self.get_da_dT(T, a)
        enthalpy_pressure_corrected = (
            R * T * (Z - 1) + (T * da_dT - a) / (2 * 2 ** 0.5 * b) * np.log(
                (Z + (1 + 2 ** 0.5) * B) / (Z + (1 - 2 ** 0.5) * B)
                )
            )
        return enthalpy_pressure_corrected
    
    def get_capacity_pressure(self, T, P, v, a, b, is_H2O=False):
        if not is_H2O:
            # Partial differentiation of P with respect to T at v=const
            dP_dT = self.get_dP_dT(T, v, a, b)
            
            # Partial differentiation of P with respect to v at T=const
            dP_dv = self.get_dP_dv(T, v, a, b)
    
            # Second order partial differentiation of a with respect to T
            d2a_dT2 = self.get_d2a_dT2(T, a)
            
            K = 1 / (2 * 2 ** 0.5 * b) * np.log(
                (v + (1 - 2 ** 0.5) * b) / (v + (1 + 2 ** 0.5) * b)
                )        
            heat_capacity = - T * dP_dT / dP_dv - self.R - T * d2a_dT2 * K
        else:
            heat_capacity = 0
        return heat_capacity

class VaporizationEnthalpy(Data):
    def __init__(self):
        """The method to compute enthalpy of vaporization only for water"""
        self.vaporization_constants = self.vaporization_constants["H2O"]
        self.Tc_H2O = self.Tc["H2O"]

    def __call__(self):
        return self.get_vaporization_enthalpy()
    
    def get_vaporization_enthalpy(self):
        T = 100 + 273.15
        return (
            # J/mol
            self.vaporization_constants[0] * (1 - T / self.vaporization_constants[1]) ** self.vaporization_constants[2] 
            * 1000
            )

class Enthalpy(EnthalpyTemperature, EnthalpyPressure):
    def __init__(self):
        super().__init__()
        EnthalpyPressure.__init__(self)

    def __call__(self, T, P, is_H2O = False, indexes = None, method = "enthalpy"):
        if indexes is None:
            indexes = np.arange(self.F.size)

        if method.lower() == "enthalpy":    
            return self.get_total_enthalpy(T, P, is_H2O, indexes=indexes)
        elif method.lower() == "capacity":
            return self.heat_capacity(T, P, indexes=indexes)

    def get_total_enthalpy(self, T, P, is_H2O, indexes=None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        enthalpy_temperature_corrected = self.get_enthalpy_temperature(T, is_H2O=is_H2O, indexes=indexes)
        enthalpy_pressure_corrected = self.get_enthalpy_pressure(T, P, is_H2O=is_H2O, indexes=indexes)
        return enthalpy_temperature_corrected.sum() + enthalpy_pressure_corrected

    def heat_capacity(self, T, P, is_H2O = False, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        a = self.get_am(T, indexes=indexes)
        b = self.get_bm(indexes)
        # v_init = 0.0001

        # Peng-Robinson method to compute real gas molar volume
        v = self.approximate_v(T, P, a, b)

        cp_temperature = self.get_capacity_temperature(T, P=P, is_H2O=is_H2O, indexes=indexes)
        cp_pressure = self.get_capacity_pressure(T, P, v, a, b, is_H2O=is_H2O)

        return cp_temperature + cp_pressure

class EnthalpyReaction(Enthalpy):
    def __init__(self):
        super().__init__()
        self.H_heat = -41100
        self.F_init = self.F.copy()
        self.stoich_coefs = np.array([-1, -1, +1, +1, 0, 0, 0])
        self.A0_idx = 0

    def __call__(self, T, P, X):
        return self.get_heat_enthalpy(T, P, X)
    
    def get_theta(self, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return self.F[indexes] / self.F[self.A0_idx]

    def get_theta_m(self, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        theta = self.get_theta(indexes=indexes)
        theta_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        for i in range(len(indexes)):
            theta_mat[i] = theta * theta[i]
            f_mat[i] = f * f[i]

        theta_m = np.sum(
            np.sqrt(theta_mat) * f_mat
            )
        
        return theta_m
    
    def get_heat_enthalpy(self, T, P, X):
        enthalpy_react = 0
        enthalpy_prod = 0

        is_H2O = False
        indexes = np.array([0, 2, 3])
        enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600
        
        is_H2O = True
        indexes = np.array([1])
        enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600
        
        self.F = self.F_init + self.stoich_coefs * X * self.F_init[self.A0_idx]
        
        is_H2O = False
        indexes = np.array([0, 2, 3])
        enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600

        is_H2O = True
        indexes = np.array([1])
        enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600

        enthalpy = enthalpy_prod - enthalpy_react + self.H_heat * X * self.F_init[self.A0_idx] * 1000 / 3600
        self.F = self.F_init.copy()
        return enthalpy
    
class OutletTemperature(EnthalpyReaction):
    """The class to compute outlet temperature from heat of reaction"""
    def __init__(self):
        super().__init__()
    
    def __call__(self, T, P, X):
        return self.get_T_final(T, P, X)

    def get_T_final(self, T, P, X):
        """
        heat_prev:          Substract previous heat.
        T_init:             Deprecated
        """
       
        cp_theta = 0
        self.F = self.F_init + self.stoich_coefs * X * self.F_init[self.A0_idx]
        for i in range(7):
            is_H2O = False
            if i == 1:
                is_H2O = True
            indexes = np.array([i])
            # theta = self.get_theta(indexes=indexes)
            theta = self.F[indexes] * 1000 / 3600
            cp_i = self.heat_capacity(T, P, is_H2O=is_H2O, indexes=indexes)
            cp_theta += cp_i * theta

        self.F = self.F_init.copy()
        heat_enthalpy = self.get_heat_enthalpy(T, P, X)

        dT = -heat_enthalpy / cp_theta # / 25
        T_final = T + dT
        return T_final[0], heat_enthalpy, dT

class OutletTemperatureProfile(OutletTemperature):
    """
    Method to determine temperature profile over conversion.
    """
    def __init__(self):
        super().__init__()
        self.F_start = self.F.copy()
    
    def __call__(self, T, P, X_objective, n_samples=100):
        if X_objective >= 1 or X_objective <= 0:
            raise ValueError("X_objective should be between 0 and 1".format(X_objective))
        T_init = T

        X = np.linspace(0.0, X_objective, n_samples)
        X_change = X[1] - X[0]
        T_profile = np.zeros(n_samples)
        enthalpy_profile = np.zeros(n_samples)
        dTs = np.zeros(n_samples)
        
        for i, X_value in enumerate(X):
            T, heat_enthalpy, dT = self.get_T_final(T=T, P=P, X=X_change)
            # Pushing initial flow rates to the next case of new X value.
            self.F_init = self.F_start + self.stoich_coefs * X_value * self.F_start[self.A0_idx]
            T_profile[i] = T
            enthalpy_profile[i] = heat_enthalpy
            dTs[i] = dT

        return X, T_profile, enthalpy_profile, dTs

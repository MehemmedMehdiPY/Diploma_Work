# import numpy as np
import numpy as np
from scipy.optimize import root
from constants import Data

def inverse_ReLU(x):
    return (x if x < 0 else 0)

class PengRobinson(Data):
    def __init__(self):
        self.Tc = np.array([
            self.Tc["CO"], self.Tc["H2O"], self.Tc["CO2"], self.Tc["H2"],
            self.Tc["Ar"], self.Tc["N2"], self.Tc["CH4"]
        ])
        self.Pc = np.array([
            self.Pc["CO"], self.Pc["H2O"], self.Pc["CO2"], self.Pc["H2"],
            self.Pc["Ar"], self.Pc["N2"], self.Pc["CH4"]
        ])
        self.w = np.array([
            self.w["CO"], self.w["H2O"], self.w["CO2"], self.w["H2"],
            self.w["Ar"], self.w["N2"], self.w["CH4"]
        ])
        self.F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])

        self.A0_idx = 0

        self.cp_constants_gas_arr = np.array(
            [self.cp_constants_gas["CO"],
            self.cp_constants_gas["H2O"],
            self.cp_constants_gas["CO2"],
            self.cp_constants_gas["H2"],
            self.cp_constants_gas["Ar"],
            self.cp_constants_gas["N2"],
            self.cp_constants_gas["CH4"]]
        ).reshape(7, 5)

        self.cp_constants_liq_arr = np.array(
            self.cp_constants_liq["H2O"]
        ).reshape(1, 5)

        self.R = 8.3145
        self.func = lambda v, T, a, b, R: R * T / (v - b) - a / (v * (v + b) + b * (v - b))
        self.func_loss = lambda v, P, T, a, b, R: (R * T / (v - b) - a / (v * (v + b) + b * (v - b)) - P)[0]

    def __call__(self, T, P, v = 0.001, optimize = True, indexes = None):
        return self.run(T, P, v, optimize=optimize, indexes=indexes)
        
    def run(self, T, P, v = 0.0001, optimize = True, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)

        a = self.get_am(T=T, indexes=indexes)
        b = self.get_bm(indexes=indexes)

        if optimize:
            v = self.approximate_v(T, P, a, b)
            return v
        else:
            return self.func(v, T, a, b, self.R)
    
    def approximate_v(self, T, P, a, b, v=0.0001):
        f = self.func_loss
        args = (P, T, a, b, self.R)
        results = root(f, x0=v, args=args)
        return results.x

    def get_a(self, T, indexes):
        a_Tc = 0.45724 * self.R ** 2 * self.Tc[indexes] ** 2 / self.Pc[indexes]
        k = self.get_k(indexes)
        Tr = T / self.Tc[indexes]
        a_Tr_w = (1 + k * (1 - np.sqrt(Tr))) ** 2
        a = a_Tc * a_Tr_w
        return a

    def get_am(self, T, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        a = self.get_a(T, indexes=indexes)
        a_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        
        for i in range(len(indexes)):
            a_mat[i] = a * a[i]
            f_mat[i] = f * f[i]

        am = np.sum(
            np.sqrt(a_mat) * f_mat
            )
        return am

    def get_Tc_m(self, indexes = None):        
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        Tc = self.Tc[indexes]
        Tc_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        for i in range(len(indexes)):
            Tc_mat[i] = Tc * Tc[i]
            f_mat[i] = f * f[i]

        Tc_m = np.sum(
            np.sqrt(Tc_mat) * f_mat
            )
        return Tc_m

    def get_Pc_m(self, indexes = None):        
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        Pc = self.Pc[indexes]
        Pc_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        
        for i in range(len(indexes)):
            Pc_mat[i] = Pc * Pc[i]
            f_mat[i] = f * f[i]

        Pc_m = np.sum(
            np.sqrt(Pc_mat) * f_mat
            )
        return Pc_m

    def get_b(self, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return (0.07780 * self.R * self.Tc[indexes] / self.Pc[indexes])

    def get_bm(self, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        f = self.F[indexes] / self.F[indexes].sum()
        return np.sum(f * self.get_b(indexes))

    def get_k(self, indexes):
        k = 0.37464 + 1.54226 * self.w[indexes] - 0.26992 * self.w[indexes] ** 2
        return k
    
    def get_k_m(self, indexes = None):
        """k of mixture"""
        if indexes is None:
            indexes = np.arange(self.F.size)
        
        k = self.get_k(indexes=indexes)
        k_mat = np.zeros((len(indexes), len(indexes)))
        
        f_mat = np.zeros((len(indexes), len(indexes)))
        f = self.F[indexes] / self.F[indexes].sum()
        
        for i in range(len(indexes)):
            k_mat[i] = k * k[i]
            f_mat[i] = f * f[i]

        k_m = np.sum(
            np.sqrt(k_mat) * f_mat
            )
        return k_m

    def get_da_dT(self, T, a, indexes = None):
        """First order partial differentiation"""
        if indexes is None:
            indexes = np.arange(self.F.size)
        k = self.get_k_m(indexes=indexes)

        # Critical temperature of mixture
        Tc_m = self.get_Tc_m(indexes=indexes)

        # Critical pressure of mixture
        Pc_m = self.get_Pc_m(indexes=indexes)

        return (
            -0.45724 * (self.R ** 2 * Tc_m ** 2) / Pc_m * \
            k * np.sqrt(a / (T * Tc_m))
            )
    
    def get_d2a_dT2(self, T, a, indexes = None):
        """Second order partial differentiation"""
        if indexes is None:
            indexes = np.arange(self.F.size)

        k = self.get_k_m(indexes=indexes)

        # First differentiation of a with respect to T
        da_dT = self.get_da_dT(T, a, indexes=indexes)

        # Critical temperature of mixture
        Tc_m = self.get_Tc_m(indexes=indexes)

        # Critical pressure of mixture
        Pc_m = self.get_Pc_m(indexes=indexes)

        return (
            -0.45724 * (self.R ** 2 * Tc_m ** 1.5) / Pc_m * \
            k * 
            np.sqrt(1 / T) * 1 / 2 * np.sqrt(1 / a) * da_dT + 
            np.sqrt(a) * (-1) / 2 * np.sqrt(1 / T ** 3)
            )

    def get_dP_dT(self, T, v, a, b):
        da_dT = self.get_da_dT(T, a)
        return self.R / (v - b) - da_dT / (v ** 2 + 2 * b * v - b ** 2)

    def get_dP_dv(self, T, v, a, b):
        return - self.R * T / (v - b) ** 2 + 2 * a * (v + b) / (v ** 2 + 2 * b * v - b ** 2) ** 2

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
        if indexes is None:
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
            enthalpy_temperature_corrected = (
                self.cp_constants_liq_arr[0, 0] / 1 * (T ** 1 - T_ref ** 1) + 
                self.cp_constants_liq_arr[0, 1] / 2 * (T ** 2 - T_ref ** 2) + 
                self.cp_constants_liq_arr[0, 2] / 3 * (T ** 3 - T_ref ** 3) + 
                self.cp_constants_liq_arr[0, 3] / 4 * (T ** 4 - T_ref ** 4) + 
                self.cp_constants_liq_arr[0, 4] / 5 * (T ** 5 - T_ref ** 5)
            )
        return enthalpy_temperature_corrected
    
    def get_capacity_temperature(self, T, is_H2O = False, indexes = None):
        if indexes is None:
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
            heat_capacity = (
                    self.cp_constants_liq_arr[0, 0] * T ** 0 + 
                    self.cp_constants_liq_arr[0, 1] * T ** 1 + 
                    self.cp_constants_liq_arr[0, 2] * T ** 2 + 
                    self.cp_constants_liq_arr[0, 3] * T ** 3
                )
        return heat_capacity
    
class EnthalpyPressure(PengRobinson):
    def __init__(self):
        super().__init__()
        self.R = 8.3145

    def __call__(self, T, P, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return self.get_enthalpy_pressure(T, P, indexes=indexes)
    
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

        cp_temperature = self.get_capacity_temperature(T, is_H2O=is_H2O, indexes=indexes)
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
        
        # for i in range(4):
        #     is_H2O = False
        #     if i == 1:
        #         is_H2O = True
        #     indexes = [i]
        #     enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[i] * 1000 / 3600
        
        self.F = self.F_init + self.stoich_coefs * X * self.F_start[self.A0_idx]
        
        is_H2O = False
        indexes = np.array([0, 2, 3])
        enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600

        is_H2O = True
        indexes = np.array([1])
        enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[indexes].sum() * 1000 / 3600

        # self.F = self.F_init + self.stoich_coefs * X * self.F_init[self.A0_idx]
        # for i in range(4):
        #     is_H2O = False
        #     if i == 1:
        #         is_H2O = True
        #     indexes = [i]
        #     enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O, indexes=indexes) * self.F[i] * 1000 / 3600
        
        enthalpy = enthalpy_prod - enthalpy_react + self.H_heat * X * self.F_init[self.A0_idx] * 1000 / 3600
        self.F = self.F_init.copy()
        return enthalpy
    
class OutletTemperature(EnthalpyReaction):
    """The class to compute outlet temperature from heat of reaction"""
    def __init__(self):
        super().__init__()
    
    def __call__(self, T, P, X):
        return self.get_T_final(T, P, X)

    def get_T_final(self, T, P, X, T_init_ref = None, heat_enthalpy_previous = 0, heat_prev = 0, T_prev = None, X_prev = None):
        """
        heat_prev:          Substract previous heat.
        T_init:             Deprecated
        """
        if T_init_ref is None:
            T_init_ref = T

        # theta_m = self.get_theta_m(indexes)
        
        # is_H2O = False
        # cp_theta += self.heat_capacity(T, P, is_H2O=is_H2O, indexes=indexes) * theta_m
        # print(cp_theta)

        # indexes = [1]
        # theta_m = self.get_theta_m(indexes)
        
        # is_H2O = False
        # cp_theta += self.heat_capacity(T, P, is_H2O=is_H2O, indexes=indexes) * theta_m
        # print(cp_theta)
        
        cp_theta = 0
        self.F = self.F_init + self.stoich_coefs * X * self.F_init[self.A0_idx]
        for i in range(7):
            is_H2O = False
            if i == 1:
                is_H2O = True
            indexes = np.array([i])
            theta = self.get_theta(indexes=indexes)
            cp_i = self.heat_capacity(T, P, is_H2O=is_H2O, indexes=indexes)
            cp_theta += cp_i * theta

        self.F = self.F_init.copy()
        heat_enthalpy = self.get_heat_enthalpy(T, P, X)

        dT = -heat_enthalpy / cp_theta
        T_final = T + dT        
        # print((heat_enthalpy - heat_prev), T, X)
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
        heat_enthalpy = 0
        X_prev = 0
        T_prev = T
        for i, X_value in enumerate(X):
            # print(self.F_init)
            T, heat_enthalpy, dT = self.get_T_final(T=T, P=P, X=X_change, #heat_enthalpy_previous=heat_enthalpy, 
                                                T_prev=T_prev, X_prev=X_prev,
                                                heat_prev=heat_enthalpy)
            # Pushing initial flow rates to the next case of new X value.
            self.F_init = self.F_start + self.stoich_coefs * X_value * self.F_start[self.A0_idx]
            # X_change = (1 - self.F_start[self.A0_idx] * (1 - X_value) / self.F_init[self.A0_idx])
            X_prev = X_value
            T_prev = T
            T_profile[i] = T
            enthalpy_profile[i] = heat_enthalpy
            dTs[i] = dT

        return X, T_profile, enthalpy_profile, dTs
    
class KineticModels(Data):
    def __init__(self, model: str = "rase"):
        super().__init__()
        if model == "rase":
            self.kinetic_model = self.Rase

        self.func = lambda T, P, X: (self.kinetic_model(T, P, X)) # / 1000 / 2.3 * np.exp(-X)
        
        self.Tc = np.array([
            self.Tc["CO"], self.Tc["H2O"], self.Tc["CO2"], self.Tc["H2"],
            self.Tc["Ar"], self.Tc["N2"], self.Tc["CH4"]
        ])
        self.Pc = np.array([
            self.Pc["CO"], self.Pc["H2O"], self.Pc["CO2"], self.Pc["H2"],
            self.Pc["Ar"], self.Pc["N2"], self.Pc["CH4"]
        ])
        self.w = np.array([
            self.w["CO"], self.w["H2O"], self.w["CO2"], self.w["H2"],
            self.w["Ar"], self.w["N2"], self.w["CH4"]
        ])
        self.F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])

        # self.Tc = np.array([
        #     self.Tc["CO"], self.Tc["H2O"], self.Tc["CO2"], self.Tc["H2"]
        # ])
        # self.Pc = np.array([
        #     self.Pc["CO"], self.Pc["H2O"], self.Pc["CO2"], self.Pc["H2"]
        # ])
        # self.w = np.array([
        #     self.w["CO"], self.w["H2O"], self.w["CO2"], self.w["H2"]
        # ])
        # self.F = np.array([
        #     self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
        #     self.F["Ar"], self.F["N2"], self.F["CH4"]
        # ])

        self.F_total = self.F[:].sum()
        self.f = self.F / self.F_total

    def __call__(self, T, P, X):
        return self.func(T, P, X)

    def get_final_f(self, X, f_A0, f_B0, stoich_coef):
        return f_B0 + stoich_coef * X * f_A0

    def Rase(self, T, P, X):
        if type(X) is float or type(X) is int:
            X = np.array([X])
        if type(T) in (float, int):
            T = np.array([T])
            
        T = T / (5 / 9)

        # bulk_density = 1173.1 * 2.20462 / 3.28084 ** 3
        bulk_density = self.bulk_density

        k = np.exp(12.88 - 3340 / T) # for copper-zinc catalyst
        
        K = np.zeros(T.size)
        K[T <= 1060] = np.exp(-4.72 + 8640 / T[T <= 1060])
        K[T > 1060] = np.exp(-4.33 + 8240 / T[T > 1060])
        A = 4.33
        f_final_CO = self.get_final_f(X, self.F[0], self.F[0], -1) / self.F_total
        f_final_H2O = self.get_final_f(X, self.F[0], self.F[1], -1) / self.F_total
        f_final_CO2 = self.get_final_f(X, self.F[0], self.F[2], +1) / self.F_total
        f_final_H2 = self.get_final_f(X, self.F[0], self.F[3], +1) / self.F_total
        
        rate = A * k * (f_final_CO * f_final_H2O - f_final_CO2 * f_final_H2 / K) / (379 * bulk_density)
        
        return rate

    def YongtaekHarvey(self, X, T, P):
        """
        """
        if type(X) is float or type(X) is int:
            X = np.array([X])

        eq_constant = self.get_eq_constant(T)
        R = 8.3145
        
        # kg / m^3
        d_cat = 5904

        # mol / (g h atm^2)
        A0 = 2.96 * 10 ** 5
        
        # J / mol
        Ea = 47400

        porosity = 0

        CO = (self.F[0] - self.F[0] * X)
        H2O = (self.F[1] - self.F[0] * X)
        CO2 = (self.F[2] + self.F[0] * X)
        H2 = (self.F[3] + self.F[0] * X)
        
        total = (CO + H2O + CO2 + H2)
        
        CO = CO / total
        H2O = H2O / total
        CO2 = CO2 / total
        H2 = H2 / total
        rate = d_cat * (1 - porosity) * A0 * np.exp(-Ea / (R * T)) * P ** 2 * (CO * H2O - CO2 * H2 / eq_constant)

        # mol / m^3 h
        rate = 1000 * rate / 101325 ** 2
        return rate

    def LH_1_3(self, X, T, P):
        """
        J. L. Ayastuy, et al. "Kinetics of the Low-Temperature WGS Reaction over a
                                CuO/ZnO/Al2O3 Catalyst"
        """
        if type(X) is float or type(X) is int:
            X = np.array([X])
        eq_constant = self.get_eq_constant(T)
        R = 8.3145

        Ea = -36.558 
        H_CO = -45.99651
        H_H2O = -79.963
        H_CO2 = -16.474
        H_H2 = -13.279
        
        k = 1.188 * np.exp(Ea * 1000 / (R * T))
        K_CO = 2.283 * 10 ** -24 * np.exp(H_CO * 1000 / (R * T))
        K_H2O = 1.957 * 10 ** -28 * np.exp(H_H2O * 1000 / (R * T))
        K_CO2 = 5.419 * 10 ** -4 * np.exp(H_CO2 * 1000 / (R * T))
        K_H2 = 2.349 * 10 ** -4 * np.exp(H_H2 * 1000 / (R * T))

        P_CO = P_final(PB0=P * self.f[0], PA0=P * self.f[0], X=X, stoich_coef=-1)
        P_H2O = P_final(PB0=P * self.f[1], PA0=P * self.f[0], X=X, stoich_coef=-1)
        P_CO2 = P_final(PB0=P * self.f[2], PA0=P * self.f[0], X=X, stoich_coef=+1)
        P_H2 = P_final(PB0=P * self.f[3], PA0=P * self.f[0], X=X, stoich_coef=+1)

        nominator = k * (P_CO * P_H2O - P_CO2 * P_H2 / eq_constant)
        denominator = (
            1 + K_CO * P_CO + K_H2O * P_H2O + K_CO2 * P_CO2 + K_H2 * P_H2
            ) ** 2

        # mol/ (g * h)
        rate = nominator / denominator
        
        # mol/ (g * s)
        rate = rate / 3600
        
        # 1 / s
        rate = rate * 28
        return rate


    def LH_7_3(self, X, T, P):
        """
        J. L. Ayastuy, et al. "Kinetics of the Low-Temperature WGS Reaction over a
                                CuO/ZnO/Al2O3 Catalyst"
        Args:
            X:                      conversion (float)
            T:                      temperature (float or int)
            P:                      pressure in Pa
            eq_constsant:           equilibrium constant (float)
            R:                      universal gas constant (float)
            kwargs:                 additional argumewnts for fractions and total pressure (dict)
        """
        if type(X) is float or type(X) is int:
            X = np.array([X])

        # theta_CO = params["theta"]["theta_CO"]
        # theta_CO2 = params["theta"]["theta_CO2"]
        # theta_H2O = params["theta"]["theta_H2O"]
        # theta_H2 = params["theta"]["theta_H2"]
        
        eq_constant = self.get_eq_constant(T)

        ln_A0 = 16.99
        Ea_R = 6049.2

        ln_A_CO = -2.362
        H_CO_R = 1782.1

        ln_A_H2O = -3.403
        H_H2O_R = 2088.8

        ln_A_H2 = -3.459
        H_H2_R = 2057.7

        ln_A_CO2 = -5.765
        H_CO2_R = 3003.5

        k0 = np.exp(ln_A0 - Ea_R / T)
        K_CO = np.exp(ln_A_CO - H_CO_R / T)
        K_H2O = np.exp(ln_A_H2O - H_H2O_R / T)
        K_H2 = np.exp(ln_A_H2 - H_H2_R / T)
        K_CO2 = np.exp(ln_A_CO2 - H_CO2_R / T)

        P_CO = P_final(PB0=P * self.f[0], PA0=P * self.f[0], X=X, stoich_coef=-1)
        P_H2O = P_final(PB0=P * self.f[1], PA0=P * self.f[0], X=X, stoich_coef=-1)
        P_CO2 = P_final(PB0=P * self.f[2], PA0=P * self.f[0], X=X, stoich_coef=+1)
        P_H2 = P_final(PB0=P * self.f[3], PA0=P * self.f[0], X=X, stoich_coef=+1)

        nominator = k0 * (P_CO * P_H2O - P_CO2 * P_H2 / eq_constant)
        denominator = (
            1 + K_CO * P_CO + K_H2O * P_H2O + K_CO2 * P_CO2 * P_H2 ** 0.5 + K_H2 ** 0.5 * P_H2 ** 0.5
            ) ** 2

        # mol/ (g * h)
        rate = nominator / denominator
        
        # mol/ (g * s)
        rate = rate / 3600
        
        return rate
    
    def get_eq_constant(self, T):
        return np.exp(
            5693.5 / T + 1.077 * np.log(T) + 5.44 * 10 ** -4 * T - 1.125 * 10 ** -7 * T ** 2 - 49170 / T ** 2 - 13.148
        )
    
    def P_final(self, PB0, PA0, X, stoich_coef: int = 1):
        return PB0 * (1 + stoich_coef * X * PA0 / PB0)

class PressureDrop(Data):
    def __init__(self):
        self.Tc = np.array([
            self.Tc["CO"], self.Tc["H2O"], self.Tc["CO2"], self.Tc["H2"]
        ])
        self.Pc = np.array([
            self.Pc["CO"], self.Pc["H2O"], self.Pc["CO2"], self.Pc["H2"]
        ])
        self.w = np.array([
            self.w["CO"], self.w["H2O"], self.w["CO2"], self.w["H2"]
        ])
        self.F = np.array([
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"],
            self.F["Ar"], self.F["N2"], self.F["CH4"]
        ])
        self.nu_const = np.array([
            self.nu_gas["CO"], self.nu_gas["H2O"], self.nu_gas["CO2"], self.nu_gas["H2"],
            self.nu_gas["Ar"], self.nu_gas["N2"], self.nu_gas["CH4"]
            ]
         )
        
        self.Mw = np.array([
            self.Mw["CO"], self.Mw["H2O"], self.Mw["CO2"], self.Mw["H2"],
            self.Mw["Ar"], self.Mw["N2"], self.Mw["CH4"]
        ])
        
        self.F_total = self.F.sum()

        self.f = self.F / self.F_total

        # g/mol or kg/kmol
        self.Mw_mixture = (self.f * self.Mw).sum()
        
        # Particle diameter
        self.dP = self.diameter / 1000

        # Bulk density in kg/m^3
        self.bulk_density = self.bulk_density / self.conversion
        self.PR_model = PengRobinson()
        
    def __call__(self, T0, P0, T, weight, A_cross):
        indexes = np.array([0, 2, 3, 4, 5, 6])
        nu = self.get_viscosity_mixture(T=T.mean(), indexes=indexes)

        # m^3 / mol
        volume = self.PR_model(T0, P0, optimize=True, indexes=indexes)
        
        # kg/m^3
        density = self.Mw_mixture / volume / 1000

        # catalyst_density = bulk_density / (1 - self.porosity)
        
        # m^3 / s
        volumetic_flowrate = self.F_total * volume * 1000 / 3600

        # m / s
        velocity = volumetic_flowrate / A_cross

        # superficial velocity (kg / m^2 s)
        G = density * velocity

        term_1 = (150 * (1 - self.porosity) * nu) / self.dP
        term_2 = 1.75 * G
        beta = G * (1 - self.porosity) / (density * self.dP * self.porosity ** 3) * term_1 * term_2
        alfa = beta / (A_cross * self.bulk_density)
        P = np.sqrt(P0**2 - 2 * alfa * T / T0 * P0 * weight)
        return P - P0
        
    def get_viscosity(self, T, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return (self.nu_const[indexes, 0] + self.nu_const[indexes, 1] * T + self.nu_const[indexes, 2] * T ** 2) * 10 ** -7

    def get_viscosity_mixture(self, T, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        return (self.get_viscosity(T, indexes) * self.f[indexes]).sum()

def P_final(PB0, PA0, X, stoich_coef: int = 1):
    return PB0 * (1 + stoich_coef * X * PA0 / PB0)

def get_eq_constant(T):
    return np.exp(
        5693.5 / T + 1.077 * np.log(T) + 5.44 * 10 ** -4 * T - 1.125 * 10 ** -7 * T ** 2 - 49170 / T ** 2 - 13.148
    )

if __name__ == "__main__":
    # model = PengRobinson()
    P = 34.2 * 10 ** 5
    # indexes = np.array([0, 2, 3])
    # v = model(T=478, P=P, v=0.0001, optimize=True, indexes=indexes)
    # print(v)
    # P_calc = model(T=478, P=P, v=v, optimize=False, indexes=indexes)
    # print(P_calc, P - P_calc)
    
    # CO, H2O, CO2, H2
    # Heat of reaction
    # model = EnthalpyReaction(X_objective=0.95)
    # print(model(T=478, P=P))
    
    # model = Enthalpy()
    T = 475
    # print(model(T=T, P=P, indexes=[3], method="capacity"))

    # model = OutletTemperature(X_objective=0.9)
    # Tout = model(T, P)
    # print(Tout)

    # model = PressureDrop()
    # weight = 1173.1 * 50
    # d = 4
    # A_cross = np.pi * d ** 2 / 4
    # print(model(T0=478, P0=34.2e5, T=501, weight=weight, A_cross=A_cross))
    # # EnthalpyTemperature()


    # import matplotlib.pyplot as plt

    # model = OutletTemperatureProfile()
    # X, T_profile, enthalpy_profile = model(T=478, P=34.2e5, X_objective=0.9)
    
    # # # plt.plot(X, -enthalpy_profile)
    # # # plt.show()

    # plt.plot(X, T_profile)
    # plt.show()

    # model = KineticModels()

    # print(X.shape, T_profile.shape)
    # rates = model(T=T_profile, P=34.2e5, X=X)

    # plt.plot(X, rates)
    # plt.show()

    # print(rates.shape)
    
    # import matplotlib.pyplot as plt

    # plt.title("Rate profile versus Conversion")
    # plt.xlabel("Conversion")
    # plt.ylabel("Rate")
    # plt.plot(results["X_profile"], results["rate_profile"])
    # plt.savefig("./images/rate_profile.png")

    # plt.title("Heat profile versus Conversion")
    # plt.xlabel("Conversion")
    # plt.ylabel("Heat")
    # plt.plot(results["X_profile"], results["enthalpy_profile"])
    # plt.savefig("./images/heat_profile.png")

    # plt.title("Temperature profile versus Conversion")
    # plt.xlabel("Conversion")
    # plt.ylabel("Temperature")
    # plt.plot(results["X_profile"], results["T_profile"])
    # plt.savefig("./images/temperature_profile.png")

    # plt.title("Pressure Drop versus Conversion")
    # plt.xlabel("Conversion")
    # plt.ylabel("Pressure Drop")
    # plt.plot(results["X_profile"], results["pressure_drop"])
    # plt.savefig("./images/pressure_drop_profile.png")
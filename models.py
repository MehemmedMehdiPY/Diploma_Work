import numpy as np
from scipy.optimize import root, minimize, Bounds
from scipy.integrate import simpson
from constants import Data

def inverse_ReLU(x):
    return (x if x < 0 else 0)

class PengRobinson(Data):
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
            self.F["CO"], self.F["H2O"], self.F["CO2"], self.F["H2"]
        ])

        self.cp_constants_gas_arr = np.array(
            [self.cp_constants_gas["CO"],
            self.cp_constants_gas["H2O"],
            self.cp_constants_gas["CO2"],
            self.cp_constants_gas["H2"]]
        ).reshape(4, 5)

        self.cp_constants_liq_arr = np.array(
            self.cp_constants_liq["H2O"]
        ).reshape(1, 5)

        # self.Tc = np.array([
        #     self.Tc["CO"], self.Tc["CO2"], self.Tc["H2"]
        # ])
        # self.Pc = np.array([
        #     self.Pc["CO"], self.Pc["CO2"], self.Pc["H2"]
        # ])
        # self.w = np.array([
        #     self.w["CO"], self.w["CO2"], self.w["H2"]
        # ])
        # self.F = np.array([
        #     self.F["CO"], self.F["CO2"], self.F["H2"]
        # ])

        # self.f = self.F / self.F.sum()
        self.R = 8.3145
        self.func = lambda v, T, a, b, R: R * T / (v - b) - a / (v * (v + b) + b * (v - b))
        self.func_loss = lambda v, P, T, a, b, R: R * T / (v - b) - a / (v * (v + b) + b * (v - b)) - P

    def __call__(self, T, P, v = 0.001, optimize = True, indexes = None):
        return self.run(T, P, v, optimize=optimize, indexes=indexes)
        
    def run(self, T, P, v = 0.001, optimize = True, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        self.indexes = indexes

        f = self.F[self.indexes] / self.F[self.indexes].sum()
        a = self.get_am(T=T, f=f)
        b = self.get_bm(f=f)

        if optimize:
            v = self.approximate_v(v, T, P, a, b)
            return v
        else:
            return self.func(v, T, a, b, self.R)
    
    def approximate_v(self, v, T, P, a, b):
        f = self.func_loss
        results = root(f, x0=v, args=(P, T, a, b, self.R))
        return results.x

    def get_a(self, T):
        a_Tc = 0.45724 * self.R ** 2 * self.Tc[self.indexes] ** 2 / self.Pc[self.indexes]
        k = self.get_k()
        Tr = T / self.Tc[self.indexes]
        a_Tr_w = (1 + k * (1 - np.sqrt(Tr))) ** 2
        a = a_Tc * a_Tr_w
        return a

    def get_am(self, T, f):
        a = self.get_a(T)
        a_mat = np.zeros((len(self.indexes), len(self.indexes)))
        f_mat = np.zeros((len(self.indexes), len(self.indexes)))
        
        for i in range(len(self.indexes)):
            a_mat[i] = a * a[i]
            f_mat[i] = f * f[i]

        am = np.sum(
            np.sqrt(a_mat) * f_mat
            )
        return am

    def get_b(self):
        return (0.07780 * self.R * self.Tc[self.indexes] / self.Pc[self.indexes])

    def get_bm(self, f):
        return np.sum(f * self.get_b())

    def get_k(self):
        k = 0.37464 + 1.54226 * self.w[self.indexes] - 0.26992 * self.w[self.indexes] ** 2
        return k
    
    def get_da_dT(self, T, a):
        """First order partial differentiation"""
        k = self.get_k()
        return (
            -0.45724 * (self.R ** 2 * self.Tc[self.indexes] ** 2) / self.Pc[self.indexes] * \
            k * np.sqrt(a / (T * self.Tc[self.indexes]))
            )
    
    def get_d2a_dT2(self, T, a):
        """Second order partial differentiation"""
        k = self.get_k()
        da_dT = self.get_da_dT(T, a)
        return (
            -0.45724 * (self.R ** 2 * self.Tc[self.indexes] ** 1.5) / self.Pc[self.indexes] * \
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
        self.indexes = indexes
        return self.get_enthalpy_temperature(T, is_H2O)
    
    def get_enthalpy_temperature(self, T, is_H2O):
        T_ref = 298
        if not is_H2O:
            enthalpy_temperature_corrected = (
                self.cp_constants_gas_arr[self.indexes, 0] / 1 * (T ** 1 - T_ref ** 1) + 
                self.cp_constants_gas_arr[self.indexes, 1] / 2 * (T ** 2 - T_ref ** 2) + 
                self.cp_constants_gas_arr[self.indexes, 2] / 3 * (T ** 3 - T_ref ** 3) + 
                self.cp_constants_gas_arr[self.indexes, 3] / 4 * (T ** 4 - T_ref ** 4) + 
                self.cp_constants_gas_arr[self.indexes, 4] / 5 * (T ** 5 - T_ref ** 5)
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
    
    def get_capacity_temperature(self, T, P, v, a, b, is_H2O=False):
        if not is_H2O:
            heat_capacity = (
                    self.cp_constants_gas_arr[self.indexes, 0] * T ** 0 + 
                    self.cp_constants_gas_arr[self.indexes, 1] * T ** 1 + 
                    self.cp_constants_gas_arr[self.indexes, 2] * T ** 2 + 
                    self.cp_constants_gas_arr[self.indexes, 3] * T ** 3 + 
                    self.cp_constants_gas_arr[self.indexes, 4] * T ** 4
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
        self.indexes = indexes
        return self.get_enthalpy_pressure(T, P)
    
    def get_enthalpy_pressure(self, T, P, is_H2O=True):
        if is_H2O:
            """Water is negligbly compressible and pressure influence is insignificant"""
            return 0

        R = self.R
        f = self.F[self.indexes] / self.F[self.indexes].sum()
        a = self.get_am(T, f)
        b = self.get_bm(f)
        
        v_init = 0.0001
        v_real = self.approximate_v(v_init, T, P, a, b)
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
        self.indexes = indexes

        if method.lower() == "enthalpy":    
            return self.get_total_enthalpy(T, P, is_H2O)
        elif method.lower() == "capacity":
            return self.heat_capacity(T, P)

    def get_total_enthalpy(self, T, P, is_H2O):
        enthalpy_temperature_corrected = self.get_enthalpy_temperature(T, is_H2O=is_H2O)
        enthalpy_pressure_corrected = self.get_enthalpy_pressure(T, P, is_H2O=is_H2O)
        return enthalpy_temperature_corrected + enthalpy_pressure_corrected

    def heat_capacity(self, T, P, is_H2O = False):
        f = self.F[self.indexes] / self.F[self.indexes].sum()
        a = self.get_am(T, f)
        b = self.get_bm(f)
        v_init = 0.0001
        v = self.approximate_v(v_init, T, P, a, b)
        cp_temperature = self.get_capacity_temperature(T, P, v, a, b, is_H2O=is_H2O)
        cp_pressure = self.get_capacity_pressure(T, P, v, a, b, is_H2O=is_H2O)
        # cp_pressure = 0
        return cp_temperature + cp_pressure

class EnthalpyReaction(Enthalpy):
    def __init__(self, X_objective):
        super().__init__()
        self.H_heat = -41100
        self.X_objective = X_objective
        self.F_init = self.F.copy()
        self.stoich_coefs = np.array([-1, -1, +1, +1])
        self.A0_idx = 0
        
    def __call__(self, T, P):
        return self.get_heat_enthalpy(T, P)    

    def get_heat_enthalpy(self, T, P):
        enthalpy_react = 0
        enthalpy_prod = 0

        for i in range(4):
            is_H2O = False
            if i == 1:
                is_H2O = True
            self.indexes = [i]
            enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=is_H2O) * self.F[i] * 1000 / 3600
        
        # Mixing considered
        # self.indexes = [0, 2, 3]
        # enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=False) * self.F[self.indexes].sum() * 1000 / 3600

        # self.indexes = [1]
        # enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=True) * self.F[self.indexes].sum() * 1000 / 3600
        
        self.F = self.F_init + self.stoich_coefs * self.X_objective * self.F_init[self.A0_idx]
        for i in range(4):
            is_H2O = False
            if i == 1:
                is_H2O = True
            self.indexes = [i]
            enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O) * self.F[i] * 1000 / 3600

        # Mixing considered
        # self.indexes = [0, 2, 3]
        # enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=False) * self.F[self.indexes].sum() * 1000 / 3600
        
        # self.indexes = [1]
        # enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=True) * self.F[self.indexes].sum() * 1000 / 3600
            
        enthalpy = enthalpy_prod - enthalpy_react + self.H_heat * self.X_objective * self.F_init[self.A0_idx] * 1000 / 3600

        self.F = self.F_init.copy()

        return enthalpy
    
class OutletTemperature(EnthalpyReaction):
    """The class to compute outlet temperature from heat of reaction"""
    def __init__(self, X_objective):
        super().__init__(X_objective)
        self.theta = self.F / self.F[self.A0_idx]
    
    def __call__(self, T, P):
        cp_theta = 0
        for i in range(4):
            is_H2O = False
            if i == 1:
                is_H2O = True
            self.indexes = [i]
            cp_i = self.heat_capacity(T, P, is_H2O=is_H2O)
            cp_theta += cp_i * self.theta[i]
            
        heat_enthalpy = self.get_heat_enthalpy(T, P)
        T_final = T - heat_enthalpy * self.X_objective / cp_theta * np.exp(-self.X_objective * 1.3)  * np.exp(-298 / T * 5)
        return T_final              

class KineticModels(Data):
    def __init__(self, X_objective, model: str = "LH_7_3"):
        super().__init__()
        if model == "LH_7_3":
            self.kinetic_model = self.Rase

        self.X = X_objective
        self.func = lambda T, P, X: (self.kinetic_model(T, P, X)) # / 1000 / 2.3 * np.exp(-X)
        
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
        self.F_total = self.F.sum()

        self.f = self.F / self.F_total
        
    def __call__(self, T, P, X=None):
        if X is None:
            X = self.X
        return self.func(T, P, X)

    def get_final_f(self, X, f_A0, f_B0, stoich_coef):
        return f_B0 + stoich_coef * X * f_A0

    def Rase(self, T, P, X):
        if type(X) is float or type(X) is int:
            X = np.array([X])
        T = T / (5 / 9)
        bulk_density = 1173.1 * 2.20462 / 3.28084 ** 3

        k = np.exp(12.88 - 3340 / T) # for copper-zinc catalyst
        K = np.exp(-4.72 + 8640 / T)
        A = 4.33
        
        f_final_CO = self.get_final_f(X, self.F[0], self.F[0], -1) / self.F_total
        f_final_H2O = self.get_final_f(X, self.F[0], self.F[1], -1) / self.F_total
        f_final_CO2 = self.get_final_f(X, self.F[0], self.F[2], +1) / self.F_total
        f_final_H2 = self.get_final_f(X, self.F[0], self.F[3], +1) / self.F_total
        
        rate = A * k * (f_final_CO * f_final_H2O - f_final_CO2 * f_final_H2 / K) / (379 * bulk_density)
        
        K = 162.2926
        k = 8901.193
        
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

class Cost(Data):
    def __init__(self, func, X_objective, kinetic_model_name):
        self.func = func
        self.kinetic_model = KineticModels(X_objective=X_objective, model=kinetic_model_name)
        self.flow_rate = self.F["CO"] * 1000 / 3600
        self.density = 1173.1
        self.X_objective = X_objective

    def __call__(self, args):
        T, P = args
        volume = self.get_volume(T, P)
        return self.func(T, P, volume)
    
    def get_volume(self, T, P):
        xs = np.linspace(0, self.X_objective, num=10000)
        ys = 1 / self.kinetic_model(T=T, P=P, X=xs)
        weight = simpson(y=ys, x=xs) * self.flow_rate
        return weight / self.density
    
class Optimization(Data):
    def __init__(self, X_objective, cost_func, kinetic_model_name = "LH_7_3"):
        self.cost = Cost(cost_func, X_objective, kinetic_model_name)
        self.kinetic_model_name = kinetic_model_name
        self.temperature_estimator = OutletTemperature(X_objective=X_objective)
        self.X_objective = X_objective
        

    def __call__(self, T, P, *args):
        return self.converge(T=T, P=P)
    
    def converge_(self, T, P):
        variables = np.array([T, P])
        for _ in range(1000):            
            gradients = self.get_gradients(*variables.tolist())
            variables = variables - gradients * 0.1
        return variables
    
    def converge(self, T, P):
        bounds = Bounds(lb=(400, 24.2 * 1e5), ub=np.inf)
        variables = [T, P]
        results = minimize(fun=self.cost, x0=variables, bounds=bounds)
        return results.x, self.cost(results.x)

def P_final(PB0, PA0, X, stoich_coef: int = 1):
    return PB0 * (1 + stoich_coef * X * PA0 / PB0)

def P_final_modified(P_total, Pi_init, X, stoich_coef: int = 1, theta: float = 1, eps: float = 0):
    """
    theta = FB0 / FA0
    """
    P_ref = 10e5
    p = P_total / P_ref
    Pi_final = Pi_init * (theta + stoich_coef * X) / (1 + eps * X) * p
    return Pi_final

def R_eq_1(X, params):
    """
    Seyed R. J., et al. "Simulation and optimization of water gas shift process in ammonia plant: 
                        Maximizing CO conversion and controlling methanol byproduct"

    Problem: The model has the suspicious value of a rate constant and no description for units is provided.

    Args:
        X:                      conversion (float)
        T:                      temperature (float or int)
        eq_constsant:           equilibrium constant (float)
        R:                      universal gas constant (float)
        kwargs:                 additional argumewnts for fractions and total pressure (dict)
    """
    T = params["T"]
    eq_constant = params["eq_constant"]
    R = 8.3145
    P_total = params["P_total"] / 10 ** 5
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]
        
    P_CO = P_final(PB0=P_total * f_CO, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_H2O = P_final(PB0=P_total * f_H2O, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_CO2 = P_final(PB0=P_total * f_CO2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    P_H2 = P_final(PB0=P_total * f_H2, PA0=P_total * f_CO, X=X, stoich_coef=+1)

    return (
        2.96 * 10 ** 5 * np.exp(-47400 / R / T) * \
            (P_CO * P_H2O - P_CO2 * P_H2 / eq_constant)
    )

def LH_1_3(X, params):
    """
    J. L. Ayastuy, et al. "Kinetics of the Low-Temperature WGS Reaction over a
                            CuO/ZnO/Al2O3 Catalyst"
    """
    if type(X) is float or type(X) is int:
        X = np.array([X])
    eq_constant = params["eq_constant"]
    T = params["T"]
    P_total = params["P_total"] / 10 ** 5
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]
    
    ln_A0 = 18.38
    Ea_R = 6940.6

    ln_A_CO = -1.735
    H_CO_R = 1370.5

    ln_A_H2O = -3.051
    H_H2O_R = 1800.0

    ln_A_H2 = -3.100
    H_H2_R = 1743.7

    ln_A_CO2 = -3.322
    H_CO2_R = 1978.0

    k0 = np.exp(ln_A0 - Ea_R / T)
    K_CO = np.exp(ln_A_CO - H_CO_R / T)
    K_H2O = np.exp(ln_A_H2O - H_H2O_R / T)
    K_H2 = np.exp(ln_A_H2 - H_H2_R / T)
    K_CO2 = np.exp(ln_A_CO2 - H_CO2_R / T)
    
    P_CO = P_final(PB0=P_total * f_CO, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_H2O = P_final(PB0=P_total * f_H2O, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_CO2 = P_final(PB0=P_total * f_CO2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    P_H2 = P_final(PB0=P_total * f_H2, PA0=P_total * f_CO, X=X, stoich_coef=+1)

    nominator = k0 * (P_CO * P_H2O - P_CO2 * P_H2 / eq_constant)
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

def LH_7_3(X, params):
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
    eq_constant = params["eq_constant"]
    T = params["T"]
    P_total = params["P_total"] / 10 ** 5
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]

    theta_CO = params["theta"]["theta_CO"]
    theta_CO2 = params["theta"]["theta_CO2"]
    theta_H2O = params["theta"]["theta_H2O"]
    theta_H2 = params["theta"]["theta_H2"]
    
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

    P_CO = P_final(PB0=P_total * f_CO, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_H2O = P_final(PB0=P_total * f_H2O, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_CO2 = P_final(PB0=P_total * f_CO2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    P_H2 = P_final(PB0=P_total * f_H2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    
    # P_CO = P_final_modified(P_total, P_total * f_CO, X, stoich_coef=-1, eps=0, theta=theta_CO)
    # P_H2O = P_final_modified(P_total, P_total * f_H2O, X, stoich_coef=-1, eps=0, theta=theta_H2O)
    # P_CO2 = P_final_modified(P_total, P_total * f_CO2, X, stoich_coef=+1, eps=0, theta=theta_CO2)
    # P_H2 = P_final_modified(P_total, P_total * f_H2, X, stoich_coef=+1, eps=0, theta=theta_H2)

    nominator = k0 * (P_CO * P_H2O - P_CO2 * P_H2 / eq_constant)
    denominator = (
        1 + K_CO * P_CO + K_H2O * P_H2O + K_CO2 * P_CO2 * P_H2 ** 0.5 + K_H2 ** 0.5 * P_H2 ** 0.5
        ) ** 2

    # mol/ (g * h)
    rate = nominator / denominator
    
    # mol/ (g * s)
    rate = rate / 3600
    
    return rate

def Power_Law(X, params):
    if type(X) is float or type(X) is int:
        X = np.array([X])
    
    T = params["T"]
    eq_constant = params["eq_constant"]
    P_total = params["P_total"] / 10 ** 5
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]

    ln_A0 = 19.25
    Ea = 79.7e3
    k = np.exp(ln_A0 - Ea / 8.3145 / T)
    
    a = 0.47
    b = 0.72
    c = -0.65
    d = -0.38

    P_CO = P_final(PB0=P_total * f_CO, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_H2O = P_final(PB0=P_total * f_H2O, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_CO2 = P_final(PB0=P_total * f_CO2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    P_H2 = P_final(PB0=P_total * f_H2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    
    beta = 1 - P_H2 * P_CO2 / (P_CO * P_H2O) * 1 / eq_constant
    
    # mol/ (g * h)
    rate = k * (P_CO ** a) * (P_H2O ** b) * (P_H2 ** c) * (P_CO2 ** d) * (1 - beta)

    # mol/ (g * s)
    rate = rate / 3600
    
    # 1 / s
    rate = rate * 28
    return rate

def R_eq__(X, params, threshold: float = 0):
    """
    Diogo M., et al. "Determination of the Low-Temperature Water-Gas Shift Reaction Kinetics Using
                        a Cu-Based Catalyst"

    The model has the suspicious value of a rate constant and no description for units is provided.

    Args:
        X:                      conversion (float)
        T:                      temperature (float or int)
        eq_constsant:           equilibrium constant (float)
        R:                      universal gas constant (float)
        kwargs:                 additional argumewnts for fractions and total pressure (dict)
    """
    eq_constant = params["eq_constant"]
    P_total = params["P_total"]
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]

    k0 = 3.101e-4
    K_CO2 = 1.882e-2

    # CO
    P_CO = P_final(P_total * f_CO, X, -1)
    
    # H2O
    P_H2O = P_final(P_total * f_H2O, X, -1)

    # CO2
    P_CO2 = P_final(P_total * f_CO2, X, +1)

    # H2
    P_H2 = P_final(P_total * f_H2, X, +1)

    nominator = k0 * (P_H2O - P_CO2 * P_H2 / (eq_constant * P_CO))
    denominator = (
        1 + K_CO2 * P_CO2 / P_CO
        )
    ratio = nominator / denominator
    return (
        ratio
    )

def Temkin(X, params):
    if type(X) is float or type(X) is int:
        X = np.array([X])
    
    T = params["T"]
    eq_constant = np.exp(4577.8 / T - 4.33)
    P_total = params["P_total"] / 10 ** 5
    f_CO = params["fractions"]["f_CO"]
    f_CO2 = params["fractions"]["f_CO2"]
    f_H2O = params["fractions"]["f_H2O"]
    f_H2 = params["fractions"]["f_H2"]

    P_CO = P_final(PB0=P_total * f_CO, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_H2O = P_final(PB0=P_total * f_H2O, PA0=P_total * f_CO, X=X, stoich_coef=-1)
    P_CO2 = P_final(PB0=P_total * f_CO2, PA0=P_total * f_CO, X=X, stoich_coef=+1)
    P_H2 = P_final(PB0=P_total * f_H2, PA0=P_total * f_CO, X=X, stoich_coef=+1)

    beta = 1 - P_H2 * P_CO2 / (P_CO * P_H2O) * 1 / eq_constant

    k = 6 * 10 ** 11 * np.exp(-26800 / (1.987 * T))
    k = k * 101325
    
    A = 2.5 * 10 ** 9 * np.exp(-21500 / (1.987 * T))

    rate = k * P_H2O * P_CO * (1 - beta) / (A * P_H2O + P_CO2)

    return rate

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

    model = OutletTemperature(X_objective=0.9)
    Tout = model(T, P)
    print(Tout)
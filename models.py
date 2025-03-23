import numpy as np
from scipy.optimize import root
from constants import Data

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
        # print(self.func(results.x, T, a, b, self.R) / 10 ** 5)
        # print(results.x)
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
        
    def get_da_dT(self, T, a):
        k = self.get_k()
        return (
            -0.45724 * (self.R ** 2 * self.Tc[self.indexes] ** 2) / self.Pc[self.indexes] * \
            k * np.sqrt(a / (T * self.Tc[self.indexes]))
            )        

class Enthalpy(EnthalpyTemperature, EnthalpyPressure):
    def __init__(self):
        super().__init__()
        EnthalpyPressure.__init__(self)

    def __call__(self, T, P, indexes = None):
        if indexes is None:
            indexes = np.arange(self.F.size)
        self.indexes = indexes
        return self.get_total_enthalpy(T, P)
    
    def get_total_enthalpy(self, T, P, is_H2O):
        enthalpy_temperature_corrected = self.get_enthalpy_temperature(T, is_H2O=is_H2O)
        enthalpy_pressure_corrected = self.get_enthalpy_pressure(T, P, is_H2O=is_H2O)
        return enthalpy_temperature_corrected + enthalpy_pressure_corrected

class EnthalpyReaction(Enthalpy):
    def __init__(self, X_objective):
        super().__init__()
        self.H_heat = -41100
        self.X_objective = X_objective
        self.F_init = self.F.copy()
        self.stoich_coefs = np.array([-1, -1, +1, +1])
        self.A0_idx = 0
        
    def __call__(self, T, P):
        enthalpy_react = 0
        enthalpy_prod = 0
        for i in range(4):
            is_H2O = False
            if i == 1:
                is_H2O = True
            self.indexes = [i]
            enthalpy_react += self.get_total_enthalpy(T, P, is_H2O=is_H2O) * self.F[i] * 1000 / 3600
        
        self.F = self.F_init + self.stoich_coefs * self.X_objective * self.F_init[self.A0_idx]
        for i in range(4):
            is_H2O = False
            if i == 1:
                is_H2O = True
            self.indexes = [i]
            enthalpy_prod += self.get_total_enthalpy(T, P, is_H2O=is_H2O) * self.F[i] * 1000 / 3600
        enthalpy = enthalpy_prod - enthalpy_react + self.H_heat * self.X_objective * self.F_init[self.A0_idx] * 1000 / 3600

        self.F = self.F_init.copy()
        return enthalpy

class KineticModels(Data):
    def __init__(self, model: str = "LH_7_3"):
        super().__init__()
        if model == "LH_7_3":
            self.kinetic_model = self.LH_7_3
            pass

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
        self.f = self.F / self.F.sum()

    def __call__(self, X, T, P):
        return self.kinetic_model(X, T, P)

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
        print(P_CO, P_H2O, P_CO2, P_H2)

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
    model = EnthalpyReaction(X_objective=0.95)
    print(model(T=478, P=P))
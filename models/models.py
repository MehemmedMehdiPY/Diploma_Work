import numpy as np
from scipy.optimize import root
import sys
sys.path.append("../")
from constants.constants import Data

class SaturationTemperature(Data):
    def __init__(self):
        self.sat_constants = self.sat_constants["H2O"]
        self.func = lambda T: (133.32236836846923 * 10 ** (
            self.sat_constants[0] + self.sat_constants[1] / T + self.sat_constants[2] * np.log10(T)
            + self.sat_constants[3] * T + self.sat_constants[4] * T ** 2
            ))
        self.func_loss = lambda T, P: (133.32236836846923 / 10 ** 5 * 10 ** (
            self.sat_constants[0] + self.sat_constants[1] / T + self.sat_constants[2] * np.log10(T)
            + self.sat_constants[3] * T + self.sat_constants[4] * T ** 2
            ) - P
            )
        
    def __call__(self, P, T = 500, optimize = True):
        return self.run(T, P, optimize=optimize)
        
    def run(self, T, P, optimize = True):

        if optimize:
            T_sat = self.approximate_sat_T(T, P)
            return T_sat
        else:
            return self.func(T)

    def approximate_sat_T(self, T, P):
        P = P / 10 ** 5
        f = self.func_loss
        args = (P, )
        results = root(f, x0=T, args=args)
        return results.x    

    def get_sat_temp(self, T):
        return self.func(T)

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


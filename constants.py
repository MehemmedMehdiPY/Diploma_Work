import numpy as np

class Data:
    """Constant parameters and steady variables"""
    Tc = {
        "CO": 132.92,
        "H2O": 647.13,
        "CO2": 304.19,
        "H2": 33.18,
        "Ar": 150.86,
        "N2": 126.10,
        "CH4": 190.58
    }

    Pc = {
        "CO": 34.99 * 10 ** 5,
        "H2O": 220.55 * 10 ** 5,
        "CO2": 73.82 * 10 ** 5,
        "H2": 13.13 * 10 ** 5,
        "Ar": 48.98 * 10 ** 5,
        "N2": 33.94 * 10 ** 5,
        "CH4": 46.04 * 10 ** 5
    }

    w = {
        "CO": 0.066,
        "H2O": 0.345,
        "CO2": 0.228,
        "H2": -0.22,
        "Ar": 0.000,
        "N2": 0.040,
        "CH4": 0.011
    }
    
    Mw = {
        "CO": 28,
        "H2O": 18,
        "CO2": 44,
        "H2": 2,
        "Ar": 40,
        "N2": 28,
        "CH4": 16
    }
    
    # kmol/hr
    # Other resource
    # F = {
    # "CO": 266,
    # "H2O": 2712, # 2712
    # "CO2": 1239,
    # "H2": 4726,
    # "Ar": 20,
    # "N2": 21,
    # "CH4": 1624
    # }

    F = {
    "CO": 289.64,
    "H2O": 4082.96,
    "CO2": 1844.69,
    "H2": 7503.02,
    "Ar": 3.0,
    "N2": 2532.28,
    "CH4": 49.87
    }

    F["Total"] = F["CO"] + F["H2O"] + F["CO2"] + F["H2"] + F["Ar"] + F["N2"] + F["CH4"]
    # F["Total"] = F["CO"] + F["CO2"] + F["H2"]

    sat_constants = {
        "CO": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2O": np.array([29.8605, -3152.2, -7.3037, 2.4247E-09, 1.809E-06]),
        "CO2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "Ar": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "N2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "CH4": np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    }

    vaporization_constants = {
        "CO": np.array([np.nan, np.nan, np.nan]),
        "H2O": np.array([52.053, 647.13, 0.321]),
        "CO2": np.array([np.nan, np.nan, np.nan]),
        "H2": np.array([np.nan, np.nan, np.nan]),
        "Ar": np.array([np.nan, np.nan, np.nan]),
        "N2": np.array([np.nan, np.nan, np.nan]),
        "CH4": np.array([np.nan, np.nan, np.nan])
    }

    cp_constants_gas = {
        "CO": np.array([29.556, -6.5807e-5, 2.013e-5, -1.2227e-8, 2.2617e-12]),
        "H2O": np.array([33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12]),
        "CO2": np.array([27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9, -2.9872e-13]),
        "H2": np.array([25.399, 2.0178e-2, -3.8549e-5, 3.1880e-8, -8.7585e-12]),
        "Ar": np.array([20.786, 0, 0, 0, 0]),
        "N2": np.array([29.342, -3.5395e-3, 1.0076e-5, -4.3116e-9, 2.5935e-13]),
        "CH4": np.array([34.942, -3.9957e-2, 1.9184e-4, -1.5303e-7, 3.9321e-11])
    }
    
    cp_constants_liq = {
        "CO": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2O": np.array([92.053, -3.9953e-2, -2.1103e-4, 5.3469e-7, 0]),
        "CO2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "Ar": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "N2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "CH4": np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    }
    
    # dynamic viscosity
    # kinematic viscosity = dynamic viscosity / density
    # micropose = 10^-7 Pa*s (kg/ms)
    # A + BT + CT^2
    nu_gas = {
        "CO": np.array([35.086, 5.0651e-1, -1.3314e-4]),
        "H2O": np.array([-36.826, 4.29e-1, -1.62e-5]),
        "CO2": np.array([11.336, 4.9918e-1, -1.0876e-4]),
        "H2": np.array([27.758, 2.12e-1, -3.28e-5]),
        "Ar": np.array([44.997, 6.3892e-1, -1.2455e-4]),
        "N2": np.array([42.606, 4.75e-1, -9.88e-5]),
        "CH4": np.array([3.844, 4.0112e-1, -1.4303e-4])
    }


    # ----- Reactor Properties ----- 
    L_D_ratio = 0.67

    # ----- Catalyst Properties -----

    # conversion from 1 kg/m^3 to lb / ft^3
    conversion = 2.20462 / 3.28084 ** 3

    # lb / ft^3
    particle_density = 155
    # lb / ft^3
    bulk_density = 90
    
    porosity = (1 - bulk_density / particle_density)

    # Catalyst dimensions 1/4 x 1/8 in
    # converted to mm
    diameter = 1/4 / 39.3701 * 1000
    length = 1/8 / 39.3701 * 1000

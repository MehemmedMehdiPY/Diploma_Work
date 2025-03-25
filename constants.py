import numpy as np

class Data:
    """Constant parameters and steady variables"""
    Tc = {
        "CO": 132.92,
        "H2O": 647.13,
        "CO2": 304.19,
        "H2": 33.18
    }

    Pc = {
        "CO": 34.99 * 10 ** 5,
        "H2O": 220.55 * 10 ** 5,
        "CO2": 73.82 * 10 ** 5,
        "H2": 13.13 * 10 ** 5
    }

    w = {
        "CO": 0.066,
        "H2O": 0.345,
        "CO2": 0.228,
        "H2": -0.22
    }

    F = {
    # "CO": 289.64,
    # "H2O": 4082.96,
    # "CO2": 1844.69,
    # "H2": 7503.02,
    "CO": 266,
    "H2O": 2712,
    "CO2": 1239,
    "H2": 4726,
    "Ar": 20,
    "N2": 21,
    "CH4": 1624
    }
    F["Total"] = F["CO"] + F["H2O"] + F["CO2"] + F["H2"]
    # F["Total"] = F["CO"] + F["CO2"] + F["H2"]

    cp_constants_gas = {
        "CO": np.array([29.556, -6.5807e-5, 2.013e-5, -1.2227e-8, 2.2617e-12]),
        "H2O": np.array([33.933, -8.4186e-3, 2.9906e-5, -1.7825e-8, 3.6934e-12]),
        "CO2": np.array([27.437, 4.2315e-2, -1.9555e-5, 3.9968e-9, -2.9872e-13]),
        "H2": np.array([25.399, 2.0178e-2, -3.8549e-5, 3.1880e-8, -8.7585e-12])
    }
    
    cp_constants_liq = {
        "CO": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2O": np.array([92.053, -3.9953e-2, -2.1103e-4, 5.3469e-7, 0]),
        "CO2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
        "H2": np.array([np.nan, np.nan, np.nan, np.nan, np.nan])
    }

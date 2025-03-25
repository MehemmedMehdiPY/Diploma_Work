from models import EnthalpyReaction, KineticModels
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, simpson

def get_volume():
    X = np.linspace(0, 0.95, 5000)
    T = 478
    P = 34.2e5
    model = KineticModels(X_objective=0.9)
    bulk_density = 1173.1 * 2.20462 / 3.28084 ** 3
    
    rate = model(T, P, X)
    # print((rate < 0).sum())
    weight = simpson(y=(- 1 / rate), x=X) * model.F[0] * 1000 / 453.592
    print(-weight)
    print(-weight / bulk_density / 3.28084 ** 3)
    
    print(rate[-1] - rate[-2])
    plt.plot(X, rate)
    plt.show()
    
def heat_versus_T():
    T = np.linspace(298, 501, 100)
    P = 34.2e5
    model = EnthalpyReaction(X_objective=0.9)
    enthalpies = []
    for T_value in T:
        enthalpies.append(model(T=T_value, P=P)[0])
    plt.plot(T - 273.15, enthalpies)
    plt.savefig("./images/heat_versus_T.png")
    plt.show()

def heat_versus_P():
    T = 490
    P = np.linspace(1, 34.2, 100) * 10 ** 5
    print(P)
    model = EnthalpyReaction(X_objective=0.9)
    enthalpies = []
    for P_value in P:
        enthalpies.append(model(T=T, P=P_value)[0])
    plt.plot(P, enthalpies)
    plt.savefig("./images/heat_versus_P.png")
    plt.show()

def rate_versus_T():
    X = np.linspace(0, 0.9, 1000)
    T = 478
    P = 34.2e5
    print(T)
    model = KineticModels(X_objective=0.9)
    rate = model(T, P, X)
    print(rate.shape)

    plt.plot(X, rate)
    plt.show()

# rate_versus_T()
get_volume()
# rate_versus_T()
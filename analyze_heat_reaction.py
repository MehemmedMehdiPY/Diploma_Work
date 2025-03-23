from models import EnthalpyReaction, KineticModels
import numpy as np
import matplotlib.pyplot as plt

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
    X = np.linspace(0, 0.99, 1000)
    X = 0.9
    T = 478
    P = 34.2e5
    print(X, T, P)
    model = KineticModels()
    rate = model(X, T, P)
    # plt.plot(X, rate)
    # plt.show()

rate_versus_T()
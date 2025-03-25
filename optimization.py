from models import Optimization
import numpy as np

def constraints(T, P):
    return 400 - T, 20 * 10 ** 5 - P
    
def cost_func(T, P, volume):
    r = 1
    barrier_sum = 0
    # for g in constraints(T, P):
    #     if g >= 0:
    #         return 1e10
    #     barrier_sum -= r * np.log(-g)
    cost = (
        T * 182 + P * 1029 + volume * 120 + barrier_sum
        )
    return cost

T = 478
P = 34.2 * 10 ** 5
model = Optimization(X_objective=0.90, cost_func=cost_func)

T, P = model(T=T, P=P)
# print(cost_func(T, P, model.get_volume(T, P)))
print(T, P)
import sys
sys.path.append("../")
from optimization import Solution

solver = Solution()
X_objective = 0.9
ld_ratio = 3.0
T = 455.4792
P = 10.0e5
n_samples = 1000
results = solver(T=T, P=P, X_objective=X_objective, ld_ratio=ld_ratio, n_samples=n_samples)
outlet_temp = results["final_temperature"]
catalyst_weight = results["weight"]
reactor_volume = results["volume"]

print("Outlet temperature: {} K".format(outlet_temp))
print("Catalyst weight: {} kg".format(catalyst_weight))
print("Reactor volume: {} m^3".format(reactor_volume))
import cyipopt
import numpy as np

class HS071(cyipopt.Problem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        return x[0] ** 2 - 2 * x[1]

    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        return np.array([
            2 * x[0],
            -2
        ])

    def constraints(self, x):
        """Returns the constraint values."""
        return np.array(
            [
                [x[0] - 5], # x1 - 5 >= 0
                [x[0] * x[1]]  # x2 + 3 >= 0
            ])  

    def jacobian(self, x):
        """Returns the Jacobian of the constraints."""
        return np.array([
            [1, 0],
            [x[1], x[0]]
            ])  # [dc/dx1, dc/dx2]

# # Define bounds and constraint limits
# lb = [-10, -10]    # x1 >= 5, x2 >= -30
# ub = [30, 30]    # x1 <= 30, x2 <= 30
# cl = [0, 0]         # c(x) >= 0 (i.e., x1 - 5 >= 0)
# cu = [50, 100]       # c(x) <= 100 (redundant with ub, but kept)

# # Initial guess (feasible)
# x0 = [0, 0]  # x1 = 6 > 5, x2 = 5

# # Instantiate the problem
# nlp = HS071(
#     n=len(x0),    # Number of variables
#     m=len(cl),    # Number of constraints
#     lb=lb,
#     ub=ub,
#     cl=cl,
#     cu=cu
# )
# nlp.add_option('tol', 1e-16)

# # Solve
# x, info = nlp.solve(x0)

# # Print results
# print("Solver status:", info['status_msg'])
# print("Solution:", x)
# print("Objective value:", nlp.objective(x))
# print("Constraint value:", nlp.constraints(x))
# print("Constraint satisfied:", x[0] >= 5, x[1] >= -3)
# print("Bounds satisfied:", all(lb[i] <= x[i] <= ub[i] for i in range(len(x))))

import jax.numpy as jnp

def tac(x):
    """
    D - Diameter
    L - Length
    P - Pressure
    T_in - Inlet temperature
    T_out - Outlet temperature
    F_CO - CO flow rate
    F_H2O - H2O flow rate
    DELTA_H_RXN - Heat of reaction through reactor
    C_P - Heat capacity of mixture
    W_cat - Weight of catalyst
    """
    # Unpack decision variables
    D, L, P, F_CO, F_H2O, W_cat, delta_P = x  # [m, m, K, Pa, mol/s, mol/s]
    
    # Constants (dummy values, adjust as needed)
    HOURS_PER_YEAR = 8760 * 3600  # 24-hour operation
    X_CO = 0.9  # Assumed CO conversion
    F_H2 = F_CO * X_CO  # H2 production (simplified)
    F_total = F_CO + F_H2O  # Total flow, no inerts
    
    # Pseudo-calculations
    # Q = F_total * C_P * (T_out - T_in) + DELTA_H_RXN * F_CO * X_CO  # Heat duty (J/s)
    Q = 0
    
    # Cost coefficients (dummy, $/unit)
    a1, a2, a3, a4, a5, a6, a7 = 100, 500, 0.1, 0.05, 1e-6, 0.01, 1.7e3
    
    # TAC components ($/year)
    C_vessel = a1 * D * L * P**0.5  # CAPEX: Vessel cost
    C_cat = a2 * W_cat  # CAPEX + OPEX: Catalyst cost
    C_feed = a3 * (F_CO + F_H2O) * HOURS_PER_YEAR  # OPEX: Feed cost
    C_product = -a4 * F_H2 * HOURS_PER_YEAR  # OPEX: H2 revenue
    C_heat = a5 * jnp.abs(Q) * HOURS_PER_YEAR  # OPEX: Heating/cooling
    C_comp = a6 * F_total * P**0.3 * HOURS_PER_YEAR  # OPEX: Compression
    C_deltaP = a7 * F_total * delta_P * HOURS_PER_YEAR  # OPEX: Pressure drop
    
    # Total annualized cost ($/year)
    return (C_vessel + C_cat + C_feed + C_product + C_heat + C_comp + C_deltaP) / 10 ** 6 / 10 ** 6

# Example usage (for testing, optional)
if __name__ == "__main__":
    x0 = jnp.array([0.1, 1.0, 473.0, 5e6, 1.0, 2.0])  # [D, L, T_in, P, F_CO, F_H2O]
    x0 = jnp.array([1, 5, 34.2e5, 266 * 1000 / 3600, 2000 * 1000 / 3600, 4e4, 3000])
    cost = tac(x0)
    print(f"TAC: ${cost:.2f}/year")


quit()
from jax import config

# Enable 64 bit floating point precision
config.update("jax_enable_x64", True)

# We use the CPU instead of GPU und mute all warnings if no GPU/TPU is found.
config.update('jax_platform_name', 'cpu')

import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev
from cyipopt import minimize_ipopt

def objective(x):
    return x[0]*x[3]*np.sum(x[:3]) + x[2]

# def eq_constraints(x):
#     return np.sum(x**2) - 40

def ineq_constrains(x):
    return np.prod(x) - 25

def ineq_constrains_2(x):
    return np.exp(x[0]) - x[1]

obj_jit = jit(objective)
# con_eq_jit = jit(eq_constraints)
con_ineq_jit = jit(ineq_constrains)

con_ineq_jit_2 = jit(ineq_constrains_2)

# build the derivatives and jit them
obj_grad = jit(grad(obj_jit))  # objective gradient
obj_hess = jit(jacrev(jacfwd(obj_jit))) # objective hessian

con_ineq_jac = jit(jacfwd(con_ineq_jit))  # jacobian
con_ineq_jac_2 = jit(jacfwd(con_ineq_jit_2))

con_ineq_hess = jacrev(jacfwd(con_ineq_jit))  # hessian
con_ineq_hessvp = jit(lambda x, v: con_ineq_hess(x) * v[0]) # hessian vector-product

con_ineq_hess_2 = jacrev(jacfwd(con_ineq_jit_2))
con_ineq_hessvp_2 = jit(lambda x, v: con_ineq_hess_2(x) * v[0])

# constraints
cons = [
    {'type': 'ineq', 'fun': con_ineq_jit, 'jac': con_ineq_jac, 'hess': con_ineq_hessvp},
    {'type': 'ineq', 'fun': con_ineq_jit_2, 'jac': con_ineq_jac_2, 'hess': con_ineq_hessvp_2}
 ]

# starting point
x0 = np.array([1.0, 5.0, 5.0, 1.0])

# variable bounds: 1 <= x[i] <= 5
bnds = [(1, 5) for _ in range(x0.size)]

# executing the solver
res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                  constraints=cons, options={'disp': 5})
print(res)
print(ineq_constrains(res.x))
print(ineq_constrains_2(res.x))
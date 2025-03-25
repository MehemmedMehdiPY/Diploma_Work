import numpy as np
from scipy.optimize import minimize

# Define the function f(x1, x2) to be set to zero
def f(x):
    x1, x2 = x
    return x1**2 + 2 * x1 * x2 + x2**2

# Define inequality constraints as g_i(x) < 0
# For x1 > 2: g1(x) = 2 - x1 < 0
# For x2 > 5: g2(x) = 5 - x2 < 0
def constraints(x):
    x1, x2 = x
    return [2 - x1, 5 - x2]  # List of g_i(x)

# Augmented objective function: penalty for f(x) = 0 + barrier for constraints
def objective(x, r):
    penalty = f(x)**2  # Penalize deviation from f = 0
    barrier_sum = 0
    for g in constraints(x):
        if g >= 0:  # If constraint is violated, return a large penalty
            return 1e10
        barrier_sum -= r * np.log(-g)  # Log barrier for g < 0
    return penalty + barrier_sum

# Gradient of the objective (optional, for better convergence)
def gradient(x, r):
    x1, x2 = x
    # Gradient of f(x)^2
    df_dx1 = 2 * f(x) * (2 * x1 + 2 * x2)
    df_dx2 = 2 * f(x) * (2 * x1 + 2 * x2)
    # Gradient of barrier terms
    g1, g2 = constraints(x)
    if g1 >= 0 or g2 >= 0:
        return np.array([1e5, 1e5])  # Large gradient if infeasible
    dg1_dx1 = -1
    dg2_dx2 = -1
    barrier_grad_x1 = -r * (-dg1_dx1) / g1  # From -r * ln(-g1)
    barrier_grad_x2 = -r * (-dg2_dx2) / g2  # From -r * ln(-g2)
    return np.array([df_dx1 + barrier_grad_x1, df_dx2 + barrier_grad_x2])

# Sequential Unconstrained Minimization Technique (SUMT)
def solve_constrained_optimization(x0, r0=1.0, eta=0.5, tol=1e-6, max_iter=20):
    x = np.array(x0)
    r = r0
    for k in range(max_iter):
        # Define the objective with current r
        obj = lambda x: objective(x, r)
        grad = lambda x: gradient(x, r)
        
        # Minimize using BFGS (or SLSQP if constraints are added directly)
        result = minimize(obj, x, method='BFGS', jac=grad, tol=1e-8,
                         options={'disp': False})
        x_new = result.x
        
        # Check convergence
        if np.linalg.norm(x_new - x) < tol and abs(f(x_new)) < tol:
            print(f"Converged after {k+1} iterations")
            break
        
        x = x_new
        r *= eta  # Decrease r (barrier parameter)
        print(f"Iteration {k+1}: x = {x}, f(x) = {f(x)}, r = {r}")
    
    return x

# Test the implementation
x0 = [3, 6]  # Initial guess satisfying x1 > 2, x2 > 5
solution = solve_constrained_optimization(x0)
print(f"Final solution: x1 = {solution[0]}, x2 = {solution[1]}")
print(f"f(x) at solution: {f(solution)}")
print(f"Constraints satisfied: x1 > 2: {solution[0] > 2}, x2 > 5: {solution[1] > 5}")

# Verify the analytical solution trend
print(f"Analytical check: x1 + x2 = {solution[0] + solution[1]} (should be 0)")


# import numpy as np
# from scipy.optimize import minimize, basinhopping

# # Define y(x)
# def y(x1, x2):
#     return (x1**2 * np.exp(-x1) + np.cos(x1) - x2 * x1 ) ** 2

# # Grid Search
# x1 = np.linspace(0, 10, 100)
# x2 = np.linspace(0, 10, 100)
# y_vals = y(x1, x2)
# x1 = x1[np.argmin(y_vals)]
# x2 = x2[np.argmin(y_vals)]

# # Gradient Descent (approximate derivative)
# def grad_x1(x1, x2, h=1e-6):
#     return (y(x1 + h, x2) - y(x1, x2)) / h

# def grad_x2(x1, x2, h=1e-6):
#     return (y(x1, x2 + h) - y(x1, x2)) / h

# alpha = 0.01
# for _ in range(1000):
#     x1 -= alpha * grad_x1(x1, x2)
#     x2 -= alpha * grad_x2(x1, x2)
#     print(y(x1, x2))
# print(f"Gradient Descent: x1 = {x1:.4f} and x2 = {x2:.4f}, y = {y(x1, x2):.4f}")

# # Newton’s Method via minimize
# # result = minimize(y, x_initial, method='Newton-CG', jac=grad_y)
# # print(f"Newton: x = {result.x[0]:.4f}, y = {result.fun:.4f}")
# # 
# # Simulated Annealing
# # result = basinhopping(y, x_initial, niter=100)
# # print(f"Simulated Annealing: x = {result.x[0]:.4f}, y = {result.fun:.4f}")
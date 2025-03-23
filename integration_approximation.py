def Simpson(f, a, b, n):
    """Approximate the integral of f(x) from a to b using Simpson's rule."""
    if n % 2 == 1:
        raise ValueError("Simpson's rule requires an even number of subintervals (n).")
    
    h = (b - a) / n  # Step size
    integral = f(a) + f(b)  # First and last terms

    # Sum terms with coefficients 4 (odd indices)
    for i in range(1, n, 2):
        x = a + i * h
        integral += 4 * f(x)
    
    # Sum terms with coefficients 2 (even indices)
    for i in range(2, n-1, 2):
        x = a + i * h
        integral += 2 * f(x)
        
    integral *= h / 3  # Multiply by h/3
    return -integral

def Trapezoidal(f, a, b, n):
    """
    Approximate the integral of f(x) from a to b using the Trapezoidal Rule.
    
    Parameters:
        f: function to integrate
        a: lower limit of integration
        b: upper limit of integration
        n: number of subintervals (trapezoids)
    
    Returns:
        Approximate integral value.
    """
    h = (b - a) / n  # Width of each subinterval
    result = (f(a) + f(b)) / 2  # First and last terms in the sum

    for i in range(1, n):
        x = a + i * h
        result += f(x)
    
    return -result * h

if __name__ == "__main__":
    # Define the function to integrate
    def func(x):
        return 2*x  # Example: Gaussian function

    # Integration limits and number of trapezoids
    a = 2
    b = 5
    n = 1000

    # Compute the integral
    integral_value = Trapezoidal(func, a, b, n)
    print(f"Approximate integral: {integral_value:.6f}")
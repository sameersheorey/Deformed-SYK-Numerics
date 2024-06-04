from numpy import *

def rungekutta4(f, alpha, a, b, N, F):
    """
    Args:
        f (function): The function defining the system of ODEs. It should accept two arguments: the independent variable and the dependent variable(s).
        alpha (numpy.ndarray): The initial condition(s) for the dependent variable(s).
        a (float): The initial value of the independent variable.
        b (float): The final value of the independent variable.
        N (int): The number of steps to take from a to b.
        F (function): Intended to be Fnthorder, defined below, which converts a higher-order ODE into a system of first-order ODEs. 
    Returns:
        tuple: A tuple containing:
            - xs (numpy.ndarray): The array of independent variable values.
            - ys (numpy.ndarray): The array of dependent variable values, with each row corresponding to the dependent variables at a particular value of the independent variable.
    """
    h = (b - a) / N
    n = alpha.size
    xs = arange(a, b + h / 2, h)
    ys = zeros((N + 1, n))
    ys[0] = alpha  # starting point for the recursion
    for i in range(0, N):
        xi = a + i * h
        K0 = h * F(f, xi, ys[i])
        K1 = h * F(f, xi + h / 2, ys[i] + K0 / 2)
        K2 = h * F(f, xi + h / 2, ys[i] + K1 / 2)
        K3 = h * F(f, xi + h, ys[i] + K2)
        ys[i + 1] = ys[i] + 1 / 6 * (K0 + 2 * K1 + 2 * K2 + K3)
    return xs, ys


def rungekutta4_Legendre(f, alpha, a, b, N, F, LGLpoints):
    """
    Args:
        f (function): The function defining the system of ODEs. It should accept two arguments: the independent variable and the dependent variable(s).
        alpha (numpy.ndarray): The initial condition(s) for the dependent variable(s).
        a (float): The initial value of the independent variable.
        b (float): The final value of the independent variable.
        N (int): The number of steps to take from a to b.
        F (function): Intended to be Fnthorder, defined below, which converts a higher-order ODE into a system of first-order ODEs.
        LGLpoints (numpy.ndarray): Array of Legendre-Gauss-Lobatto points for the independent variable.

    Returns:
        tuple: A tuple containing:
            - xs (numpy.ndarray): The array of Legendre-Gauss-Lobatto points.
            - ys (numpy.ndarray): The array of dependent variable values, with each row corresponding to the dependent variables at a particular value of the independent variable.
    """
    xs = LGLpoints
    n = alpha.size
    ys = zeros((N + 1, n))
    ys[0] = alpha  # starting point for the recursion
    for i in range(0, N):
        # print(i)
        h = xs[i+1]-xs[i]
        xi = a + i * h
        K0 = h * F(f, xi, ys[i])
        K1 = h * F(f, xi + h / 2, ys[i] + K0 / 2)
        K2 = h * F(f, xi + h / 2, ys[i] + K1 / 2)
        K3 = h * F(f, xi + h, ys[i] + K2)
        ys[i + 1] = ys[i] + 1 / 6 * (K0 + 2 * K1 + 2 * K2 + K3)
    return xs, ys


def Fnthorder(f, x, yvec):
    """
    Converts a higher-order ODE into a system of first-order ODEs.

    Arrgs:
        f (function): The function defining the higher-order ODE.
        x (float): The current value of the independent variable.
        yvec (numpy.ndarray): The current values of the dependent variables.

    Returns:
        numpy.ndarray: The derivatives of the dependent variables.
    """
    res = zeros(yvec.size)
    for i in range(yvec.size - 1):
        res[i] = yvec[i + 1]
    res[yvec.size - 1] = f(x, yvec)
    return res


def bisection(f, a, b, bisection_iterations,tol):
    """
    Finds a root of the function using the bisection method with a specified tolerance.

    Args:
        f (function): The function for which to find the root.
        a (float): The start of the interval in which to search for the root.
        b (float): The end of the interval in which to search for the root.
        bisection_iterations (int): The number of iterations to perform.
        tol (float): The tolerance for the approximation to the root.

    Returns:
        float: The approximation to the root.
    """
    for i in range(bisection_iterations):
        # print(i)
        c = (a + b) / 2
        # print("a =", a, "b =", b, "c =", c)
        # print("[", a, ",", b, "]: f(", c, ")=", f(c))
        if abs(f(c)) < tol:
            print("zero found: ", c)
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        print("[",a,",",b,"]: f(",c,")=",f(c))
    return c
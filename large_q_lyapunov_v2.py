# Compute the Lyapunov exponent of the deformed model at large q

from matplotlib import pyplot as plt
import time
from shooting_functions import *

betaExprange = arange(10, 10.1, 0.1)

# gmrange = arange(-1, -1.5, -0.5)
s = 0.0001

exponent = dict()

for betaExp in betaExprange:
    start_time = time.time()

    print("beta = 10**x: x = ", betaExp)

    # print(gm)
    # epsilon = 0.00001

    beta = 10**betaExp
    a = 0
    b = 3*beta

    alpha = 0.01  # Can be any number but keep small to keep shooting end value small
    u0 = 0

    target = 0

    rungekutta_iterations = 10**5

    # Compute v from beta and s
    nu = Nu(beta, s)

    def G(mu):

        # We define e^g(b/2+it) for the flow
        def exp_g(nu, beta, s, t):        
            return (4 * nu**4) / ((s**2 * beta**2 + sqrt(beta**2 * (s**4 * beta**2 + nu**2)) * cosh((2 * t * nu) / beta))**2)
        
        # We define e^(g(b/2+it)/2) for the flow
        def exp_g_2(nu, beta, s, t):        
            return (2 * nu**2) / ((s**2 * beta**2 + sqrt(beta**2 * (s**4 * beta**2 + nu**2)) * cosh((2 * t * nu) / beta)))

        # We define mu = (l*beta)/(2*pi) so in f l**2 is replaced by ((2*pi*mu)/(beta)) ** 2)

        def f(x, yvec):
            # return 2*exp(yvec[0])
            return (((2*pi*mu)/(beta)) ** 2)/4 * yvec[0] - 2* (exp_g(nu, beta, s, x)+ s**2 * exp_g_2(nu, beta, s, x)) * yvec[0]

        xs, ys = rungekutta4(f, array([alpha, u0]), a, b, rungekutta_iterations, Fnthorder)
        return ys[-1, 0] - target  # ys[n,i] where n indentifies the position, and i identifies y or y'

    bisection_lower = 0.5
    bisection_upper = 1.05
    bisection_iterations = 12
    tol = 10**-10

    mu_out = bisection(G, bisection_lower, bisection_upper, bisection_iterations, tol)
    print("mu = ", mu_out)

    end_time = time.time()
    print("time (mins) = ", round((end_time-start_time)/60, 2))

    # For the plot can comment out the following lines
    # -------------------------------------------------
    # We define e^g(b/2+it) for the flow
    def exp_g(nu, beta, s, t):        
        return (4 * nu**4) / ((s**2 * beta**2 + sqrt(beta**2 * (s**4 * beta**2 + nu**2)) * cosh((2 * t * nu) / beta))**2)
    
    # We define e^(g(b/2+it)/2) for the flow
    def exp_g_2(nu, beta, s, t):        
        return (2 * nu**2) / ((s**2 * beta**2 + sqrt(beta**2 * (s**4 * beta**2 + nu**2)) * cosh((2 * t * nu) / beta)))

    # We define mu = (l*beta)/(2*pi) so in f l**2 is replaced by ((2*pi*mu)/(beta)) ** 2)

    def f(x, yvec):
        # return 2*exp(yvec[0])
        return (((2*pi*mu_out)/(beta)) ** 2)/4 * yvec[0] - 2* (exp_g(nu, beta, s, x)+ s**2 * exp_g_2(nu, beta, s, x)) * yvec[0]

    xs, ys = rungekutta4(f, array([alpha, u0]), a, b, rungekutta_iterations, Fnthorder)

    plt.plot(xs, ys[:, 0], label = r"$\beta\mathcal{J}$ = " + str(round(b, 1)))
    plt.show()
    # -------------------------------------------------

    exponent[beta] = mu_out

# save("exponent.npy", exponent)
# exponent = load('exponent.npy', allow_pickle='TRUE').item()

keys = exponent.keys()
values = exponent.values()

# plt.xscale("log")
# plt.scatter(keys, values)

savetxt(f"beta_range_s_{s}.csv", list(keys), delimiter=",")
savetxt(f"exponents_s_{s}.csv", list(values), delimiter=",")





# Compute the Lyapunov exponent of the deformed model at large q

from matplotlib import pyplot as plt
import time
from shooting_functions import *

gmrange = arange(-1, -17.5, -0.5)
# gmrange = arange(-1, -1.5, -0.5)
s = 0.01

exponent = dict()

for gm in gmrange:
    start_time = time.time()

    # print(gm)
    epsilon = 0.00001
    a = gm-epsilon
    b = 3*gm

    alpha = 0.01  # Can be any number but keep small to keep shooting end value small
    u0 = 0

    target = 0

    rungekutta_iterations = 10**5

    # Compute beta from gm
    thet = arccos(((-2 + 4 * exp(gm)) * s**2 + exp(2 * gm)) / (2 * s**2 + exp(gm)))
    beta = 2 * exp(-gm/2) * thet / sqrt(4 * s**2 + exp(gm))

    print("beta = 10**x: x = ", round(log(beta) / log(10), 1))

    def G(mu):

        # We define mu = (l*beta)/(2*pi) so in f l**2 is replaced by ((2*pi*mu)/(beta)) ** 2)

        def f(x, yvec):
            # return 2*exp(yvec[0])
            return ((-8 * exp(x) * (s ** 2 + exp(x)) + ((2*pi*mu)/(beta)) ** 2) * yvec[0] + 4 * exp(x) * (
                        2 * s ** 2 + exp(x)) * yvec[1]) / \
                   (4 * (exp(gm) - exp(x)) * (4 * s ** 2 + (exp(gm) + exp(x))))

        xs, ys = rungekutta4(f, array([alpha, u0]), a, b, rungekutta_iterations, Fnthorder)
        return ys[-1, 0] - target  # ys[n,i] where n indentifies the position, and i identifies y or y'

    bisection_lower = 0.5
    bisection_upper = 1.05
    bisection_iterations = 22
    tol = 10**-10

    mu_out = bisection(G, bisection_lower, bisection_upper, bisection_iterations, tol)
    print("mu = ", mu_out)

    end_time = time.time()
    print("time (mins) = ", round((end_time-start_time)/60, 2))

    def f(x, yvec):
        return ((-8 * exp(x) * (s ** 2 + exp(x)) + ((2*pi*mu_out)/(beta)) ** 2) * yvec[0] + 4 * exp(x) * (
                2 * s ** 2 + exp(x)) * yvec[1]) / \
               (4 * (exp(gm) - exp(x)) * (4 * s ** 2 + (exp(gm) + exp(x))))

    xs, ys = rungekutta4(f, array([alpha, u0]), a, b, rungekutta_iterations, Fnthorder)

    # plt.plot(xs, ys[:, 0], label = r"$\beta\mathcal{J}$ = " + str(round(b, 1)))
    # plt.show()

    exponent[beta] = mu_out

# save("exponent.npy", exponent)
# exponent = load('exponent.npy', allow_pickle='TRUE').item()

keys = exponent.keys()
values = exponent.values()

plt.xscale("log")
plt.scatter(keys, values)

# savetxt("lyapunov_outputs/beta_range.csv", list(keys), delimiter=",")
# savetxt("lyapunov_outputs/exponents.csv", list(values), delimiter=",")
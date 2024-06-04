# Compute the entropy of the deformed SYK model in the large q limit

from matplotlib import pyplot as plt
import time
from shooting_functions import *
from scipy import optimize

# alpha is a list, a is the initial point [a,b] is the interval on which we want solution
# N is the number of subintervals, so that h = (b-a)/N is the small increment

n = 2

a = 0
alpha = 0
target = 0  # here target is the gradient of g(tau) halfway
s_squared = 10**(-2)

bisection_iterations = 100

temp_range = 10 ** arange(-1.9, 0, 0.1)[::-1]
# temp_range = [10**(-9)]

bisection_mid = -2 / 3
# bisection_mid = -2.00039997
entropy = []
u0_list = []
for temp in temp_range:

    b = 1/temp

    # The larger the inverse temperature b is, the greater the number of steps needed in the 
    # Runge-Kutta method to maintain accuracy.

    if 10**9 > b >= 10**7:
        rungekutta_iterations = 10 ** 6

    elif 10**7 > b >= 10**5.9:
        rungekutta_iterations = 10**5

    elif 10**5.9 > b >= 10**3.5:
        rungekutta_iterations = 10 ** 4

    elif 10**3.5 > b >= 10 ** 1.5:
        rungekutta_iterations = 10 ** 3

    else: rungekutta_iterations = 300

    print(str(round(log(temp)/log(10), 2)), rungekutta_iterations)

    # We are only shooting to halfway along the interval [0, b]
    b = b/2

    bisection_lower = bisection_mid - 1
    bisection_upper = bisection_mid + 1

    LGLpoints = genfromtxt("LegendrePointsWeights/LGLpoints_" + str(rungekutta_iterations) + ".csv", delimiter=",")
    LGLpoints = b/2 * (LGLpoints + 1)
    LGLweights = genfromtxt("LegendrePointsWeights/LGLweights_" + str(rungekutta_iterations) + ".csv", delimiter=",")

    # Temporary fix - for 10**6 accidently created one less LGL point/weights than needed
    # TODO: recreate LGL points/weights for 10**6 and remove this fix
    if 10**9 > 2*b >= 10**7:
        rungekutta_iterations = 10 ** 6 - 1

    runtime_start = time.time()

    def f(x,yvec):
        return 2*n*s_squared*exp((1/n)*yvec[0]) + 2*exp(yvec[0]) 

    def G(u):
        xs, ys = rungekutta4_Legendre(f, array([alpha, u]), a, b, rungekutta_iterations, Fnthorder, LGLpoints)
        return ys[-1, 1] - target  # ys[n,i] where n indentifies the position, and i identifies y or y'

    u0 = bisection(G, bisection_lower, bisection_upper, bisection_iterations, 10**-10)   # solving for y(b) = target

    bisection_mid = u0

    u0_list.append(u0)

    xs, ys = rungekutta4_Legendre(f, array([alpha, u0]), a, b, rungekutta_iterations, Fnthorder, LGLpoints)

    # plt.scatter(xs, exp(ys[:, 0]), label = r"$\beta\mathcal{J}$ = " + str(round(b, 1)))
    # plt.xlabel(r"$\frac{\tau}{\beta\mathcal{J}}$", fontsize=12)
    # plt.ylabel(r"$G(\tau)$", fontsize=12)
    # plt.legend(loc="upper right")
    # plt.ylim([-0.5, 2])

    # check: y(b) = beta?
    # print(2*b, u0, ys[-1, 1])

    #---------------------------------------------
    # Entropy calculation

    # Calculate q^2*S/N - see appendix of Ads2 flow Geometries paper.
    # We multiply the integrals by 4 to account for the fact that we have only used half of g(tau) and symmetry.
    # The factor of 4 comes from two factors of 2 described below:
    # Firstly we must take beta (b) -> 2*beta (2b) since we redefined b to be b/2 in the above code
    # Secondly we must double the integral since we are only integrating with half of g(tau)


    SbyN = b / 2 * b / 8 * 4 * dot(LGLweights, (1 / 2) * (ys[:, 1] ** 2) - (2 * (n ** 2)) * s_squared * exp((1 / n) * ys[:, 0]) - 2 * exp(ys[:, 0]))

    runtime_end = time.time()

    runtime = runtime_end - runtime_start
    print(str(round(log(temp)/log(10), 2)), " time=", runtime, "SbyN=", SbyN)

    entropy.append(SbyN)

    plt.scatter(temp_range[0:len(entropy)], entropy, label=r"$s^2 = 10^-4$", facecolors='none', edgecolors='mediumblue', s=20, linewidth=2)

# SAVE OUTPUT
# savetxt("outputs/test_entropy_v2.csv", entropy, delimiter=",")
# savetxt("outputs/test_u0_v2.csv", u0_list, delimiter=",")
# savetxt("outputs/test_temp_range_v2.csv", temp_range, delimiter=",")



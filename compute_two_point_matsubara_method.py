# Calculate the SYK large N Euclidean two point function at finite T
# by numerically solving the SD equations using a recursive algorithm and the Fast Fourier Transform

import time
from functions_matsubara_method import *
import matplotlib.pyplot as plt

# set variables for run
dt = 10**-5  # Time step - must divide t0 exactly.

s_squared = 1

iteration_length = 100
# temp_range = 10 ** np.arange(-2.2, 2.01, 0.1)  # Set range of temperatures to calculate S/N for
temp_range = [0.1]

# Define initial guess for two point function: usually taken to be free theory two point function

q, q_t = 4, 2

def G_input(x):
    return 1 / 2 * np.sign(x)

for temp in temp_range:
    start_time = time.time()

    t0 = -1 / temp
    t0 = round(t0, 2)  # rounding since we want dt to divide t0

    t, G, w, S, Sf, Gf = G_mod_SD_q_tilde(t0, dt, G_input, q, q_t, s_squared,
                                      iteration_length)

    end_time = time.time()

    print("G time", end_time - start_time)


    plt.plot(t[len(G)//2:], G[len(G)//2:], color="blue")
    # plt.scatter(w[1::2], abs(Gf[1::2])/temp, color="red")




# Calculate the SYK large N Euclidean two point function at finite T
# by numerically solving the SD equations using the LGL method

from functions_LGL_method import *
import matplotlib.pyplot as plt
import time

#  ---------------------------------------------------------------------------------------------------------------------

# Note that we create the Legendre points and weights and intialise the matrices 
# L_il, D_kl and S_li for a given N before runnig this script. Ensure that you
# have run the scripts initialise_LGL_matrices.py script and compute_Legendre_points_weights.py
# and saved the relevant data for your chosen N before running this script.

N = 300  # N+1 is the number of Legendre points used

q = 4
q_t = 2
s_squared = 1

max_iters = 1000

accuracy = 10**-10

# temp_range = 10 ** np.arange(-2.2, -1.7, 0.02)  # Set range of temperatures to calculate G for

temp_range = [0.1]

#  ---------------------------------------------------------------------------------------------------------------------
G_l_initial = -np.array([1] + [0] * N)/2

LGLpoints = np.genfromtxt("LegendrePointsWeights/LGLpoints_" + str(N) + ".csv", delimiter=",")
LGLweights = np.genfromtxt("LegendrePointsWeights/LGLweights_" + str(N) + ".csv", delimiter=",")

L_il = np.genfromtxt("LGL_initialisation/L_il_"+str(N)+".csv", delimiter=",")
D_kl = np.genfromtxt("LGL_initialisation/D_kl_"+str(N)+".csv", delimiter=",")
S_li = np.genfromtxt("LGL_initialisation/S_li_"+str(N)+".csv", delimiter=",")

for temp in temp_range:

    beta = 1/temp

    start_G = time.time()
    tau, G, S, G_l, diff = G_SD_def(G_l_initial, beta, N, max_iters, accuracy, D_kl, L_il, S_li, LGLpoints, s_squared, q, q_t)

    end_G = time.time()

    print("temp: ", round(temp,4), " G time: ", end_G - start_G)

    plt.plot(tau, -G, color="red")

    temp = round(temp, 4)
    # np.savetxt(f"tau_q_{q}_qt_{q_t}_s2_{s_squared}_temp_{temp}.csv", tau, delimiter=",")
    # np.savetxt(f"G_q_{q}_qt_{q_t}_s2_{s_squared}_temp_{temp}.csv", G, delimiter=",")
    


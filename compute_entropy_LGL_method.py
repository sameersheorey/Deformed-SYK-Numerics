# Calculate the large N two point function by numerically solving the SD equations
# and compare to the analytical solution in the conformal limit (deep IR)

from functions_LGL_method import *
import matplotlib.pyplot as plt
import time

#  ---------------------------------------------------------------------------------------------------------------------

N = 300 

# Once the G and S fields are compputed we compute the entropy based on (G.238) of 1604.07818.
# For this we need S in Fourier space. To achieve this we first interpolate G with a grid of evenly spaced points 
# in time, then compute S and fourier transform. The fourier transorm can be done over smaller intervals of time
# (set by m) and then recombined to allow for finer grid in frequency space, without running into memory issues.
# To compute the entropy for low ttemperatures (bJ > 10^3) try using N=3000, Nw=10^8 and m=10.

Nw = int(10**6)  #10**8; number of matsubara frequency points
m = 1  # number of intervals that we split our frequency points into - must divide Nw exacly

if Nw % m != 0:
    raise ValueError("WARNING! m must divide Nw exactly for this method to work")

q = 4
q_t = 2
s_squared = 1

max_iters = 1000
accuracy = 10**-10

temp_range = 10 ** np.arange(-2, 1.1, 0.1)  # Set range of temperatures to calculate S/N for
# temp_range = [10**(-2)]

#  ---------------------------------------------------------------------------------------------------------------------
G_l_initial = -np.array([1] + [0] * N)/2

LGLpoints = np.genfromtxt("LegendrePointsWeights/LGLpoints_" + str(N) + ".csv", delimiter=",")
LGLweights = np.genfromtxt("LegendrePointsWeights/LGLweights_" + str(N) + ".csv", delimiter=",")

L_il = np.genfromtxt("LGL_initialisation/L_il_"+str(N)+".csv", delimiter=",")
D_kl = np.genfromtxt("LGL_initialisation/D_kl_"+str(N)+".csv", delimiter=",")
S_li = np.genfromtxt("LGL_initialisation/S_li_"+str(N)+".csv", delimiter=",")

SbyN = []

for temp in temp_range:

    beta = 1/temp

    start_G = time.time()
    tau, G, S, G_l, diff = G_SD_def(G_l_initial, beta, N, max_iters, accuracy, D_kl, L_il, S_li, LGLpoints, s_squared, q, q_t)

    end_G = time.time()

    print("G time", end_G - start_G)

    # plt.plot(tau, -G, color="red")

    start_entropy = time.time()
    entropy = compute_entropy_def(G, S, tau, m, beta, Nw, LGLweights, s_squared, q, q_t)
    end_entropy = time.time()
    print("entropy time", end_entropy - start_entropy)

    SbyN.append(entropy)

    print(s_squared, np.round(np.log10(temp), 1), entropy)

# np.savetxt("outputs/q"+str(q)+"qt"+str(q_t)+"_temp_range_s2_"+str(s_squared)+".csv", temp_range, delimiter=",")
# np.savetxt("outputs/q"+str(q)+"qt"+str(q_t)+"_SbyN_s2_"+str(s_squared)+".csv", SbyN, delimiter=",")

plt.xscale("log")
plt.scatter(temp_range, np.array(SbyN), facecolors='none', edgecolors='red', s=20, label="numerics", linewidth=2)

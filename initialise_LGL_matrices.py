from functions_LGL_method import *
import time

N = 3000

LGLpoints = np.genfromtxt("LegendrePointsWeights/LGLpoints_" + str(N) + ".csv", delimiter=",")
LGLweights = np.genfromtxt("LegendrePointsWeights/LGLweights_" + str(N) + ".csv", delimiter=",")

start_initialisation = time.time()

L_il, D_kl, S_li = Initialisation(N, LGLpoints, LGLweights)

end_initialisation = time.time()

print("initialisation time", end_initialisation - start_initialisation)

np.savetxt("Initialisation/L_il_"+str(N)+".csv", L_il, delimiter=",")
np.savetxt("Initialisation/D_kl_"+str(N)+".csv", D_kl, delimiter=",")
np.savetxt("Initialisation/S_li_"+str(N)+".csv", S_li, delimiter=",")
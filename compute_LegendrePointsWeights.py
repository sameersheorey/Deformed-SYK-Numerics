from functions_LGL_method import *
import time
from scipy.special import roots_legendre

N = 3000

start = time.time()

LGLpoints, LGLweights = roots_legendre(N + 1)

end = time.time()

print("Time", end - start)

np.savetxt("LegendrePointsWeights/LGLpoints_"+str(N)+".csv", LGLpoints, delimiter=",")
np.savetxt("LegendrePointsWeights/LGLweights_"+str(N)+".csv", LGLweights, delimiter=",")


import numpy as np
from scipy.special import eval_legendre
from scipy.fft import ifft

def Initialisation(N, LGLpoints, LGLweights):
    """
    Initialize matrices for LGL spectral method.

    Args:
        N (int): N+1 gives the degree of the Legendre polynomial basis and the number of LGL points used.
        LGLpoints (array): N+1 LGL points found as the roots of the N+1'th Legendre polynomial.
        LGLweights (array): The weights in the Legendre-Gauss-Lobatto quadrature for the N+1'th Legendre polynomial.

    Returns:
    tuple: A tuple containing three matrices:
        - L_il (ndarray): Matrix for Legendre polynomials evaluated at LGL points.
        - D_kl (ndarray): Matrix for the differentiation operator.
        - S_li (ndarray): Matrix used to transform from Euclidean time to Legendre coefficients.

    The function initializes matrices L_il, D_kl, and S_li required for LGL method computations.
    """

    # Create the matrix L_il

    L_il = np.transpose(np.array([eval_legendre(l, LGLpoints) for l in range(N + 1)]))

    # Compute matrix Dil

    D_kl = np.zeros((N + 1, N + 1))
    for k in range(N + 1):
        for l in range(k, N + 1):
            if (k + l) % 2 == 1:
                D_kl[k][l] = 2 * k + 1
    D_kl = np.array(D_kl)

    # Compute matrix S_li

    S_li = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for l in range(N + 1):
            S_li[l, i] = (2 * l + 1) / 2 * LGLweights[i] * L_il[i, l]

    return L_il, D_kl, S_li


def G_SD_def(G_l_initial, beta, N, max_iters, accuracy, D_kl, L_il, S_li, LGLpoints, s_squared, q, q_t):
    """Compute large N two point function from Schwinger Dyson equations using the LGL method - see arXiv:2206.13547v2.
    Args:
        G_l_initial: An array used as the initial guess for G_l, the coefficients of G when decomposed in Legendre polynomials.
        beta: Float denoting inverse temperature.
        N: Integer denoting number of Legendre points.
        max_iters: Integer denoting maximum number of iterations to carry out if convergence condition not satisfied.
        accuracy: Used to set convergence condition. When the difference in the absolute sum of the Legendre coefficients
            G_l between successive iterations is less than the accuracy the iteration stops.
        D_kl, L_il, S_li: Arrays containing initialised matrices needed for algorithm. These are created separately and
            saved since they only depend on the number of Legendre points and not any other model parameter.
        LGLpoints: An array containing legendre points.
        s_squared: Float defining the coupling between two SYK models.
        q: Even integer ≥2. Defines number of fermions involved in random interactions in fist SYK Hamiltonian.
        q_t: Even integer ≥2. Defines number of fermions involved in random interactions in second SYK Hamiltonian.

    Returns:
        tau: Array containing time points on which functions are evaluated on.
        G: Array containing large N two point function.
        S: Array containing large N sigma field.
        Gl: Array containing Legendre coefficients of G.
        diff: Float containing the difference between the absolute sum of all Legendre coefficients between final and
            penultimate iteration.

    """

    u = 1/2

    # Initial guess for Gl
    G_l = G_l_initial

    for iteration in range(max_iters):

        # Compute discretised G (t) and Sigma(t)

        G = np.dot(L_il, G_l)
        S = ((2**(q-1))/q) * G**(q-1) + s_squared * ((2**(q_t-1))/q_t) * G**(q_t-1)

        S_l = np.dot(S_li, S)

        # Compute convolution matrix using recursion relations

        SStar_kl = np.zeros((N+1, N+1))

        SStar_kl[0, 0] = -2 * S_l[1]/3

        for k in range(1, N):
            SStar_kl[k, 0] = 2 * (S_l[k-1]/(2 * k - 1) - S_l[k+1]/(2 * k + 3))

        SStar_kl[N, 0] = 2 * (S_l[k - 1] / (2 * k - 1))

        SStar_kl[0, 1] = -SStar_kl[1, 0]/3

        for k in range(1, N):
            SStar_kl[k, 1] = SStar_kl[k-1, 0] / (2 * k - 1) - SStar_kl[k+1, 0] / (2 * k + 3)

        SStar_kl[N, 1] = SStar_kl[k - 1, 0] / (2 * k - 1)

        for l in range(1, N):
            for k in range(l+1, N):
                SStar_kl[k, l+1] = - (2 * l + 1) / (2 * k + 3) * SStar_kl[k+1, l] + (2 * l + 1) / (2 * k - 1) * SStar_kl[k-1, l] + SStar_kl[k, l-1]
            SStar_kl[N, l + 1] = (2 * l + 1) / (2 * k - 1) * SStar_kl[k - 1, l] + SStar_kl[k, l - 1]

        for l in range(2, N + 1):
            for k in range(0, l):
                SStar_kl[k, l] = (-1) ** (l + k) * (2 * k + 1) / (2 * l + 1) * SStar_kl[l, k]

        # Solve (A8) subject to (A3)

        matrix = (D_kl[0:-1, :] + (beta**2)/4 * SStar_kl[0:-1, :])

        vector = [(-1)**l + 1 for l in range(N+1)]

        matrix = np.vstack([matrix, vector])

        G_l_old = G_l

        G_l = (1-u)*G_l + u * np.linalg.solve(matrix, [0]*(N) + [-1])

        diff_new = np.sum(np.abs(G_l - G_l_old))
        if iteration > 1:
            # print(diff-diff_new)
            # print(diff_new > diff)
            if diff_new > diff:
                u = 0.5 * u
                # print("u = " + str(u))
        diff = diff_new

        if diff < accuracy: break
        # print(iteration, diff)

    tau = [beta*(1+x)/2 for x in LGLpoints]

    return tau, G, S, G_l, diff


#------------------------------------------------------------------------------


def compute_entropy_def(G, S, tau, m, beta, Nw, LGLweights, s_squared, q, q_t):
    """
    Computes the entropy based on (G.238) of 1604.07818 (Comments on the Sachdev-Ye-Kitaev model).

    Args:
        G (numpy.ndarray): Array representing the function G(tau).
        S (numpy.ndarray): Array representing the function Sigma(tau).
        tau (numpy.ndarray): Array of time values corresponding to G and S.
        m (int): Number of divisions for the Fourier transform.
        beta (float): Inverse temperature parameter.
        Nw (int): Number of frequency points.
        LGLweights (numpy.ndarray): Array of Legendre-Gauss-Lobatto weights.
        s_squared (float): Float defining the coupling between two SYK models.
        q (int): Even integer ≥2. Defines number of fermions involved in random interactions in fist SYK Hamiltonian.
        q_t (int): Even integer ≥2. Defines number of fermions involved in random interactions in second SYK Hamiltonian.

    Returns:
        float: The computed entropy value.
    """

    b = 0

    for gamma in range(m):

        S_sum = 0

        for i in range(m):
            print("gamma = ", gamma, "i = ", i)
            t = (beta / Nw) * np.arange(i * Nw / m, (i + 1) * (Nw / m), 1)

            Gnew = np.interp(t, tau, G)

            Snew = ((2 ** (q - 1)) / q) * (-Gnew) ** (q - 1) + s_squared * ((2 ** (q_t - 1)) / q_t) * (-Gnew) ** (
                        q_t - 1)

            Snew = Snew * np.exp((1j * np.pi * t/beta * (2*gamma+1)))

            S_sum += Snew

        w = np.pi / (beta) * (2 * np.arange(gamma, Nw, m) + 1)

        w2 = np.pi / (beta) * (2 * np.arange(Nw - gamma - 1, -1, -m) + 1)

        Sf_sum = (beta / m) * ifft((S_sum))

        # plt.scatter(w, Sf_sum.real)

        b += (1 / 2) * (np.sum(np.log(1 + Sf_sum / (1j * w))) + np.sum(np.log(1 + Sf_sum / (1j * -w2))))

    a = 1 / 2 * np.log(2)

    c = beta / 2 * beta / 2 * np.dot(LGLweights, G * S - (2 ** (q - 1) / q ** 2) * G ** q - s_squared * (
                2 ** (q_t - 1) / q_t ** 2) * G ** q_t)

    result1 = (a + b - c).real

    result2 = beta * beta / 2 * np.dot(LGLweights, (2 ** (q - 1) / q ** 2) * G ** q + s_squared * (
            2 ** (q_t - 1) / q_t ** 2) * G ** q_t)

    entropy = result1 - result2

    print("a =", a, "b =", b, "c =", c, "result1 = ", result1, "result2= ", result2)
    
    return entropy

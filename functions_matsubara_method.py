import numpy as np
from scipy.fft import fft, fftfreq, ifft, rfft, irfft, fftshift, ifftshift

def G_mod_SD_q_tilde(t0, dt, G_input, q, q_t, s_squared, iteration_length):
    """Compute large N two point function from Schwinger Dyson equations.

    Solves the Schwinger Dyson equations numerically with an iterative algorithm and using the FFT
    to switch between coordinate and Fourier space,.

    Args:
        t0: A float. -t0 is the inverse temperature β of the model and sets the limits of the time interval
        on which G(t) is evaluated.
        dt: A float that sets the time step.
        G_input: A function used as the initial guess for G.
        q: Even integer ≥2. Defines number of fermions involved in random interactions in fist SYK Hamiltonian.
        q_t: Even integer ≥2. Defines number of fermions involved in random interactions in second SYK Hamiltonian.
        s_squared: Coupling between two SYK models.
        iteration_length: Integer denoting many iterations to carry out.

    Returns:
        t: Array containing time points on which functions are evaluated on.
        G: Array containing large N two point function.
        w: Array containing points in frequency space.
        S: Array containing sigma field.
        Sf: Array containing fourier transform of sigma field.
    """

    t = np.arange(t0, -t0, dt)  # define time points on which G(t) is evaluated

    # initialize G and sigma fields
    G = G_input(t)
    S = ((2**(q-1))/q) * G**(q-1) + s_squared * ((2**(q_t-1))/q_t) * G**(q_t-1)

    # Compute Fourier transform by scipy's FFT function
    Gf = fft(G)
    Gf[::2] = 0
    # frequency normalization factor is 2*np.pi/dt
    w = fftfreq(G.size) * 2 * np.pi / dt

    w = -w  # convention of sign in exponential of definition of Fourier transform
    #In order to get a discretisation of the continuous Fourier transform
    #we need to multiply g by a phase factor

    phase = 0.5 * dt * np.exp((-complex(0, 1) * t0) * w)

    Gf = Gf * phase

    # Compute Fourier transform by scipy's FFT function
    Sf = fft(S)
    Sf = Sf * phase

    a = 0.5

    for k in range(1, iteration_length):
        Gf_adjustment = np.reciprocal(-1j * w[1::2] - Sf[1::2])
        Gf_adjustment -= Gf[1::2]
        Gf_adjustment *= a
        Gf[1::2] += Gf_adjustment
        diff_new = np.sum(np.abs(Gf_adjustment))
        if k > 1:
            if diff_new > diff:
                a = 0.5 * a
        diff = diff_new

        G = ifft(Gf / phase, t.size)
        S = ((2**(q-1))/q) * G**(q-1) + s_squared * ((2**(q_t-1))/q_t) * G**(q_t-1)
        Sf = fft(S) * phase

    return t, G, w, S, Sf, Gf






import numpy as np
from scipy.linalg import expm, block_diag


def rotation_generators(s):
    """
    Construct the generators (Jx, Jy, Jz) of the rotation operators for spin 's'.
    """
    # implementation based on https://en.wikipedia.org/wiki/Spin_(physics)
    s2 = round(2*s)
    # ensure that s is a non-negative (half-)integer
    assert(s >= 0 and abs(s2/2 - s) < 1e-13)
    d = [0.5 * np.sqrt(2*i*(s + 1) - i*(i + 1)) for i in range(1, s2 + 1)]
    jx =    np.diag(d, k=-1) +    np.diag(d, k=1)
    jy = 1j*np.diag(d, k=-1) - 1j*np.diag(d, k=1)
    jz = np.diag([s - i for i in range(s2 + 1)])
    return jx, jy, jz


def rotation_operator(s, v):
    """
    Rotation operator for spin quantum number 's' and rotation axis 'v'.
    """
    j = rotation_generators(s)
    return expm(-1j*sum(v[i]*j[i] for i in range(3)))


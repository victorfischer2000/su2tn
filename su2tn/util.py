import numpy as np
from sympy.physics.quantum.cg import CG
from sympy.physics.wigner import wigner_6j


def comm(a, b):
    """
    Commutator [a, b] = a @ b - b @ a.
    """
    return a @ b - b @ a


def anticomm(a, b):
    """
    Anti-commutator {a, b} = a @ b + b @ a.
    """
    return a @ b + b @ a


def crandn(size=None, rng: np.random.Generator=None):
    """
    Draw random samples from the standard complex normal (Gaussian) distribution.
    """
    if rng is None:
        rng = np.random.default_rng()
    # 1/sqrt(2) is a normalization factor
    return (rng.normal(size=size) + 1j*rng.normal(size=size)) / np.sqrt(2)


def F_factor(jA, jB, jC, jd, je, jABC):
    """
    Return the F-factor F^(jd, je)_(ja, jb, jc, jabc)
    """
    return float(
        (-1) ** (jA + jB + jC + jABC) * np.sqrt((2 * jd + 1) * (2 * je + 1)) * wigner_6j(jA, jB, jd, jC, jABC, je)
    )


def Cfuse(ja, jb, jab):
    """
    Function returning the Fuse Clebsch-Gordan Tensor in-in-out fusing irreps ja and jb to jab.
    """
    assert any(np.isclose(np.arange(np.abs(ja - jb), ja + jb + 1), jab, atol=1e-5))
    # assert jab in np.arange(np.abs(ja - jb), ja + jb + 1)

    cfuse = np.zeros((int(2 * ja + 1), int(2 * jb + 1), int(2 * jab + 1)))
    for ma_idx, ma in enumerate(np.arange(-ja, ja + 1)):
        for mb_idx, mb in enumerate(np.arange(-jb, jb + 1)):
            for mab_idx, mab in enumerate(np.arange(-jab, jab + 1)):
                cfuse[ma_idx, mb_idx, mab_idx] = CG(j1=ja, m1=ma, j2=jb, m2=mb, j3=jab, m3=mab).doit()

    return cfuse


def Csplit(jab, ja, jb):
    """
    Function returning the Split Clebsch-Gordan Tensor in-out-out splitting irrep jab into irreps ja and jb.
    """
    assert any(np.isclose(np.arange(np.abs(ja - jb), ja + jb + 1), jab, atol=1e-5))
    # assert jab in np.arange(np.abs(ja - jb), ja + jb + 1)

    csplit = np.zeros((int(2 * jab + 1), int(2 * ja + 1), int(2 * jb + 1)))
    for mab_idx, mab in enumerate(np.arange(-jab, jab + 1)):
        for ma_idx, ma in enumerate(np.arange(-ja, ja + 1)):
            for mb_idx, mb in enumerate(np.arange(-jb, jb + 1)):
                csplit[mab_idx, ma_idx, mb_idx] = np.conj(CG(j1=ja, m1=ma, j2=jb, m2=mb, j3=jab, m3=mab).doit())

    return csplit


def clebsch_gordan_charge_sectors(j1_list, j2_list):
    """
    Calculate all possible charge sectors for the fused charge j12 from the two incoming charges j1 and j2, which can
    take the values in j1_list and j2_list
    """
    charge_sectors = []
    for j1 in j1_list:
        for j2 in j2_list:
            for j12 in np.arange(np.abs(j1 - j2), j1 + j2 + 1):
                charge_sectors.append([j1, j2, j12])

    return np.array(charge_sectors)


def calculate_possible_j12_values(j1_list, j2_list):
    """
    Calculate all possible j3 values for the fused charge j12 from the two incoming charges j1 and j2, which can
    take the values in j1_list and j2_list
    """
    j12_set = set()
    for j1 in j1_list:
        for j2 in j2_list:
            for j12 in np.arange(np.abs(j1 - j2), j1 + j2 + 1):
                j12_set.add(j12)

    return list(j12_set)


def CUP(j):
    """
    Returns the cup tensor for a given irrep value j.
    """
    return np.sqrt(2*j+1) * Cfuse(j, j, 0)


def CAP(j):
    """
    Returns the cap tensor for a given irrep value j.
    """
    return np.sqrt(2*j+1) * Csplit(0, j, j)

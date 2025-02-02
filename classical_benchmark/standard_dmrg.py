import numpy as np
from su2tn.classical_benchmark.standard_MPO import MPO
from su2tn.classical_benchmark.standard_MPS import (MPS, is_left_orthonormal, is_right_orthonormal,split_mps_tensor,
                                                    local_orthonormalize_right_qr, local_orthonormalize_left_qr)
from su2tn.classical_benchmark.standard_lanczos_method import eigh_krylov


def contract_left_block(A, W, L):
    """
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

      _________           _____
     /         \         /     \
     |        0|---   ---|1 A 2|---
     |         |         \__0__/
     |         |            |
     |         |
     |         |          __|__
     |         |         /  1  \
     |    L   1|---   ---|2 W 3|---
     |         |         \__0__/
     |         |            |
     |         |
     |         |          __|__
     |         |         /  0  \
     |        2|---   ---|1 A*2|---
     \_________/         \_____/
    """

    assert A.ndim == 3
    assert W.ndim == 4
    assert L.ndim == 3

    # multiply with conjugated A tensor
    T = np.tensordot(L, A.conj(), axes=(2, 1))

    # multiply with W tensor
    T = np.tensordot(W, T, axes=((0, 2), (2, 1)))

    # multiply with A tensor
    Lnext = np.tensordot(A, T, axes=((0, 1), (0, 2)))

    return Lnext


def contract_right_block(A, W, R):
    """
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           _________
         /     \         /         \
      ---|1 A 2|---   ---|0        |
         \__0__/         |         |
            |            |         |
                         |         |
          __|__          |         |
         /  1  \         |         |
      ---|2 W 3|---   ---|1   R    |
         \__0__/         |         |
            |            |         |
                         |         |
          __|__          |         |
         /  0  \         |         |
      ---|1 A*2|---   ---|2        |
         \_____/         \_________/
    """

    assert A.ndim == 3
    assert W.ndim == 4
    assert R.ndim == 3

    # multiply with A tensor
    T = np.tensordot(A, R, 1)

    # multiply with W tensor
    T = np.tensordot(W, T, axes=((1, 3), (0, 2)))

    # interchange levels 0 <-> 2 in T
    T = T.transpose((2, 1, 0, 3))

    # multiply with conjugated A tensor
    Rnext = np.tensordot(T, A.conj(), axes=((2, 3), (0, 2)))

    return Rnext


def compute_right_operator_blocks(psi, op):
    """
    Compute all partial contractions from the right.
    """
    L = psi.nsites
    assert L == op.nsites
    BR = [None for _ in range(L)]
    # initialize rightmost dummy block
    BR[-1] = np.array([[[1]]], dtype=psi.dtype)
    for i in reversed(range(L-1)):
        BR[i] = contract_right_block(psi.A[i+1], op.A[i+1], BR[i+1])
    return BR


def construct_local_two_site_hamiltonian(V, W, L, R):
    """
    Construct the two-site local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor of degree 8)::

                    6       4               5        7
      _________     |       |               |        |    _________
     /         \    |       |               |        |   /         \
     |        0|---/        |               |        \---|0        |
     |         |            |               |            |         |
     |         |            |               |            |         |
     |         |            |               |            |         |
     |         |          __|__           __|__          |         |
     |         |         /  1  \         /  1  \         |         |
     |    L   1|---   ---|2 V 3|---   ---|2 W 3|---   ---|1   R    |
     |         |         \__0__/         \__0__/         |         |
     |         |            |               |            |         |
     |         |            |               |            |         |
     |         |            |               |            |         |
     |         |            |               |            |         |
     |        2|---\        |               |        /---|2        |
     \_________/    |       |               |       |    \_________/
                    |       |               |       |
                    2       0               1       3
    """
    # repeated indices are summed over
    return np.einsum(V, (0, 1, 2, 3), W, (4, 5, 3, 6), L, (7, 2, 8), R, (9, 6, 10), (0, 4, 8, 10, 1, 5, 7, 9))


def merge_two_MPS(A, B):
    out =  np.einsum(A, (0, 1, 2), B, (3, 2, 4), (0, 3, 1, 4))
    s = out.shape
    out = np.reshape(out, (s[0] * s[1], s[2], s[3]))
    return out


def dmrg_two_site(H: MPO, psi: MPS, numsweeps, tol=1e-5, numiter=3, abort_condition=1e-4):
    """
    Approximate the ground state MPS by left and right sweeps and local two-site optimizations.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        tol: "tolerance" for SVD truncation

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1.0]]], dtype=BR[0].dtype)

    en_min = np.zeros(numsweeps)
    en_list = []
    # Number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0
        sweep_list = []
        # sweep from left to right (rightmost two lattice sites are handled by right-to-left sweep)
        for i in range(L - 2):
            print('step: ', i, i+1)
            Hloc = construct_local_two_site_hamiltonian(H.A[i], H.A[i + 1], BL[i], BR[i + 1])
            s = Hloc.shape
            assert s[0] == s[1] == psi.local_dim
            assert s[4] == s[5] == psi.local_dim
            # reshape into a matrix
            Hloc = np.reshape(Hloc, (s[0] * s[1] * s[2] * s[3], s[4] * s[5] * s[6] * s[7]))
            # The following can be accelerated by Krylov methods and a "matrix free" application of the local Hamiltonian.
            # wloc, vloc = np.linalg.eigh(Hloc)
            Aloc = merge_two_MPS(psi.A[i], psi.A[i+1])
            vstart = np.reshape(Aloc, (Aloc.shape[0] * Aloc.shape[1] * Aloc.shape[2]))
            wloc, vloc = eigh_krylov(A=Hloc, vstart=vstart, numiter=numiter, numeig=1)
            # wloc, vloc = np.linalg.eigh(Hloc)
            # select first eigenvector corresponding to lowest energy
            en = wloc[0]
            # optimized local tensor for two sites
            Aloc = np.reshape(vloc[:, 0], (s[0] * s[1], s[2], s[3]))
            psi.A[i], psi.A[i + 1] = split_mps_tensor(Aloc, psi.local_dim, psi.local_dim, "right", tol)
            assert is_left_orthonormal(psi.A[i])
            # update the left blocks
            BL[i + 1] = contract_left_block(psi.A[i], H.A[i], BL[i])
            print(en)
            sweep_list.append(en)

        # sweep from right to left
        for i in reversed(range(L - 1)):
            print('step: ', i, i+1)
            Hloc = construct_local_two_site_hamiltonian(H.A[i], H.A[i + 1], BL[i], BR[i + 1])
            s = Hloc.shape
            assert s[0] == s[1] == psi.local_dim
            assert s[4] == s[5] == psi.local_dim
            # reshape into a matrix
            Hloc = np.reshape(Hloc, (s[0] * s[1] * s[2] * s[3], s[4] * s[5] * s[6] * s[7]))
            # The following can be accelerated by Krylov methods and a "matrix free" application of the local Hamiltonian.
            Aloc = merge_two_MPS(psi.A[i], psi.A[i+1])
            vstart = np.reshape(Aloc, (Aloc.shape[0] * Aloc.shape[1] * Aloc.shape[2]))
            wloc, vloc = eigh_krylov(A=Hloc, vstart=vstart, numiter=numiter, numeig=1)
            # select first eigenvector corresponding to lowest energy
            en = wloc[0]
            # optimized local tensor for two sites
            Aloc = np.reshape(vloc[:, 0], (s[0] * s[1], s[2], s[3]))
            psi.A[i], psi.A[i + 1] = split_mps_tensor(Aloc, psi.local_dim, psi.local_dim, "left", tol)
            assert is_right_orthonormal(psi.A[i + 1])
            # update the right blocks
            BR[i] = contract_right_block(psi.A[i + 1], H.A[i + 1], BR[i + 1])
            print(en)
            sweep_list.append(en)

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _ = local_orthonormalize_right_qr(psi.A[0], np.array([[[1.0]]]))
        en_list.append(sweep_list)
        # record energy after each sweep
        en_min[n] = en

        if abort_condition is None:
            pass
        elif n >= 1 and en_min[-2] - en_min[-1] < abort_condition:
            print('Algorithm converged')
            break

        print("sweep {} completed, current energy: {}".format(n + 1, en))

    return en_min, en_list


def construct_local_one_site_hamiltonian(V, L, R):
    """
    Construct the two-site local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor of degree 8)::

                    6       4               5        7
      _________     |         |        |    _________
     /         \    |       |           |   /         \
     |        0|---/        |           \---|0        |
     |         |            |               |         |
     |         |            |               |         |
     |         |            |               |         |
     |         |          __|__             |         |
     |         |         /  1  \            |         |
     |    L   1|---   ---|2 V 3|---     ---|1   R    |
     |         |         \__0__/           |         |
     |         |            |              |         |
     |         |            |              |         |
     |         |            |              |         |
     |         |            |              |         |
     |        2|---\        |         /---|2        |
     \_________/    |       |         |    \_________/
                    |       |         |
                    2       0               1       3
    """
    # repeated indices are summed over
    return np.einsum(V, (0, 1, 2, 3), L, (4, 2, 5), R, (6, 3, 7), (0, 5, 7, 1, 4, 6))


def dmrg_one_site(H: MPO, psi: MPS, numsweeps, numiter=3, return_initial_energy=False):
    """
    Approximate the ground state MPS by left and right sweeps and local two-site optimizations.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps
        tol: "tolerance" for SVD truncation

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # right-normalize input matrix product state
    psi.orthonormalize(mode='right')

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = np.array([[[1.0]]], dtype=BR[0].dtype)

    init_en = None
    if return_initial_energy:
        H_mat = H.as_matrix()
        v_init = psi.as_vector()
        print(np.linalg.norm(v_init))
        v_init = v_init / np.linalg.norm(v_init)
        init_en = v_init @ H_mat @ v_init
        print('Initial energy ', init_en)

    en_min = np.zeros(numsweeps)
    en_list = []
    # Number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0
        sweep_list = []

        # sweep from left to right (rightmost two lattice sites are handled by right-to-left sweep)
        for i in range(L - 1):
            print('step: ', i)
            Hloc = construct_local_one_site_hamiltonian(H.A[i], BL[i], BR[i])
            s = Hloc.shape
            assert s[0] == s[3] == psi.local_dim
            # reshape into a matrix
            Hloc = np.reshape(Hloc, (s[0] * s[1] * s[2], s[3] * s[4] * s[5]))
            # The following can be accelerated by Krylov methods and a "matrix free" application of the local Hamiltonian.
            # wloc, vloc = np.linalg.eigh(Hloc)
            vstart = np.reshape(psi.A[i], (psi.A[i].shape[0] * psi.A[i].shape[1] * psi.A[i].shape[2]))
            wloc, vloc = eigh_krylov(A=Hloc, vstart=vstart, numiter=numiter, numeig=1)
            # select first eigenvector corresponding to lowest energy
            en = wloc[0]
            # optimized local tensor for two sites
            Aloc = np.reshape(vloc[:, 0], (s[0], s[1], s[2]))
            psi.A[i], psi.A[i + 1] = local_orthonormalize_left_qr(Aloc, psi.A[i + 1])
            # psi.A[i], psi.A[i + 1] = split_mps_tensor(Aloc, psi.local_dim, psi.local_dim, "right", tol)
            assert is_left_orthonormal(psi.A[i])
            # update the left blocks
            BL[i + 1] = contract_left_block(psi.A[i], H.A[i], BL[i])
            print(en)
            sweep_list.append(en)

        # sweep from right to left
        for i in reversed(range(1, L)):
            print('step: ', i)
            Hloc = construct_local_one_site_hamiltonian(H.A[i], BL[i], BR[i])
            s = Hloc.shape
            assert s[0] == s[3] == psi.local_dim
            # reshape into a matrix
            Hloc = np.reshape(Hloc, (s[0] * s[1] * s[2], s[3] * s[4] * s[5]))
            # The following can be accelerated by Krylov methods and a "matrix free" application of the local Hamiltonian.
            # wloc, vloc = np.linalg.eigh(Hloc)
            vstart = np.reshape(psi.A[i], (psi.A[i].shape[0] * psi.A[i].shape[1] * psi.A[i].shape[2]))
            wloc, vloc = eigh_krylov(A=Hloc, vstart=vstart, numiter=numiter, numeig=1)
            # select first eigenvector corresponding to lowest energy
            en = wloc[0]
            # optimized local tensor for two sites
            Aloc = np.reshape(vloc[:, 0], (s[0], s[1], s[2]))
            psi.A[i], psi.A[i - 1] = local_orthonormalize_right_qr(Aloc, psi.A[i - 1])
            assert is_right_orthonormal(psi.A[i])
            # update the right blocks
            BR[i-1] = contract_right_block(psi.A[i], H.A[i], BR[i])
            print(en)
            sweep_list.append(en)

        en_list.append(sweep_list)

        # right-normalize leftmost tensor to ensure that 'psi' is normalized
        psi.A[0], _ = local_orthonormalize_right_qr(psi.A[0], np.array([[[1.0]]]))

        # record energy after each sweep
        en_min[n] = en

        print("sweep {} completed, current energy: {}".format(n + 1, en))

    return init_en, en_min, en_list


def operator_average(op:MPO, psi:MPS):
    """
    Compute the expectation value `<psi | op | psi>`.

    Args:
        psi: wavefunction represented as MPS
        op:  operator represented as MPO

    Returns:
        complex: `<psi | op | psi>`
    """

    assert psi.nsites == op.nsites

    if psi.nsites == 0:
        return 0

    # initialize T by identity matrix
    T = np.identity(psi.A[-1].shape[2], dtype=psi.dtype)
    T = np.reshape(T, (psi.A[-1].shape[2], 1, psi.A[-1].shape[2]))

    for i in reversed(range(psi.nsites)):
        T = contract_right_block(psi.A[i], op.A[i], T)

    # T should now be a 1x1x1 tensor
    assert T.shape == (1, 1, 1)

    return T[0, 0, 0]

import numpy as np


def construct_heisenberg_mpo(L, J, pbc):
    """
    Return the MPO tensor list.
    """
    X = np.array([[0., 1.], [1.,  0.]])
    Y = np.array([[0., -1.j], [1.j, 0.]])
    Z = np.array([[1., 0.], [0., -1.]])
    I = np.identity(2)
    O = np.zeros((2, 2))

    A = np.array([[I, O, O, O, O],
                  [X, O, O, O, O],
                  [Y, O, O, O, O],
                  [Z, O, O, O, O],
                  [O, -J * X, -J * Y, -J * Z, I]])

    # flip the ordering of the virtual bond dimensions and physical dimensions
    # (D[i], D[i+1], m[i], n[i]) to (m[i], n[i], D[i], D[i+1])
    A = np.transpose(A, (2, 3, 0, 1))

    if pbc:
        pass

    else:
        return [A[:, :, 4:5, :] if i == 0 else A if i < L - 1 else A[:, :, :, 0:1] for i in range(L)]
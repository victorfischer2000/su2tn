import numpy as np


def merge_mpo_tensor_pair(A0, A1):
    """
    Merge two neighboring MPO tensors.
    """
    A = np.tensordot(A0, A1, (3, 2))
    # pair original physical dimensions of A0 and A1
    A = np.transpose(A, (0, 3, 1, 4, 2, 5))
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2]*A.shape[3], A.shape[4], A.shape[5]))
    return A


class MPO(object):
    """
    Matrix product operator (MPO) class.

    The i-th MPO tensor has dimension `[d, d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(cls, Alist):
        """
        Create a matrix product operator.
        """
        cls.A = [np.array(Aj) for Aj in Alist]
        # consistency checks
        for i in range(len(cls.A)-1):
            assert cls.A[i].ndim == 4
            assert cls.A[i].shape[3] == cls.A[i+1].shape[2]
        assert cls.A[0].shape[2] == cls.A[-1].shape[3]

    @property
    def nsites(cls):
        """Number of lattice sites."""
        return len(cls.A)

    @property
    def bond_dims(cls):
        """Virtual bond dimensions."""
        if len(cls.A) == 0:
            return []
        else:
            D = [cls.A[i].shape[2] for i in range(len(cls.A))]
            D.append(cls.A[-1].shape[3])
            return D

    def as_matrix(cls):
        """Merge all tensors to obtain the matrix representation on the full Hilbert space."""
        op = cls.A[0]
        for i in range(1, len(cls.A)):
            op = merge_mpo_tensor_pair(op, cls.A[i])
        assert op.ndim == 4
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        op = np.trace(op, axis1=2, axis2=3)
        return op


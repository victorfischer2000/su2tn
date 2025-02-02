import numpy as np
from classical_benchmark.standard_dmrg_util import crandn, split_matrix_svd


def local_orthonormalize_left_qr(A, Anext):
    """
    Left-orthonormalize a MPS tensor `A` by a QR decomposition,
    and update tensor at next site.
    """
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])), mode='reduced')
    A = np.reshape(Q, (s[0], s[1], Q.shape[1]))
    # update Anext tensor: multiply with R from left
    Anext = np.transpose(np.tensordot(R, Anext, (1, 1)), (1, 0, 2))
    return A, Anext


def local_orthonormalize_right_qr(A, Aprev):
    """
    Right-orthonormalize a MPS tensor `A` by a QR decomposition,
    and update tensor at previous site.
    """
    # flip left and right virtual bond dimensions
    A = np.transpose(A, (0, 2, 1))
    # perform QR decomposition and replace A by reshaped Q matrix
    s = A.shape
    assert len(s) == 3
    Q, R = np.linalg.qr(np.reshape(A, (s[0]*s[1], s[2])), mode='reduced')
    A = np.transpose(np.reshape(Q, (s[0], s[1], Q.shape[1])), (0, 2, 1))
    # update Aprev tensor: multiply with R from right
    Aprev = np.tensordot(Aprev, R, (2, 1))
    return A, Aprev


def merge_mps_tensor_pair(A0, A1):
    """
    Merge two neighboring MPS tensors.
    """
    A = np.tensordot(A0, A1, (2, 1))
    # pair original physical dimensions of A0 and A1
    A = A.transpose((0, 2, 1, 3))
    # combine original physical dimensions
    A = A.reshape((A.shape[0]*A.shape[1], A.shape[2], A.shape[3]))
    return A


class MPS(object):
    """
    Matrix product state (MPS) class.

    The i-th MPS tensor has dimension `[d, D[i], D[i+1]]` with `d` the physical
    dimension at each site and `D` the list of virtual bond dimensions.
    """

    def __init__(cls, d, D, fill='zero'):
        """
        Create a matrix product state.
        """
        cls.d = d
        # leading and trailing bond dimensions must agree (typically 1)
        assert D[0] == D[-1]
        if fill == 'zero':
            cls.A = [np.zeros((d, D[i], D[i+1])) for i in range(len(D)-1)]
        elif fill == 'random real':
            # random real entries
            cls.A = [np.random.normal(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        elif fill == 'random complex':
            # random complex entries
            cls.A = [crandn(size=(d, D[i], D[i+1])) / np.sqrt(d*D[i]*D[i+1]) for i in range(len(D)-1)]
        else:
            raise ValueError('fill = {} invalid.'.format(fill))

    @property
    def local_dim(cls):
        """Local (physical) dimension at each lattice site."""
        return cls.d

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
            D = [cls.A[i].shape[1] for i in range(len(cls.A))]
            D.append(cls.A[-1].shape[2])
            return D

    @property
    def dtype(cls):
        """Data type of tensor entries."""
        return cls.A[0].dtype

    def orthonormalize(cls, mode='left'):
        """Left- or right-orthonormalize the MPS using QR decompositions."""
        if len(cls.A) == 0:
            return

        if mode == 'left':
            for i in range(len(cls.A) - 1):
                cls.A[i], cls.A[i+1] = local_orthonormalize_left_qr(cls.A[i], cls.A[i+1])
            # last tensor
            cls.A[-1], T = local_orthonormalize_left_qr(cls.A[-1], np.array([[[1.0]]]))
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                cls.A[-1] = -cls.A[-1]
                nrm = -nrm
            return nrm
        elif mode == 'right':
            for i in reversed(range(1, len(cls.A))):
                cls.A[i], cls.A[i-1] = local_orthonormalize_right_qr(cls.A[i], cls.A[i-1])
            # first tensor
            cls.A[0], T = local_orthonormalize_right_qr(cls.A[0], np.array([[[1.0]]]))
            # normalization factor (real-valued since diagonal of R matrix is real)
            assert T.shape == (1, 1, 1)
            nrm = T[0, 0, 0].real
            if nrm < 0:
                # flip sign such that normalization factor is always non-negative
                cls.A[0] = -cls.A[0]
                nrm = -nrm
            return nrm
        else:
            raise ValueError('mode = {} invalid; must be "left" or "right".'.format(mode))

    def as_vector(cls):
        """Merge all tensors to obtain the vector representation on the full Hilbert space."""
        psi = cls.A[0]
        for i in range(1, len(cls.A)):
            psi = merge_mps_tensor_pair(psi, cls.A[i])
        assert psi.ndim == 3
        # contract leftmost and rightmost virtual bond (has no influence if these virtual bond dimensions are 1)
        psi = np.trace(psi, axis1=1, axis2=2)
        return psi


def split_mps_tensor(A, d0, d1, svd_distr, tol=0):
    """
    Split a MPS tensor with dimension `d0*d1 x D0 x D2` into two MPS tensors
    with dimensions `d0 x D0 x D1` and `d1 x D1 x D2`, respectively.
    """
    assert A.ndim == 3
    assert d0 * d1 == A.shape[0], 'physical dimension of MPS tensor must be equal to d0 * d1'
    # reshape as matrix and split by SVD
    A = np.transpose(np.reshape(A, (d0, d1, A.shape[1], A.shape[2])), (0, 2, 1, 3))
    s = A.shape
    A0, sigma, A1 = split_matrix_svd(A.reshape((s[0]*s[1], s[2]*s[3])), tol)
    A0.shape = (s[0], s[1], len(sigma))
    A1.shape = (len(sigma), s[2], s[3])
    # use broadcasting to distribute singular values
    if svd_distr == 'left':
        A0 = A0 * sigma
    elif svd_distr == 'right':
        A1 = A1 * sigma[:, None, None]
    elif svd_distr == 'sqrt':
        s = np.sqrt(sigma)
        A0 = A0 * s
        A1 = A1 * s[:, None, None]
    else:
        raise ValueError('svd_distr parameter must be "left", "right" or "sqrt".')
    # move physical dimension to the front
    A1 = A1.transpose((1, 0, 2))
    return A0, A1


def is_left_orthonormal(A):
    """
    Test whether a MPS tensor `A` is left-orthonormal.
    """
    s = A.shape
    assert len(s) == 3
    A = np.reshape(A, (s[0]*s[1], s[2]))
    return np.allclose(A.conj().T @ A, np.identity(s[2]))


def is_right_orthonormal(A):
    """
    Test whether a MPS tensor `A` is right-orthonormal.
    """
    # call `is_left_orthonormal` with flipped left and right virtual bond dimensions
    return is_left_orthonormal(np.transpose(A, (0, 2, 1)))



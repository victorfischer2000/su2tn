import numpy as np
import pandas as pd
from su2tn.su2_tensor import SU2Tensor
from su2tn.algorithms.scalar_product import conjugate_singe_fusionNode_tensor
from su2tn.MPS_utils.orthonormailze_util import update_listOfOpenEdges


def make_left_orthonormal(A_target: SU2Tensor, A_right: SU2Tensor, verify_orthonormality=False):
    A_target.fuse_neighboring_legs(irrepName1=-1, irrepName2=-2)
    right_cs_df = pd.DataFrame(A_right.listOfChargeSectors, columns=A_right.nameOrdering)

    for idx, charge_sector in enumerate(A_target.listOfChargeSectors):
        assert len(A_target.listOfDegeneracyTensors[idx].shape) == 2
        left_deg_tensor = A_target.listOfDegeneracyTensors[idx]

        u, s, vh = np.linalg.svd(left_deg_tensor, full_matrices=False)

        assert np.allclose(u @ np.diag(s) @ vh, left_deg_tensor)
        assert charge_sector[0] == charge_sector[1]

        A_target.listOfDegeneracyTensors[idx] = u

        restMatrix = np.diag(s) @ vh
        apply_restmatrix_to_right(restMatrix, A_right, charge_sector, right_cs_df)

        # update the listOfOpenEdges, because the shape of the degeneracy tensor changed
        new_dim = s.shape[0]
        update_listOfOpenEdges(A=A_target, irrepName=-3, irrepValue=charge_sector[1], new_dim=new_dim)
        update_listOfOpenEdges(A=A_right, irrepName=-2, irrepValue=charge_sector[1], new_dim=new_dim)

    A_target.split_leg(splitIrrepName=-1)

    if verify_orthonormality:
        verify_left_orthonormality(A_target)


def apply_restmatrix_to_right(restMatrix, A_right, charge_sector, right_cs_df):
    relevant_right_cs = right_cs_df[right_cs_df[-2] == charge_sector[1]]

    for index, row in relevant_right_cs.iterrows():
        right_deg_tensor = A_right.listOfDegeneracyTensors[index]
        A_right.listOfDegeneracyTensors[index] = np.einsum(restMatrix, (0, 1),
                                                           right_deg_tensor, (2, 1, 3), (2, 0, 3))


def verify_left_orthonormality(A_test: SU2Tensor):
    A = A_test
    A_conj = conjugate_singe_fusionNode_tensor(A=A_test)

    test = SU2Tensor.einsum(A_conj, (0, 1, 2), A, (1, 2, 3))
    out = test.return_explicit_tensor_blocks()
    for block in out.values():
        assert np.allclose(block, np.identity(n=block.shape[0]))



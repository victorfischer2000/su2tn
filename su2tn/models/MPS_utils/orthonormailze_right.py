import numpy as np
import pandas as pd
from su2tn.su2_tensor import SU2Tensor
from su2tn.algorithms.scalar_product import  conjugate_MPS
from su2tn.MPS_utils.orthonormailze_util import update_listOfOpenEdges
from copy import deepcopy


def make_right_orthonormal(A_left: SU2Tensor, A_target:SU2Tensor, verify_orthonormality=False):
    # A_target.reverse_left_most_leg(reverseIrrepName=-1)
    A_target.perform_permutation(irrep1Name=-1, irrep2Name=-2)
    A_target.reverse_right_most_leg(reverseIrrepName=-1)

    # A_target.fuse_neighboring_legs(irrepName1=-1, irrepName2=-3)
    A_target.fuse_neighboring_legs(irrepName1=-1, irrepName2=-3)

    left_cs_df = pd.DataFrame(A_left.listOfChargeSectors, columns=A_left.nameOrdering)

    for idx, charge_sector in enumerate(A_target.listOfChargeSectors):
        assert len(A_target.listOfDegeneracyTensors[idx].shape) == 2
        # right_deg_tensor = np.transpose(A_target.listOfDegeneracyTensors[idx])
        right_deg_tensor = A_target.listOfDegeneracyTensors[idx]

        u, s, vh = np.linalg.svd(right_deg_tensor, full_matrices=False)

        assert np.allclose(u @ np.diag(s) @ vh, right_deg_tensor)
        assert charge_sector[0] == charge_sector[1]

        # A_target.listOfDegeneracyTensors[idx] = np.transpose(vh)
        A_target.listOfDegeneracyTensors[idx] = vh

        restMatrix = u @ np.diag(s)
        apply_restmatrix_to_left(restMatrix, A_left, charge_sector, left_cs_df)

        # update the listOfOpenEdges, because the shape of the degeneracy tensor changed
        new_dim = s.shape[0]
        update_listOfOpenEdges(A=A_target, irrepName=-2, irrepValue=charge_sector[1], new_dim=new_dim)
        update_listOfOpenEdges(A=A_left, irrepName=-3, irrepValue=charge_sector[1], new_dim=new_dim)
        # print('new_dim', charge_sector[1], new_dim)
    # A_target.split_leg(splitIrrepName=-1)
    A_target.split_leg(splitIrrepName=-1)

    # A_target.reverse_left_most_leg(reverseIrrepName=-1)
    A_target.reverse_right_most_leg(reverseIrrepName=-1)
    A_target.perform_permutation(irrep1Name=-2, irrep2Name=-1)

    A_target = deepcopy(A_target)
    A_left = deepcopy(A_left)
    return A_left, A_target
    # print(A_target.listOfOpenEdges)
    # print(A_left.listOfOpenEdges)
    # print('\n')

    # order_in_nameOrdering(A_target)


def apply_restmatrix_to_left(restMatrix, A_left, charge_sector, left_cs_df):

    relevant_left_cs = left_cs_df[left_cs_df[-3] == charge_sector[1]]

    for index, row in relevant_left_cs.iterrows():
        left_deg_tensor = A_left.listOfDegeneracyTensors[index]
        A_left.listOfDegeneracyTensors[index] = np.einsum(left_deg_tensor, (0, 1, 2),
                                                           restMatrix, (2, 3), (0, 1, 3))


def order_in_nameOrdering(A_target):
    """
    After reversing back, the nameOrdering is [-1, -3, -2]. We change it back to [-1, -2, -3].
    """
    print('name ordering ', A_target.nameOrdering)
    assert A_target.nameOrdering == [-1, -3, -2]

    df_chargeSectors = pd.DataFrame(A_target.listOfChargeSectors, columns=A_target.nameOrdering)
    # switch the columns of the irreps 1 and 2
    newChargeSectors = df_chargeSectors[[-1, -2, -3]].values.tolist()
    A_target.listOfChargeSectors = newChargeSectors

    A_target.nameOrdering = [-1, -2, -3]
    for entry in A_target.listOfOpenEdges:
        if entry['edgeName'] == -2:
            entry['edgeNumber'] = 2
        elif entry['edgeName'] == -3:
            entry['edgeNumber'] = 3
        elif entry['edgeName'] == -1:
            entry['edgeNumber'] = 1

    for idx, entry in enumerate(A_target.listOfDegeneracyTensors):
        A_target.listOfDegeneracyTensors[idx] = np.transpose(A_target.listOfDegeneracyTensors[idx], (0, 2, 1))


def order_in_nameOrdering1(A_target):
    """
    After reversing back, the nameOrdering is [-1, -3, -2]. We change it back to [-1, -2, -3].
    """
    print('name ordering ', A_target.nameOrdering)
    assert A_target.nameOrdering == [-3, -1, -2]

    df_chargeSectors = pd.DataFrame(A_target.listOfChargeSectors, columns=A_target.nameOrdering)
    # switch the columns of the irreps 1 and 2
    newChargeSectors = df_chargeSectors[[-1, -2, -3]].values.tolist()
    A_target.listOfChargeSectors = newChargeSectors

    A_target.nameOrdering = [-1, -2, -3]
    for entry in A_target.listOfOpenEdges:
        if entry['edgeName'] == -2:
            entry['edgeNumber'] = 2
        elif entry['edgeName'] == -3:
            entry['edgeNumber'] = 3
        elif entry['edgeName'] == -1:
            entry['edgeNumber'] = 1

    for idx, entry in enumerate(A_target.listOfDegeneracyTensors):
        A_target.listOfDegeneracyTensors[idx] = np.transpose(A_target.listOfDegeneracyTensors[idx], (1, 2, 0))


def verify_right_orthonormality(A_test: SU2Tensor):
    A = A_test
    # A_conj = conjugate_singe_fusionNode_tensor(A=A_test)
    A_conj = conjugate_MPS(A=A_test)

    # print(A.fusionTree)
    # print(A_conj.fusionTree)
    deg = A.listOfDegeneracyTensors[0]
    degconj = A_conj.listOfDegeneracyTensors[0]

    print(np.round(np.einsum(deg, (0, 1, 2), degconj, (2, 0, 3)), decimals=2))

    print('________ verify _________')
    # print(np.round(np.einsum(deg, (0, 1, 2), np.conj(deg), (0, 3, 2)), decimals=2))

    print('rev A')
    A.reverse_left_most_leg(reverseIrrepName=-1)
    # print('deg after reverse left')
    deg = A.listOfDegeneracyTensors[0]
    # print(np.round(np.einsum(deg, (0, 1, 2), np.conj(deg), (0, 3, 2)), decimals=2))
    print('rev A star')
    A_conj.reverse_left_most_leg(reverseIrrepName=-2)
    degconj = A_conj.listOfDegeneracyTensors[0]
    # print(np.round(np.einsum(degconj, (0, 1, 2), np.conj(degconj), (0, 1, 3)), decimals=2))

    deg = A.listOfDegeneracyTensors[0]
    degconj = A_conj.listOfDegeneracyTensors[0]

    # print(np.round(np.einsum(deg, (0, 1, 2), degconj, (2, 0, 3)), decimals=2))

    # print(np.round(np.einsum(deg, (0, 1, 2), np.conj(deg), (0, 1, 3)), decimals=2))
    # print(A.nameOrdering)
    # print(A_conj.nameOrdering)
    # test = SU2Tensor.einsum(A, (0, 1, 2), A_conj, (2, 0, 3))

    # print(np.round(test.listOfDegeneracyTensors[0], decimals=2))
    # print(test.fusionTree)
    # out = test.return_explicit_tensor_blocks()
    # for block in out.values():
    #     print(np.round(block, decimals=2))
    #     print(block.shape)
    #     # assert np.allclose(block, np.identity(n=block.shape[0]))

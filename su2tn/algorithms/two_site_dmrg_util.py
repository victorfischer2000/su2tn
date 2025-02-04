import numpy as np
import pandas as pd
from copy import deepcopy
from su2tn.su2_tensor import SU2Tensor


def merge_two_MPS_tensors(A1, A2):
    output = SU2Tensor.einsum(A1, (0, 1, 2), A2, (3, 2, 4))
    output.perform_f_move([node for node in output.fusionTree if 1 in node])
    output.perform_permutation(irrep1Name=-3, irrep2Name=-1)
    output.fuse_neighboring_legs(irrepName1=-1, irrepName2=-3)

    fuse_info = [edge for edge in output.listOfOpenEdges if edge['edgeName'] == -1][0]['originalIrreps']
    merge_two_MPS_tensors_change_labeling(output)

    return output, fuse_info


def merge_two_MPS_tensors_change_labeling(A):
    nameOrdering = A.nameOrdering.copy()
    nameOrdering[nameOrdering.index(-4)] = -3
    cs_df = pd.DataFrame(A.listOfChargeSectors, columns=nameOrdering)
    cs_df = cs_df[[-1, -2, -3]]
    A.nameOrdering = [-1, -2, -3]
    A.listOfChargeSectors = cs_df.values.tolist()

    A.fusionTree = [[-1, -2, -3]]

    for edge in A.listOfOpenEdges:
        if edge['edgeName'] == -1:
            edge['isFused'] = False
            edge['originalIrreps'] = None
        elif edge['edgeName'] == -4:
            edge['edgeName'] = -3

    for idx, deg in enumerate(A.listOfDegeneracyTensors):
        A.listOfDegeneracyTensors[idx] = np.transpose(deg, (1, 0, 2))


def split_two_MPS_tensors_change_labeling(A, fuse_info):
    nameOrdering = A.nameOrdering.copy()
    nameOrdering[nameOrdering.index(-3)] = -4
    A.nameOrdering = nameOrdering

    A.fusionTree = [[-1, -2, -4]]

    for edge in A.listOfOpenEdges:
        if edge['edgeName'] == -1:
            edge['isFused'] = True
            edge['originalIrreps'] = fuse_info
        elif edge['edgeName'] == -3:
            edge['edgeName'] = -4

    return A


def retained_bond_indices(s, j, tol):
    """
    Indices of retained singular values based on given tolerance.
    """
    s = (s**2) * (2*j+1)
    # accumulate values from smallest to largest
    sort_idx = np.argsort(s)
    s[sort_idx] = np.cumsum(s[sort_idx])
    return np.where(s > tol)[0]


def truncate_matrix_svd(u, s, v, j, tol):
    """
    Truncate small singular values based on tolerance.
    """
    # truncate small singular values
    idx = retained_bond_indices(s, j, tol)
    u = u[:, idx]
    v = v[idx, :]
    s = s[idx]
    return u, s, v


def split_MPS_tensor(A, mode, fuse_info, tol):
    A = split_two_MPS_tensors_change_labeling(A, fuse_info)

    A.split_leg(splitIrrepName=-1)

    A.perform_permutation(irrep1Name=-1, irrep2Name=-3)

    A.perform_f_move([node for node in A.fusionTree if 1 in node])
    A.reverse_left_most_leg(reverseIrrepName=-3)

    A.fuse_neighboring_legs(irrepName1=-1, irrepName2=-2)

    A.perform_permutation(irrep1Name=-3, irrep2Name=-4)

    A.fuse_neighboring_legs(irrepName1=-3, irrepName2=-4)

    Aleft = deepcopy(A)
    Aright = deepcopy(A)

    newEdgeIrreps = []


    for idx, charge_sector in enumerate(A.listOfChargeSectors):
        assert len(A.listOfDegeneracyTensors[idx].shape) == 2

        left_deg_tensor = A.listOfDegeneracyTensors[idx]
        u, s, vh = np.linalg.svd(left_deg_tensor, full_matrices=False)
        u, s, vh = truncate_matrix_svd(u, s, vh, charge_sector[0], tol)

        newEdgeIrreps.append((charge_sector[0], np.diag(s).shape[0]))

        # assert np.allclose(u @ np.diag(s) @ vh, left_deg_tensor)
        assert charge_sector[0] == charge_sector[1]

        if mode == 'left':
            Aleft.listOfDegeneracyTensors[idx] = u
            Aright.listOfDegeneracyTensors[idx] = np.diag(s) @ vh
        elif mode == 'right':
            Aleft.listOfDegeneracyTensors[idx] = u @ np.diag(s)
            Aright.listOfDegeneracyTensors[idx] = vh
        else:
            raise ValueError('Chosen mode not supported')

    # Update the open edge entries to make sure we
    for idx_left, openEdge in enumerate(Aleft.listOfOpenEdges):
        if openEdge['edgeName'] == -3:
            openEdge['edgeIrreps'] = newEdgeIrreps
            openEdge['isFused'] = False
            openEdge['originalIrreps'] = None
            Aleft.listOfOpenEdges[idx_left] = openEdge
            break
    for idx_right, openEdge in enumerate(Aright.listOfOpenEdges):
        if openEdge['edgeName'] == -1:
            openEdge['edgeIrreps'] = newEdgeIrreps
            openEdge['isFused'] = False
            openEdge['originalIrreps'] = None
            Aright.listOfOpenEdges[idx_right] = openEdge
            break

    Aleft.split_leg(splitIrrepName=-1)

    Aright.split_leg(splitIrrepName=-3)

    Aright.reverse_right_most_leg(reverseIrrepName=-3)
    Aright.perform_permutation(irrep1Name=-1, irrep2Name=-3)

    Aleft = split_two_MPS_tensors_Aleft_change_labeling(Aleft)
    Aright = split_two_MPS_tensors_Aright_change_labeling(Aright)

    return Aleft, Aright


def verify_split_two_MPS_tensors(Ainitial, Aleft, Aright):
    Atest = SU2Tensor.einsum(Aleft, (0, 1, 2), Aright, (3, 2, 4))
    print('Atest')
    for deg in Atest.listOfDegeneracyTensors:
        print(np.round(np.real(deg), decimals=3))
    print('Ainitial')
    for deg in Ainitial.listOfDegeneracyTensors:
        print(np.round(np.real(deg), decimals=3))


def split_two_MPS_tensors_Aleft_change_labeling(A):
    df_cs = pd.DataFrame(A.listOfChargeSectors, columns=A.nameOrdering)
    A.listOfChargeSectors = df_cs[[-1, -2, -3]].values.tolist()
    A.nameOrdering = [-1, -2, -3]

    for edge in A.listOfOpenEdges:
        if edge['edgeName'] == -1:
            edge['edgeNumber'] = 1
        elif edge['edgeName'] == -2:
            edge['edgeNumber'] = 2
        elif edge['edgeName'] == -3:
            edge['edgeNumber'] = 3

    return A


def split_two_MPS_tensors_Aright_change_labeling(A):
    df_cs = pd.DataFrame(A.listOfChargeSectors, columns=A.nameOrdering)
    A.listOfChargeSectors = df_cs[[-3, -1, -4]].values.tolist()
    A.nameOrdering = [-1, -2, -3]

    A.fusionTree = [[-1, -2, -3]]

    for edge in A.listOfOpenEdges:
        if edge['edgeName'] == -3:
            edge['edgeName'] = -1
            edge['edgeNumber'] = 1
        elif edge['edgeName'] == -1:
            edge['edgeName'] = -2
            edge['edgeNumber'] = 2
        elif edge['edgeName'] == -4:
            edge['edgeName'] = -3
            edge['edgeNumber'] = 3

    for idx, deg in enumerate(A.listOfDegeneracyTensors):
        A.listOfDegeneracyTensors[idx] = np.transpose(deg, (1, 2, 0))

    return A


def merge_two_MPO_tensors(H1, H2):
    output = SU2Tensor.einsum(H1, (0, 1, 2, 3), H2, (4, 2, 5, 6))

    output.perform_f_move([node for node in output.fusionTree if 3 in node])

    output.perform_f_move([node for node in output.fusionTree if 1 in node])

    output.perform_f_move([node for node in output.fusionTree if 2 in node])

    output.fuse_neighboring_legs(irrepName1=-1, irrepName2=-4)

    output.fuse_neighboring_legs(irrepName1=-3, irrepName2=-6)

    output = merge_two_MPO_tensors_change_labeling(output)
    return output


def merge_two_MPO_tensors_change_labeling(H):
    cs_df = pd.DataFrame(H.listOfChargeSectors, columns=H.nameOrdering)
    cs_df = cs_df[[1, -1, -2, -5, -3]]
    H.listOfChargeSectors = cs_df.values.tolist()
    H.nameOrdering = [1, -1, -2, -3, -4]

    for idx, deg in enumerate(H.listOfDegeneracyTensors):
        H.listOfDegeneracyTensors[idx] = np.transpose(deg, (0, 1, 3, 2))

    for edge in H.listOfOpenEdges:
        if edge['edgeName'] == -1:
            edge['edgeName'] = -1
            edge['edgeNumber'] = 1
        elif edge['edgeName'] == -2:
            edge['edgeName'] = -2
            edge['edgeNumber'] = 2
        elif edge['edgeName'] == -3:
            edge['edgeName'] = -4
            edge['edgeNumber'] = 4
        elif edge['edgeName'] == -5:
            edge['edgeName'] = -3
            edge['edgeNumber'] = 3

    H.fusionTree = [[-1, -2, 1], [1, -3, -4]]
    H.fusionTreeDirections = [-1, +1]

    return H

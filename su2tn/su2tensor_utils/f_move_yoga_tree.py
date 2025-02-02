import numpy as np
import pandas as pd
from su2tn.util import F_factor


def f_move_yoga_build_new_fusion_tree(node1, node1Direction, node1_common_leg_idx,
                                      node2, node2Direction, node2_common_leg_idx,
                                      common_leg):
    """
    Function implementing an F-move on a yoga tree.
    """
    situation = {node1_common_leg_idx, node2_common_leg_idx}

    # check if we have a [3, 1] ({2, 0}) or [2, 3] ({1, 2}) situation
    if situation == {2, 0}:
        if node1_common_leg_idx == 0:
            storeNode = node1
            node1 = node2
            node2 = storeNode

            node1Direction, node2Direction = node2Direction, node1Direction

        if node1Direction == -1:
            raise AssertionError('F-Move would result in Yoga Tree')

        # build the new nodes
        alpha = node1[0]
        gamma = node1[1]
        beta = node2[1]
        delta = node2[2]
        new_node1 = [alpha, beta, common_leg]
        new_node2 = [common_leg, gamma, delta]

        new_node1Direction = -1
        new_node2Direction = +1
        edgeAssignment = {'A': alpha, 'B': beta, 'C': gamma, 'D': delta,
                          'commonLeg': common_leg, 'Type': (2, 0)}

    elif situation == {1, 1}:
        if node1Direction == +1:
            node1, node2 = node2, node1

        # build the new nodes
        alpha = node1[0]
        gamma = node1[2]
        beta = node2[0]
        delta = node2[2]
        new_node1 = [alpha, beta, common_leg]
        new_node2 = [common_leg, gamma, delta]

        new_node1Direction = -1
        new_node2Direction = +1

        edgeAssignment = {'A': alpha, 'B': beta, 'C': gamma, 'D': delta,
                         'commonLeg': common_leg, 'Type': (1, 1)}

    return new_node1, new_node2, new_node1Direction, new_node2Direction, edgeAssignment


def f_move_yoga_get_trafo_matrix(oldChargeSectors, newChargeSectors, nameOrdering, edgeAssignment):
    """
    Get the trafo matrix that holds the factors for recoupling the original degeneracy tensors to get the new degeneracy
    tensors, and the recoupling factors.
    """
    # names of outer edges
    A = edgeAssignment['A']
    A_idx = nameOrdering.index(A)

    B = edgeAssignment['B']
    B_idx = nameOrdering.index(B)

    C = edgeAssignment['C']
    C_idx = nameOrdering.index(C)

    D = edgeAssignment['D']
    D_idx = nameOrdering.index(D)

    # name of common leg
    commonLeg = edgeAssignment['commonLeg']
    commonLeg_idx = nameOrdering.index(commonLeg)

    trafo_matrix = np.zeros((len(oldChargeSectors), len(newChargeSectors)))

    filterIrreps = nameOrdering.copy()
    filterIrreps.remove(commonLeg)

    df_newChargeSectors_save = pd.DataFrame(newChargeSectors.copy(), columns=nameOrdering)

    for oldChargeSector_idx, oldChargeSector in enumerate(oldChargeSectors):
        jA = oldChargeSector[A_idx]
        jB = oldChargeSector[B_idx]
        jC = oldChargeSector[C_idx]
        jD = oldChargeSector[D_idx]
        jX = oldChargeSector[commonLeg_idx]

        df_newChargeSectors = df_newChargeSectors_save.copy()

        # filter for all relevant charge sectors
        for irrep in filterIrreps:
            df_newChargeSectors = df_newChargeSectors[
                df_newChargeSectors[irrep] == oldChargeSector[nameOrdering.index(irrep)]
                ]

        if df_newChargeSectors.empty:
            pass
        else:

            for newChargeSector_idx, row in df_newChargeSectors.iterrows():
                jY = list(row.array)[commonLeg_idx]

                if edgeAssignment['Type'] == (1, 1):
                    factor = F_factor(jA=jA, jB=jX, jC=jD, jABC=jY, jd=jC, je=jB)
                elif edgeAssignment['Type'] == (2, 0):
                    factor = F_factor(jA=jB, jB=jX, jC=jC, jABC=jY, jd=jD, je=jA)

                trafo_matrix[oldChargeSector_idx, newChargeSector_idx] = factor

    trafo_matrix = np.transpose(np.conjugate(trafo_matrix))
    return trafo_matrix


def f_move_yoga_calculate_new_degeneracy_tensors(trafo_matrix, newChargeSectors, oldDegeneracyTensors):
    """
    Get the new charge sectors and degeneracy tensors after the F-move.
    """
    newDegeneracyTensors = []
    potentialNewChargeSectors = newChargeSectors.copy()
    newChargeSectors = []
    for newChargeSector_idx in range(trafo_matrix.shape[0]):
        if np.allclose(trafo_matrix[newChargeSector_idx], 0):
            pass
        else:
            first_nonzero_index = np.nonzero(trafo_matrix[newChargeSector_idx])[0][0]
            newDegeneracyTensor = np.zeros(oldDegeneracyTensors[first_nonzero_index].shape, dtype='complex128')
            for oldChargeSector_idx, f_factor in enumerate(trafo_matrix[newChargeSector_idx]):
                if f_factor != 0:
                    newDegeneracyTensor = newDegeneracyTensor + f_factor * oldDegeneracyTensors[oldChargeSector_idx]

            newDegeneracyTensors.append(newDegeneracyTensor)
            newChargeSectors.append(potentialNewChargeSectors[newChargeSector_idx][1:])

    return newChargeSectors, newDegeneracyTensors


def f_move_yoga_get_new_degeneracy_tensors(oldChargeSectors, newChargeSectors, nameOrdering, edgeAssignment, oldDegeneracyTensors):
    """
    Helper function of f_move_yoga_calculate_new_degeneracy_tensors.
    """
    trafo_matrix = f_move_yoga_get_trafo_matrix(oldChargeSectors, newChargeSectors, nameOrdering, edgeAssignment)

    newChargeSectors, newDegeneracyTensors = f_move_yoga_calculate_new_degeneracy_tensors(trafo_matrix, newChargeSectors, oldDegeneracyTensors)

    return newChargeSectors, newDegeneracyTensors


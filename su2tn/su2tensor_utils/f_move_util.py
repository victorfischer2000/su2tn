import numpy as np
import pandas as pd
from sympy.physics.wigner import wigner_6j
from su2tn.su2tensor_utils.f_move_yoga_tree import f_move_yoga_build_new_fusion_tree, f_move_yoga_get_new_degeneracy_tensors


def F_factor(jA, jB, jC, jd, je, jABC):
    """
    Return the F-factor F^(jd, je)_(ja, jb, jc, jabc)
    """
    return float(
        (-1) ** (jA + jB + jC + jABC) * np.sqrt((2 * jd + 1) * (2 * je + 1)) * wigner_6j(jA, jB, jd, jC, jABC, je)
    )


def f_move_build_new_fusion_tree(fNodes, oldFusionTree, oldFusionTreeDirections):
    """
    For applying an F-move to the fNodes in oldFusionTree, build the new fusion tree.

    Comment: If i every do this with not just splitting or fusing nodes, then I need to keep track of which is the
    splitting and fusion node.
    """
    assert len(oldFusionTree) == len(oldFusionTreeDirections)
    oldFusionTree = oldFusionTree.copy()
    oldFusionTreeDirections = oldFusionTreeDirections.copy()
    # first we need to remove the fNodes and the directions of the fNodes from the fusionTree and fusionTreeDirections.
    if len(fNodes) != 2:
        raise AssertionError('F-Move: 4-index F-Move needs 2 nodes.')
    node1 = fNodes[0]
    node2 = fNodes[1]

    node1_idx = oldFusionTree.index(node1)
    oldFusionTree.pop(node1_idx)
    node1Direction = oldFusionTreeDirections.pop(node1_idx)

    node2_idx = oldFusionTree.index(node2)
    oldFusionTree.pop(node2_idx)
    node2Direction = oldFusionTreeDirections.pop(node2_idx)

    # then we determine the common leg, and we check if the given nodes have a common leg.
    common_leg = list(set(node1) & set(node2))
    if len(common_leg) != 1:
        raise AssertionError('F-Move: Given Nodes need to have exactly one common leg.')
    outer_legs = list(set(fNodes[0]) ^ set(fNodes[1]))
    if len(outer_legs) != 4:
        raise AssertionError('F-Move: Given Nodes need to have exactly four outer legs.')

    common_leg = common_leg[0]
    node1_common_leg_idx = node1.index(common_leg)
    node2_common_leg_idx = node2.index(common_leg)

    # then we need to distinguish if we have 3 incoming (and 1 outgoing) or 3 outgoing legs (and 1 incoming).
    # the case for 2 incoming and 2 outgoing is not required if we only look at simple fusion trees.
    # we have two fusing node, e.g. 3 incoming legs
    if node1Direction == -1 and node2Direction == -1:
        f_move_type = 'simple'
        # check if we have a [3, 1] ({2, 0}) or [2, 3] ({1, 2}) situation
        situation = {node1_common_leg_idx, node2_common_leg_idx}

        # make an F-dagger move according to programming guide.
        if situation == {2, 0}:
            # if node is not the node with the common leg in third position, change node1 and node2. Now node 1 has the
            # common leg in third position and node 2 in first position.
            if node1_common_leg_idx == 0:
                storeNode = node1
                node1 = node2
                node2 = storeNode

            # build the new nodes
            alpha = node1[0]
            beta = node1[1]
            gamma = node2[1]
            delta = node2[2]
            new_node1 = [alpha, common_leg, delta]
            new_node2 = [beta, gamma, common_leg]
            edgeAssignment = {'A': alpha, 'B': beta, 'C': gamma, 'ABC': delta,
                               'commonLeg': common_leg, 'Type': ('AB', 'fusion')}

        # make a F-move according programming guide
        elif situation == {1, 2}:
            # node1 should have the leg in the second position and node 2 in the third position.
            if node1_common_leg_idx == 2:
                storeNode = node1
                node1 = node2
                node2 = storeNode

            # build the new nodes
            alpha = node1[0]
            delta = node1[2]
            beta = node2[0]
            gamma = node2[1]

            new_node1 = [alpha, beta, common_leg]
            new_node2 = [common_leg, gamma, delta]
            edgeAssignment = {'A': alpha, 'B': beta, 'C': gamma, 'ABC': delta,
                               'commonLeg': common_leg, 'Type': ('BC', 'fusion')}

        else:
            raise AssertionError("F-Move: The two F-move nodes are not compatible.")

    # we have two splitting nodes, e.g. three outgoing legs
    elif node1Direction == +1 and node2Direction == +1:
        f_move_type = 'simple'
        # check if we have a [3, 1] ({2, 0}) or [2, 1] ({1, 0}) situation
        situation = {node1_common_leg_idx, node2_common_leg_idx}

        # make an F~-dagger move according to programming guide.
        if situation == {2, 0}:
            # node1 should have the leg in the second position and node 2 in the third position.
            if node1_common_leg_idx == 0:
                storeNode = node1
                node1 = node2
                node2 = storeNode

            # build the new nodes
            alpha = node1[0]
            beta = node1[1]
            gamma = node2[1]
            delta = node2[2]
            new_node1 = [alpha, common_leg, delta]
            new_node2 = [common_leg, beta, gamma]
            edgeAssignment = {'A': beta, 'B': gamma, 'C': delta, 'ABC': alpha,
                               'commonLeg': common_leg, 'Type': ('CD', 'split')}

        # make an F~-dagger move according to programming guide.
        elif situation == {1, 0}:
            # node1 should have the leg in the second position and node 2 in the first position.
            if node1_common_leg_idx == 0:
                storeNode = node1
                node1 = node2
                node2 = storeNode

            # build the new nodes
            alpha = node1[0]
            delta = node1[2]
            beta = node2[1]
            gamma = node2[2]
            new_node1 = [alpha, beta, common_leg]
            new_node2 = [common_leg, gamma, delta]
            edgeAssignment = {'A': beta, 'B': gamma, 'C': delta, 'ABC': alpha,
                               'commonLeg': common_leg, 'Type': ('BC', 'split')}

        else:
            raise AssertionError("F-Move: The two F-move nodes are not compatible.")

    elif (node1Direction == +1 and node2Direction == -1) or (node1Direction == -1 and node2Direction == +1):
        f_move_type = 'yoga'

        new_node1, new_node2, new_node1Direction, new_node2Direction, edgeAssignment = f_move_yoga_build_new_fusion_tree(
            node1=node1, node1Direction=node1Direction, node1_common_leg_idx=node1_common_leg_idx,
            node2=node2, node2Direction=node2Direction, node2_common_leg_idx=node2_common_leg_idx,
            common_leg=common_leg
        )

        node1Direction, node2Direction = new_node1Direction, new_node2Direction

    else:
        raise AttributeError('F-Move: Directions need to be either -1 or +1')

    newFusionTree = oldFusionTree + [new_node1, new_node2]
    newFusionTreeDirections = oldFusionTreeDirections + [node1Direction, node2Direction]

    return newFusionTree, newFusionTreeDirections, edgeAssignment, f_move_type


def f_move_calculate_new_degeneracy_tensor(newChargeSector, edgeAssignment, nameOrdering,
                                           oldChargeSectors, oldDegeneracyTensors):
    """
    Function calculating the new degeneracy tensor for the given new charge sector from nodes, that are affected by the
    F-Move, the old list of charge sectors and corresponding old list of degeneracy tensors.
    """
    # names of outer edges
    A = edgeAssignment['A']
    A_idx = nameOrdering.index(A)
    jA = newChargeSector[A_idx]

    B = edgeAssignment['B']
    B_idx = nameOrdering.index(B)
    jB = newChargeSector[B_idx]

    C = edgeAssignment['C']
    C_idx = nameOrdering.index(C)
    jC = newChargeSector[C_idx]

    ABC = edgeAssignment['ABC']
    ABC_idx = nameOrdering.index(ABC)
    jABC = newChargeSector[ABC_idx]

    # name of common leg
    commonLeg = edgeAssignment['commonLeg']
    commonLeg_idx = nameOrdering.index(commonLeg)
    je = newChargeSector[commonLeg_idx]

    df_oldChargeSectors = pd.DataFrame(oldChargeSectors.copy(), columns=nameOrdering)

    # Get the old charge sectors, that only differ in the values of the common leg irrep, e.g. the relevant charge
    # sectors
    filterIrreps = nameOrdering.copy()
    filterIrreps.remove(commonLeg)
    for irrep in filterIrreps:
        df_oldChargeSectors = df_oldChargeSectors[
            df_oldChargeSectors[irrep] == newChargeSector[nameOrdering.index(irrep)]
            ]

    # check if dataframe is empty
    if df_oldChargeSectors.empty:
        return None

    # build a dummy tensor to add the new parts of the degeneracy tensor to.
    first_idx = df_oldChargeSectors.index[0]
    newDegeneracyTensor = np.zeros(oldDegeneracyTensors[first_idx].shape, dtype='complex128')
    for index, row in df_oldChargeSectors.iterrows():
        jd = list(row.array)[commonLeg_idx]
        if edgeAssignment['Type'] == ('AB', 'fusion') or edgeAssignment['Type'] == ('BC', 'split'):
            factor = float(
                (-1) ** (jA + jB + jC + jABC) * np.sqrt((2 * jd + 1) * (2 * je + 1)) * wigner_6j(jA, jB, jd, jC, jABC, je)
            )

        elif edgeAssignment['Type'] == ('BC', 'fusion') or edgeAssignment['Type'] == ('CD', 'split'):
            factor = float(
                (-1) ** (jA + jB + jC + jABC) * np.sqrt((2 * jd + 1) * (2 * je + 1)) * wigner_6j(jA, jB, je, jC, jABC, jd)
            )

        newDegeneracyTensor += factor * oldDegeneracyTensors[index]

    return newDegeneracyTensor


def f_move_perform_fmove(su2tensor, fNodes):
    """
    Perform an f-move on the two nodes given in fNodes.
    """
    # make copies of the list that will be changed through the function.
    oldListOfChargeSectors = su2tensor.listOfChargeSectors.copy()
    oldListOfDegeneracyTensors = su2tensor.listOfDegeneracyTensors.copy()

    newListOfDegeneracyTensors = []

    # get the new fusion tree and its directions after the f-move.
    su2tensor.fusionTree, su2tensor.fusionTreeDirections, edgeAssignment, f_move_type = f_move_build_new_fusion_tree(
        fNodes=fNodes,
        oldFusionTree=su2tensor.fusionTree,
        oldFusionTreeDirections=su2tensor.fusionTreeDirections
    )

    # Calculate and overwrite the new charge sectors.
    su2tensor.listOfChargeSectors = su2tensor.calculate_all_charge_sectors()
    nameOrdering = [0] + su2tensor.nameOrdering
    for idx in range(len(oldListOfChargeSectors)):
        oldListOfChargeSectors[idx] = [0] + oldListOfChargeSectors[idx]

    if f_move_type == 'simple':
        newListOfChargeSectors = []
        for charge_sector in su2tensor.listOfChargeSectors:
            charge_sector = [0] + charge_sector
            newDegeneracyTensor = f_move_calculate_new_degeneracy_tensor(newChargeSector=charge_sector,
                                                                         edgeAssignment=edgeAssignment,
                                                                         nameOrdering=nameOrdering,
                                                                         oldChargeSectors=oldListOfChargeSectors,
                                                                         oldDegeneracyTensors=oldListOfDegeneracyTensors
                                                                         )
            if newDegeneracyTensor is None:
                pass
            else:
                newListOfDegeneracyTensors.append(newDegeneracyTensor)
                newListOfChargeSectors.append(charge_sector[1:])
        su2tensor.listOfDegeneracyTensors = newListOfDegeneracyTensors
        su2tensor.listOfChargeSectors = newListOfChargeSectors

    elif f_move_type == 'yoga':
        newListOfChargeSectors = su2tensor.listOfChargeSectors.copy()
        for idx in range(len(newListOfChargeSectors)):
            newListOfChargeSectors[idx] = [0] + newListOfChargeSectors[idx]

        newListOfChargeSectors, newListOfDegeneracyTensors = f_move_yoga_get_new_degeneracy_tensors(
            oldChargeSectors=oldListOfChargeSectors,
            newChargeSectors=newListOfChargeSectors,
            nameOrdering=nameOrdering,
            edgeAssignment=edgeAssignment,
            oldDegeneracyTensors=oldListOfDegeneracyTensors)

        su2tensor.listOfDegeneracyTensors = newListOfDegeneracyTensors
        su2tensor.listOfChargeSectors = newListOfChargeSectors


def f_move_perform_f_move_order(su2tensor, fMoveOrder):
    """
    Applies f-moves, defined by the internalLegs in the list fMoveOrder on the su2tensor and changes it in place.
    """
    # perform the series of f-move to the su2-tensor, which changes the su2-tensor in place.
    for internalEdge in fMoveOrder:
        # find the nodes corresponding to the edge to which the f-move will be applied.
        fNodes = [fnode for fnode in su2tensor.fusionTree if internalEdge in fnode]
        su2tensor.perform_f_move(fNodes=fNodes)

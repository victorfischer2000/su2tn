import numpy as np
import pandas as pd
from su2tn.su2tensor_utils.f_move_util import F_factor
from su2tn.su2tensor_utils.remove_unnec_dummylegs import remove_unnecessary_nodes


def reverse_left_most_leg(su2tensor, reverseLeg):
    """
    Reverse the rightmost leg of a tensor. The fusion tree has already been converted such that the leg is already
    isolated and can be reversed immediately.
    """
    # TODO: Implement checks
    (newFusionTree, newFusionTreeDirections, reverseNode, reverseNodeDirection,
     newChargeSectors, newNameOrdering, use_fFactor) = (
        left_reverse_leg_get_newFusionTree(reverseLeg=reverseLeg,
                                           oldFusionTree=su2tensor.fusionTree,
                                           oldFusionTreeDirections=su2tensor.fusionTreeDirections,
                                           chargeSectors=su2tensor.listOfChargeSectors.copy(),
                                           nameOrdering=su2tensor.nameOrdering.copy())
    )

    newListOfDegerneracyTensors = (
        reverse_leg_updateDegeneracyTensors(reverseNode=reverseNode,
                                            reverseNodeDirection=reverseNodeDirection,
                                            listOfChargeSectors=su2tensor.listOfChargeSectors,
                                            nameOrdering=su2tensor.nameOrdering,
                                            listOfDegeneracyTensors=su2tensor.listOfDegeneracyTensors,
                                            use_fFactor=use_fFactor
                                            )
    )

    newNameOrdering, newListOfOpenEdges, newChargeSectors = reverse_leg_updateNameOrderingAndOpenEdgesOrdering(
                                                        reverseNodeDirection=reverseNodeDirection,
                                                        nameOrdering=newNameOrdering,
                                                        listOfOpenEdges=su2tensor.listOfOpenEdges,
                                                        listOfChargeSectors=newChargeSectors)

    newFusionTree, newFusionTreeDirections, newNameOrdering, newChargeSectors = remove_unnecessary_nodes(
        chargeSectors=newChargeSectors,
        nameOrdering=newNameOrdering,
        fusionTree=newFusionTree,
        fusionTreeDirections=newFusionTreeDirections
    )

    su2tensor.fusionTree = newFusionTree
    su2tensor.fusionTreeDirections = newFusionTreeDirections
    su2tensor.listOfDegeneracyTensors = newListOfDegerneracyTensors
    su2tensor.listOfChargeSectors = newChargeSectors
    su2tensor.nameOrdering = newNameOrdering
    su2tensor.listOfOpenEdges = newListOfOpenEdges
    su2tensor.numberOfInternalEdges = len(su2tensor.fusionTree) - 1


def left_reverse_leg_get_newFusionTree(reverseLeg, oldFusionTree, oldFusionTreeDirections,
                                  chargeSectors, nameOrdering):
    newFusionTree = oldFusionTree.copy()
    newFusionTreeDirections = oldFusionTreeDirections.copy()
    reverseNode = [node for node in oldFusionTree if reverseLeg in node]
    assert len(reverseNode) == 1
    reverseNode = reverseNode[0]
    reverseNode_idx = oldFusionTree.index(reverseNode)
    reverseNodeDirection = newFusionTreeDirections[reverseNode_idx]

    numberOfInternalEdges = len(oldFusionTree) - 1

    # reverseNode is a fusionNode
    if reverseNodeDirection == -1:
        if reverseNode[2] == reverseLeg:
            newFusionTree[reverseNode_idx] = [reverseNode[0], reverseNode[1], numberOfInternalEdges + 1]
            newFusionTree.append([reverseNode[2], numberOfInternalEdges + 1, 0])
            newFusionTreeDirections.append(-1)

            chargeSectors, nameOrdering = reverse_leg_add_internalEdgeToChargeSectors(
                chargeSectors, nameOrdering, reverseNode[2], numberOfInternalEdges)

            use_fFactor = False
            # raise Exception('Reversing: This is not yet implemented.')
        else:
            newFusionTree[reverseNode_idx] = [reverseNode[1], reverseNode[0], reverseNode[2]]
            newFusionTreeDirections[reverseNode_idx] = +1

            use_fFactor = True

    elif reverseNodeDirection == +1:
        if reverseNode[0] == reverseLeg:
            newFusionTree[reverseNode_idx] = [numberOfInternalEdges + 1, reverseNode[1], reverseNode[2]]
            newFusionTree.append([0, reverseNode[0], numberOfInternalEdges + 1])
            newFusionTreeDirections.append(+1)

            chargeSectors, nameOrdering = reverse_leg_add_internalEdgeToChargeSectors(
                chargeSectors, nameOrdering, reverseNode[0], numberOfInternalEdges)

            use_fFactor = False

            # raise Exception('Reversing: This is not yet implemented.')
        else:
            newFusionTree[reverseNode_idx] = [reverseNode[1], reverseNode[0], reverseNode[2]]
            newFusionTreeDirections[reverseNode_idx] = -1

            use_fFactor = True

    return (newFusionTree, newFusionTreeDirections, reverseNode, reverseNodeDirection,
            chargeSectors, nameOrdering, use_fFactor)


def reverse_leg_add_internalEdgeToChargeSectors(chargeSectors, nameOrdering, reversedLeg, numberOfInternalEdges):
    chargeSector_df = pd.DataFrame(chargeSectors, columns=nameOrdering)

    # the values in the charge sector for the new internal leg are the same as for the reversed leg
    reversedLegColumnData = chargeSector_df[reversedLeg].values.flatten().tolist()
    chargeSector_df[numberOfInternalEdges + 1] = reversedLegColumnData

    nameOrdering = nameOrdering[:numberOfInternalEdges] + [numberOfInternalEdges + 1] + nameOrdering[
                                                                                        numberOfInternalEdges:]
    # order the columns in the charge sector dataframe
    chargeSector_df = chargeSector_df[nameOrdering]
    chargeSectors = chargeSector_df.values.tolist()

    return chargeSectors, nameOrdering


def F_factor_splittingToFusion_left(jA, jB, jC):
    """
    Return the F-factor F^(jd, je)_(ja, jb, jc, jabc)
    """
    return F_factor(jA=jB, jB=jB, jC=jC, jABC=jC, jd=0, je=jA)


def F_factor_fusionToSplitting_left(jA, jB, jC):
    """
    Return the F-factor F^(jd, je)_(ja, jb, jc, jabc)
    """
    return F_factor(jA=jB, jB=jA, jC=jA, jABC=jB, jd=jC, je=0)


def reverse_leg_updateDegeneracyTensors(reverseNode, reverseNodeDirection, listOfDegeneracyTensors,
                                        nameOrdering, listOfChargeSectors, use_fFactor):
    """
    We need to multiply the degeneracy tensors with the respective tensors. The charge sectors remain unchanged.
    """
    nameOrdering.copy()
    nameOrdering = [0] + nameOrdering
    listOfChargeSectors = listOfChargeSectors.copy()
    for idx in range(len(listOfChargeSectors)):
        listOfChargeSectors[idx] = [0] + listOfChargeSectors[idx]

    newListOfDegeneracyTensors = listOfDegeneracyTensors.copy()
    ja_idx = nameOrdering.index(reverseNode[0])
    jb_idx = nameOrdering.index(reverseNode[1])
    jc_idx = nameOrdering.index(reverseNode[2])

    for idx, (degeneracyTensor, chargeSector) in enumerate(zip(listOfDegeneracyTensors, listOfChargeSectors)):
        ja, jb, jc = chargeSector[ja_idx], chargeSector[jb_idx], chargeSector[jc_idx]
        nDim = len(list(degeneracyTensor.shape))
        if reverseNodeDirection == -1:
            if use_fFactor:
                F_factor = F_factor_fusionToSplitting_left(jA=ja, jB=jb, jC=jc)
                factor = (-1) ** (2 * ja) * np.sqrt(2 * ja + 1) * F_factor
            else:
                factor = np.sqrt(2 * jc + 1)
            degeneracyTensor = np.transpose(degeneracyTensor, (list(range(1, nDim)) + [0]))

        elif reverseNodeDirection == 1:
            if use_fFactor:
                F_factor = F_factor_splittingToFusion_left(jA=ja, jB=jb, jC=jc)
                factor = np.sqrt(2 * jb + 1) * F_factor
            else:
                factor = np.sqrt(2 * ja + 1)
            degeneracyTensor = np.transpose(degeneracyTensor, ([nDim-1] + list(range(nDim-1))))

        newListOfDegeneracyTensors[idx] = factor * degeneracyTensor

    return newListOfDegeneracyTensors


def reverse_leg_updateNameOrderingAndOpenEdgesOrdering(reverseNodeDirection, nameOrdering,
                                                       listOfOpenEdges, listOfChargeSectors):
    df_chargeSectors = pd.DataFrame(listOfChargeSectors, columns=nameOrdering)
    openEdgeOrdering = [edgeName for edgeName in nameOrdering if edgeName < 0]
    if reverseNodeDirection == -1:
        newNameOrdering = nameOrdering[:-len(openEdgeOrdering)] + openEdgeOrdering[1:] + [openEdgeOrdering[0]]
    elif reverseNodeDirection == +1:
        newNameOrdering = nameOrdering[:-len(openEdgeOrdering)] + [openEdgeOrdering[-1]] + list(openEdgeOrdering[:-1])
        print(nameOrdering[:-len(openEdgeOrdering)])
        print([openEdgeOrdering[-1]])
        print(list(openEdgeOrdering[:-1]))
        print(newNameOrdering)

    for idx, openEdge in enumerate(listOfOpenEdges):
        openEdgeName = openEdge['edgeName']
        openEdge['edgeNumber'] = openEdgeOrdering.index(openEdgeName) + 1
        listOfOpenEdges[idx] = openEdge

    newListOfChargeSectors = df_chargeSectors[newNameOrdering].values.tolist()
    return newNameOrdering, listOfOpenEdges, newListOfChargeSectors


if __name__ == "__main__":

    ja = 1/2
    jb = 1
    jc = 3/2

    print((-1)**(2*ja)*np.sqrt(2*ja+1)*F_factor_fusionToSplitting_left(jA=ja, jB=jb, jC=jc))
    print(np.sqrt(2*ja+1)*F_factor_splittingToFusion_left(jA=jb, jB=ja, jC=jc))
    # fusionTree = [[0, -1, 1], [1, -2, -3]]
    # fusionTreeDirections = [1, 1]
    #
    # listOpenEdges = [
    #     {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
    #     {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
    #     {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None}
    # ]
    #
    # su2tensor = SU2Tensor(listOfOpenEdges=listOpenEdges,
    #                       listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges))],
    #                       fusionTree=fusionTree,
    #                       fusionTreeDirections=fusionTreeDirections)
    #
    # listOfDegeneracyTensors = []
    # for chargeSector in su2tensor.listOfChargeSectors:
    #     listOfDegeneracyTensors.append(np.random.rand(2, 2, 2, 2))
    # su2tensor.listOfDegeneracyTensors = listOfDegeneracyTensors
    #
    # # calculate the explicit tensors
    # control_dict = su2tensor.return_explicit_tensor_blocks()
    # cs_control = list(zip(su2tensor.listOfChargeSectors, su2tensor.listOfDegeneracyTensors))
    #
    # reverse_left_most_leg(su2tensor=su2tensor, reverseLeg=-1)
    # print('____ revers 2 ____')
    # reverse_left_most_leg(su2tensor=su2tensor, reverseLeg=-1)
    #
    # # print(su2tensor.fusionTree)
    # cs_test = list(zip(su2tensor.listOfChargeSectors, su2tensor.listOfDegeneracyTensors))
    #
    # for (cs_c, deg_c), (cs_t, deg_t) in zip(cs_control, cs_test):
    #     assert cs_c == cs_t
    #     assert np.allclose(deg_c, deg_t)
    #
    # print(su2tensor.fusionTree, su2tensor.fusionTreeDirections)

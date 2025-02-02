import numpy as np
import pandas as pd
from su2tn.su2tensor_utils.f_move_util import F_factor
from su2tn.su2tensor_utils.remove_unnec_dummylegs import remove_unnecessary_nodes


def reverse_left_most_leg(su2tensor, reverseLeg):
    """
    Reverse the rightmost leg of a tensor. The fusion tree has already been converted such that the leg is already
    isolated and can be reversed immediately.
    """
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

        if reverseNodeDirection == -1:
            if use_fFactor:
                F_factor = F_factor_fusionToSplitting_left(jA=ja, jB=jb, jC=jc)
                factor = (-1) ** (2 * ja) * np.sqrt(2 * ja + 1) * F_factor
            else:
                factor = np.sqrt(2 * jc + 1)

        elif reverseNodeDirection == 1:
            if use_fFactor:
                F_factor = F_factor_splittingToFusion_left(jA=ja, jB=jb, jC=jc)
                factor = np.sqrt(2 * jb + 1) * F_factor
            else:
                factor = np.sqrt(2 * ja + 1)

        newListOfDegeneracyTensors[idx] = factor * degeneracyTensor

    return newListOfDegeneracyTensors

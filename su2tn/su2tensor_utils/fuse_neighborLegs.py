import numpy as np
import pandas as pd
from su2tn.su2tensor_utils.switch_fusion_tree_make_neighbors import metric_NeighboringNodes
from su2tn.su2tensor_utils.fuse_neighboringLegs_oneNode import fuse_neighboring_legs_oneNode


def fuse_neighboring_legs(su2tensor, irrep1Name, irrep2Name):
    """
    Function that performs the fusing of the adjacent legs.
    """
    if len(su2tensor.fusionTree) == 1:
        fuse_neighboring_legs_oneNode(su2tensor, irrep1Name, irrep2Name)

    else:
        # do the checks
        fusionNode, irrep12Name = fuse_neighboring_legs_checks(su2tensor=su2tensor,
                                                               irrep1Name=irrep1Name, irrep2Name=irrep2Name)

        # calculate all the changes
        newNameOrdering, newChargeSectors, newListOfDegeneracyTensors, edgeIrreps = (
            fuse_neighboring_legs_chargesector_groups(su2tensor=su2tensor,
                                                      irrep1Name=irrep1Name, irrep2Name=irrep2Name,
                                                      irrep12Name=irrep12Name))

        newListOfOpenEdges = fuse_neighboring_legs_update_openLegs(su2tensor=su2tensor,
                                                                   irrep1Name=irrep1Name, irrep2Name=irrep2Name,
                                                                   edgeIrreps12=edgeIrreps)

        # if len(su2tensor.fusionTree) != 1:
        newFusionTree, newFusionTreeDirections = fuse_neighboring_legs_updatefusiontree(su2tensor=su2tensor,
                                                                                        fusionNode=fusionNode,
                                                                                        irrep1Name=irrep1Name,
                                                                                        irrep12Name=irrep12Name)

        # do all the changes to the su2tensor
        fuse_neighboring_legs_updatesu2tensor_inplace(su2tensor=su2tensor,
                                                      newFusionTree=newFusionTree,
                                                      newFusionTreeDirections=newFusionTreeDirections,
                                                      newNameOrdering=newNameOrdering,
                                                      newListOfOpenEdges=newListOfOpenEdges,
                                                      newChargeSectors=newChargeSectors,
                                                      newListOfDegeneracyTensors=newListOfDegeneracyTensors)


def fuse_neighboring_legs_checks(su2tensor, irrep1Name, irrep2Name):
    """
    Function that first checks whether the conditions are met to fuse two neighboring legs of a su2tensor.
    """
    # assert that the two legs are neighboring
    assert metric_NeighboringNodes(irrep1Name, irrep2Name, su2tensor.fusionTree) == 2

    fusionNode = [node for node in su2tensor.fusionTree if (irrep1Name in node and irrep2Name in node)]
    assert len(fusionNode) == 1
    fusionNode = fusionNode[0]

    # make sure to fuse two incoming or two outgoing irreps
    fusionNode_idx = su2tensor.fusionTree.index(fusionNode)
    fusionNodeDirection = su2tensor.fusionTreeDirections[fusionNode_idx]
    if fusionNodeDirection == -1:
        assert {fusionNode[0], fusionNode[1]} == {irrep1Name, irrep2Name}
        irrep12Name = fusionNode[2]
    elif fusionNodeDirection == 1:
        assert {fusionNode[1], fusionNode[2]} == {irrep1Name, irrep2Name}
        irrep12Name = fusionNode[0]

    return fusionNode, irrep12Name


def fuse_neighboring_legs_nameOrdering(su2tensor, groupingNamesList, irrep1Name, irrep2Name, irrep12Name):
    """
    Helper function of the fuse_neighboring_legs_chargesector_groups function that creates the new name ordering and the
    grouping for the charge sectors that constitute to the new degeneracy tensors.
    """
    newNameOrdering = su2tensor.nameOrdering.copy()
    newNameOrdering.remove(irrep12Name)
    newNameOrdering.remove(irrep2Name)

    internalLegsList = [leg for leg in newNameOrdering if leg > 0]
    newInternalLegNames = list(range(1, len(internalLegsList) + 1))

    df_newNameOrdering = pd.DataFrame(newNameOrdering)
    newNameOrdering = df_newNameOrdering.replace(internalLegsList, newInternalLegNames).values.flatten().tolist()

    groupingNamesList[groupingNamesList.index(irrep12Name)] = irrep1Name
    df_groupingNamesList = pd.DataFrame(groupingNamesList)
    groupingNamesList = df_groupingNamesList.replace(internalLegsList, newInternalLegNames).values.flatten().tolist()

    return newNameOrdering, groupingNamesList


def fuse_neighboring_legs_chargesector_groups(su2tensor, irrep1Name, irrep2Name, irrep12Name):
    """
    Helper function that returns new nameOrdering, listOfChargeSectors and listOfDegeneracyTensors. Also returns
    edgeIrreps = [(j_i, t_ji)] list for updating listOfOpenEdges.
    """
    # group the charge sectors, so that the ones contributing to a new charge sector are grouped together.
    cs_df = pd.DataFrame(su2tensor.listOfChargeSectors, columns=su2tensor.nameOrdering)

    groupingNamesList = su2tensor.nameOrdering.copy()
    groupingNamesList.remove(irrep1Name)
    groupingNamesList.remove(irrep2Name)
    cs_groups = cs_df.groupby(groupingNamesList)

    irrep12_cs_idx = groupingNamesList.index(irrep12Name)

    # the new name ordering is that the fused leg gets the name of irrep1
    newNameOrdering, groupingNamesList = fuse_neighboring_legs_nameOrdering(su2tensor=su2tensor,
                                                                            groupingNamesList=groupingNamesList,
                                                                            irrep1Name=irrep1Name,
                                                                            irrep2Name=irrep2Name,
                                                                            irrep12Name=irrep12Name)

    openEdgeOrder = su2tensor.nameOrdering[su2tensor.numberOfInternalEdges:]
    irrep1_idx = openEdgeOrder.index(irrep1Name)
    irrep2_idx = openEdgeOrder.index(irrep2Name)

    newChargeSectors = []
    newListOfDegeneracyTensors = []

    edgeIrreps = set()
    for idx, (newChargeSector, df) in enumerate(cs_groups):
        # make sure that the rows are ordered such that we can slice the degeneracy tensors back.
        df = df.sort_values(by=[irrep1Name, irrep2Name], ascending=True)
        newChargeSectors.append(list(newChargeSector))
        irrep12 = newChargeSector[irrep12_cs_idx]

        newDegeneracyTensor, dimFusedLeg = fuse_neighboring_legs_degeneracy_tensors(su2tensor=su2tensor,
                                                                                    df=df,
                                                                                    irrep1_idx=irrep1_idx,
                                                                                    irrep2_idx=irrep2_idx)
        newListOfDegeneracyTensors.append(newDegeneracyTensor)

        edgeIrreps.add((irrep12, dimFusedLeg))

    newChargeSectors = pd.DataFrame(newChargeSectors, columns=groupingNamesList)
    newChargeSectors = newChargeSectors[newNameOrdering].values.tolist()

    return newNameOrdering, newChargeSectors, newListOfDegeneracyTensors, sorted(list(edgeIrreps))


def fuse_neighboring_legs_degeneracy_tensors_transpose(irrep1_idx, irrep2_idx, num_dims):
    perm = list(range(num_dims))
    perm.remove(irrep2_idx)
    perm.insert(perm.index(irrep1_idx)+1, irrep2_idx)
    irrep1_idx = perm.index(irrep1_idx)
    irrep2_idx = perm.index(irrep2_idx)
    return perm, irrep1_idx, irrep2_idx


def fuse_neighboring_legs_degeneracy_tensors(su2tensor, df, irrep1_idx, irrep2_idx):
    """
    Helper function that calculates the new degeneracy tensor for each new charge sector and also returns the dimension
    (t_ji) of the new degeneracy tensor along the new dimension.
    """

    num_dims = len(list(su2tensor.listOfDegeneracyTensors[0].shape))
    perm, irrep1_idx, irrep2_idx = fuse_neighboring_legs_degeneracy_tensors_transpose(
                                                    irrep1_idx=irrep1_idx, irrep2_idx=irrep2_idx, num_dims=num_dims)

    oldDegeneracyTensors = []
    for row_idx in df.index:
        oldDegeneracyTensor = su2tensor.listOfDegeneracyTensors[row_idx]
        oldDegeneracyTensor = np.transpose(oldDegeneracyTensor, perm)
        tensorShape = list(oldDegeneracyTensor.shape)
        newDim = tensorShape[irrep1_idx] * tensorShape[irrep2_idx]
        tensorShape.pop(max(irrep1_idx, irrep2_idx))
        tensorShape[min(irrep1_idx, irrep2_idx)] = newDim

        oldDegeneracyTensor = oldDegeneracyTensor.reshape(tensorShape)
        oldDegeneracyTensors.append(oldDegeneracyTensor)

    newDegeneracyTensor = np.concatenate(oldDegeneracyTensors, axis=min(irrep1_idx, irrep2_idx))
    dimFusedLeg = newDegeneracyTensor.shape[min(irrep1_idx, irrep2_idx)]

    return newDegeneracyTensor, dimFusedLeg


def fuse_neighboring_legs_update_openLegs(su2tensor, irrep1Name, irrep2Name, edgeIrreps12):
    """
    Helper function that updates the listOfOpenEdges in the su2tensor after the fusing of two legs.
    """
    # remove the two legs that are being fused from the listOfOpenEdges
    listOfOpenEdges = su2tensor.listOfOpenEdges.copy()
    openLeg1 = [ol for ol in listOfOpenEdges if ol['edgeName'] == irrep1Name][0]
    listOfOpenEdges.remove(openLeg1)
    openLeg2 = [ol for ol in listOfOpenEdges if ol['edgeName'] == irrep2Name][0]
    listOfOpenEdges.remove(openLeg2)

    # create the new entry for listOfOpenLegs for the fused leg.
    newOpenLeg = {}
    newOpenLeg['edgeName'] = openLeg1['edgeName']
    newOpenLeg['edgeNumber'] = min(openLeg1['edgeNumber'], openLeg2['edgeNumber'])
    newOpenLeg['edgeIrreps'] = edgeIrreps12
    newOpenLeg['isFused'] = True
    if openLeg1['isFused'] == True:
        openLeg1History = openLeg1['originalIrreps']
    else:
        openLeg1History = openLeg1['edgeIrreps']

    if openLeg2['isFused'] == True:
        openLeg2History = openLeg2['originalIrreps']
    else:
        openLeg2History = openLeg2['edgeIrreps']
    newOpenLeg['originalIrreps'] = [(irrep1Name, edgeIrreps12), (irrep1Name, openLeg1History),
                                    (irrep2Name, openLeg2History)]

    listOfOpenEdges.append(newOpenLeg)

    # update the edgeNumber of all the legs that are behind the fused leg.
    for ol in listOfOpenEdges:
        if ol['edgeNumber'] >= max(openLeg1['edgeNumber'], openLeg2['edgeNumber']):
            ol['edgeNumber'] = ol['edgeNumber'] - 1

    return listOfOpenEdges


def fuse_neighboring_legs_updatefusiontree(su2tensor, fusionNode, irrep1Name, irrep12Name):
    """
    Helper function that updates the fusionTree and fusionTreeDirections after fusing two legs.
    """
    fusionNode_idx = su2tensor.fusionTree.index(fusionNode)

    # get the two lsit on how to rename the internal legs
    renameInternalLegsList = su2tensor.nameOrdering.copy()
    renameInternalLegsList.remove(irrep12Name)
    internalLegsList = [leg for leg in renameInternalLegsList if leg > 0]
    renameInternalLegsList = list(range(1, len(internalLegsList) + 1))

    # get the new fusion tree with updated names for the edges
    newFusionTree = su2tensor.fusionTree.copy()
    newFusionTree.remove(fusionNode)
    df_tree = pd.DataFrame(newFusionTree)
    # rename the fused leg to an external leg
    df_tree = df_tree.replace([irrep12Name], [irrep1Name])
    # rename the internal edges
    newFusionTree = df_tree.replace(internalLegsList, renameInternalLegsList).values.tolist()

    newFusionTreeDirections = su2tensor.fusionTreeDirections.copy()
    del newFusionTreeDirections[fusionNode_idx]

    return newFusionTree, newFusionTreeDirections


def fuse_neighboring_legs_updatefusiontree_oneNode(su2tensor, fusionNode, irrep1Name, irrep12Name):
    """
    Helper function that updates the fusionTree and fusionTreeDirections after fusing two legs.
    """
    fusionNode_idx = su2tensor.fusionTree.index(fusionNode)

    fusionNodeDirection = su2tensor.fusionTreeDirections[fusionNode_idx]

    if fusionNodeDirection == -1:
        newFusionTree = [[irrep1Name, 0, irrep12Name]]
        newFusionTreeDirections = [-1]
    elif fusionNodeDirection == +1:
        newFusionTree = [[irrep12Name, 0, irrep1Name]]
        newFusionTreeDirections = [+1]

    return newFusionTree, newFusionTreeDirections


def fuse_neighboring_legs_updatesu2tensor_inplace(su2tensor,
                                                  newFusionTree, newFusionTreeDirections,
                                                  newListOfOpenEdges,
                                                  newNameOrdering,
                                                  newChargeSectors, newListOfDegeneracyTensors):
    """
    Helper function that updates all the relevant information in the su2tensor inplace.
    """

    su2tensor.numberOfInternalEdges = len(newFusionTree) - 1
    su2tensor.fusionTree, su2tensor.fusionTreeDirections = newFusionTree, newFusionTreeDirections
    su2tensor.listOfOpenEdges = newListOfOpenEdges
    su2tensor.nameOrdering = newNameOrdering
    su2tensor.listOfChargeSectors = newChargeSectors
    su2tensor.listOfDegeneracyTensors = newListOfDegeneracyTensors
    su2tensor.listOfStructuralTensors = None
    su2tensor.numberOfOpenEdges = su2tensor.numberOfOpenEdges - 1
    su2tensor.numberOfInternalEdges = len(su2tensor.fusionTree) - 1


if __name__ == "__main__":
    from su2tn.su2_tensor import SU2Tensor

    fusionTree = [[-1, -2, -3]]
    fusionTreeDirections = [-1]

    listOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 2), (1, 2)], 'isFused': False, 'originalIrreps': None}
    ]

    su2tensor = SU2Tensor(listOfOpenEdges=listOpenEdges,
                          listOfDegeneracyTensors=[None for _ in range(len(listOpenEdges))],
                          fusionTree=fusionTree,
                          fusionTreeDirections=fusionTreeDirections)

    listOfDegeneracyTensors = []
    for chargeSector in su2tensor.listOfChargeSectors:
        listOfDegeneracyTensors.append(np.random.rand(2, 2, 2))
    su2tensor.listOfDegeneracyTensors = listOfDegeneracyTensors

    # calculate the explicit tensors
    control_dict = su2tensor.return_explicit_tensor_blocks()

    fuse_neighboring_legs(su2tensor, -1, -2)
    print([openLeg for openLeg in su2tensor.listOfOpenEdges if openLeg['edgeName'] == -1][0])
    # fuse_neighboring_legs(su2tensor, -1, -3)

    print(su2tensor.fusionTree)
    print(su2tensor.nameOrdering)

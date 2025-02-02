import numpy as np
import pandas as pd
import itertools
from collections import Counter
from su2tn.su2tensor_utils.f_move_util import f_move_build_new_fusion_tree, f_move_perform_f_move_order


def f_move_find_minimal_internal_edge_difference(initialTree, finalTree):
    """
    Returns a list of internal edges in initialTree that are different from finalTree with taking into account that the
    names for the internal edges can differ in the finalTree. We need this information to check which internal edges
    need to be changed by applying f-move to.
    """
    # the number of the internal edges is the maximum values in the names for the nodes
    numberOfInternalEdges = np.max(np.array(initialTree))
    df_finalTree = pd.DataFrame(finalTree)
    # find the permutation of internal edges where the least amount of nodes differs
    minDifferentNodes = np.inf
    for perm in itertools.permutations(range(1, numberOfInternalEdges+1)):
        # apply the permutation of internal edges and build the new final tree
        perm_finalTree = df_finalTree.replace(list(range(1, numberOfInternalEdges+1)), list(perm)).values.tolist()

        # the nodes that differ in initialTree from finalTree
        differentNodes = [node for node in initialTree if node not in perm_finalTree]

        # check if this permutation has less different nodes to find the best permutation
        if minDifferentNodes > len(differentNodes):
            minDifferentNodes = len(differentNodes)
            bestPerm_finalTree = perm_finalTree
            internalEdgeDiff = [item for item, count in Counter(np.array(differentNodes).flatten()).items() if count > 1]

    # return the list of internal edges that are really different (and not just because of labeling of the edges)
    # and the new final tree with the internal edges renamed
    return (internalEdgeDiff, bestPerm_finalTree)


def f_move_find_correct_fMove_order(internalEdgeDiff,
                                    initialTree, finalTree,
                                    initialTreeDirections, finalTreeDirections):
    """
    Finds the correct move order to apply to internal edges. We only check which order of f-moves leads to the correct
    fusion tree. With this information, we can then apply the actual f-moves to the tensor (with changing the degeneracy
    tensors as well). Because the changing of the degeneracy tensors takes the most time, we need to first make sure
    we have the correct order of f-moves, which is what we do in this function.
    """
    # check if the initial and final tree have the same amount of splitting and fusion nodes
    if sorted(initialTreeDirections) != sorted(finalTreeDirections):
        raise AssertionError(
            "F-Move: Final tree needs to have same number of splitting and fusion nodes as the initial tree.")

    if len(initialTree) != len(finalTree):
        raise AssertionError("F-Move: Final tree needs to have the same number of nodes as current tree.")

    initialTree = initialTree.copy()
    initialTreeDirections = initialTreeDirections.copy()

    # check if the edge between the splitting and fusion tree is in internalEdgeDiff and remove if necessary.
    for intEdge in internalEdgeDiff:
        # get the nodes that are connected by the internal edge
        intNodes = [node for node in initialTree if intEdge in node]
        intNodes1_idx, intNodes2_idx = initialTree.index(intNodes[0]), initialTree.index(intNodes[1])
        # the two nodes are not of the same kind (not both splitting or fusion nodes)
        numberOfSkips = 0
        if initialTreeDirections[intNodes1_idx] != initialTreeDirections[intNodes2_idx]:
            numberOfSkips += 1
            print("F-Move: For finding the correct f-move the internal edge between splitting and fusion node skipped.")
            internalEdgeDiff.remove(intEdge)
        if numberOfSkips >= 2:
            raise AssertionError("F-Move: Fusion tree is not a simple tree.")

    # make all permutations of elements in internalEdgeDiff, e.g. the possible combinations of f-moves.
    for perm in itertools.permutations(internalEdgeDiff):
        outputTree = initialTree.copy()
        outputTreeDirections = initialTreeDirections.copy()
        for edge in perm:
            # get nodes to apply the f-move to
            fNodes = [fnode for fnode in outputTree if edge in fnode]
            # apply the f-move
            outputTree, outputTreeDirections, _ = f_move_build_new_fusion_tree(fNodes, outputTree, outputTreeDirections)

        # output after applying the sequence of f-moves.
        outputTreeResult = list(zip(outputTree, outputTreeDirections))
        outputTreeResult.sort()

        # check the different permutations of the internalEdgeDiff in the final graph and compare to the output graph
        df_finalTree = pd.DataFrame(finalTree.copy())
        for perm2 in itertools.permutations(internalEdgeDiff):
            perm_finalTree = df_finalTree.replace(internalEdgeDiff, list(perm2)).values.tolist()

            # final tree with directions to compare to the output tree after the f-moves to see if they are the same
            finalTreeControl = list(zip(perm_finalTree, finalTreeDirections))
            finalTreeControl.sort()

            # check if the output tree is the final tree and, thus, we have found the correct sequence of f-moves.
            if outputTreeResult == finalTreeControl:
                return list(perm)

    raise Exception('F-Move: No possible f-move found')


def f_move_switch_fusion_tree(su2tensor, newFusionTree, newFusionTreeDirections):
    """
    Switches the su2-tensor to the given new fusion tree.
    """
    initialTree = su2tensor.fusionTree
    initialTreeDirections = su2tensor.fusionTreeDirections

    finalTree = newFusionTree.copy()
    finalTreeDirections = newFusionTreeDirections.copy()

    # find the edges really are different (and not just different labels) and change the final tree labels
    internalEdgeDiff, finalTree = f_move_find_minimal_internal_edge_difference(initialTree=initialTree,
                                                                               finalTree=finalTree)

    # find the correct order in which to apply the f-moves
    fMoveOrder = f_move_find_correct_fMove_order(internalEdgeDiff=internalEdgeDiff,
                                                 initialTree=initialTree, initialTreeDirections=initialTreeDirections,
                                                 finalTree=finalTree, finalTreeDirections=finalTreeDirections)

    f_move_perform_f_move_order(su2tensor, fMoveOrder)


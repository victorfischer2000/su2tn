import numpy as np
import networkx as nx
import json
from su2tn.su2tensor_utils.fusion_tree_util import fusion_tree_list_to_graph
from su2tn.su2tensor_utils.f_move_util import f_move_build_new_fusion_tree, f_move_perform_f_move_order


def metric_NeighboringNodes(u, v, fusionTreeNodes):
    """
    Calculates how many edges need to be transversed in the shortest path between the open edges u and v. This is
    important if the open edges u and v should belong to the same node, in which case the distance is 2.
    """
    graphTree = fusion_tree_list_to_graph(fusionTreeNodes)
    return nx.shortest_path_length(graphTree, source=f'[{u}]', target=f'[{v}]')


def get_internal_path_edges_for_neighboring(fusionTreeNodes, u, v):
    """
    For a fusion tree defined by fusionTreeNodes, returns the internal edges that are transversed when walking along the shortest path from the source node u to the
    target node v.
    """
    graphTree = fusion_tree_list_to_graph(fusionTreeNodes)
    shortestPath = nx.shortest_path(graphTree, source=f'[{u}]', target=f'[{v}]')
    intLegPath = []
    node1 = json.loads(shortestPath[1])
    for idx in range(2, len(shortestPath) - 1):
        node2 = json.loads(shortestPath[idx])
        intEdge = [leg for leg in node1 if leg in node2]
        # there is only one internal leg between two nodes
        assert len(intEdge) == 1
        assert intEdge[0] > 0
        intLegPath.append(intEdge[0])
        node1 = node2

    return intLegPath


def check_if_two_edges_next_to_each_other(nameOrdering, irrep1, irrep2):
    """
    Checks if two elements are neighboring in the nameOrdering list or on opposite ends of the list.
    """
    idx1 = nameOrdering.index(irrep1)
    idx2 = nameOrdering.index(irrep2)

    distance = abs(idx1 - idx2) # % (len(nameOrdering) - 2)
    if distance != 1:
        raise AssertionError('Permutation: The two irreps that are supposed to be switched are not neighboring.')


def check_f_move_order_neighboring_correct(fMoveOrder, fusionTreeNodes, fusionTreeDirections, u, v):
    """
    Checks if the calculated f-move order actually makes the two neighboring irreps part of the same node. This only
    checks the fusion tree and does not explicitly apply the f moves.
    """
    outputTree = fusionTreeNodes.copy()
    outputTreeDirections = fusionTreeDirections.copy()
    for intEdge in fMoveOrder:
        fNodes = [fnode for fnode in outputTree if intEdge in fnode]
        outputTree, outputTreeDirections, _ = f_move_build_new_fusion_tree(fNodes, outputTree, outputTreeDirections)

    if metric_NeighboringNodes(u, v, outputTree) != 2:
        raise AssertionError('Permuation: The proposed f-move order does not lead to the correct fusion tree.')


def switch_fusion_tree_to_make_neighboring(su2tensor, irrep1, irrep2, doChecks=True):
    """
    Switches the fusion tree of a given su2tensor to make the two neighboring irreps (irrep1, irrep2) part of the same
    node. This is an important routine for applying permutations or reshaping tensors.
    """
    # get the internal edges to which the f-moves should be applied.
    fMoveOrder = get_internal_path_edges_for_neighboring(su2tensor.fusionTree, irrep1, irrep2)

    if doChecks == True:
        check_f_move_order_neighboring_correct(fMoveOrder,
                                               su2tensor.fusionTree,
                                               su2tensor.fusionTreeDirections,
                                               irrep1, irrep2)

    # apply the f moves
    f_move_perform_f_move_order(su2tensor=su2tensor, fMoveOrder=fMoveOrder)

    if metric_NeighboringNodes(irrep1, irrep2, su2tensor.fusionTree) != 2:
        raise AssertionError("Permutation: Switching the fusion tree to make irreps neighboring did not work.")

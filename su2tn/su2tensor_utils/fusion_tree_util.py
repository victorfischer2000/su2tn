import numpy as np
import networkx as nx
from su2tn.su2tensor_utils.f_move_util import f_move_build_new_fusion_tree

def fusion_tree_list_to_graph(fusionTreeNodes):
    """
    Takes the structure for a fusion tree as a list and converts it into a networkx graph. The fusion tree nodes in the
    graph have the names of the edges they are attached to. The open legs in the fusion tree have a node at the end that
    has the name of the open edge.
    """
    numberOfOpenEdges = len(fusionTreeNodes) + 2
    numberOfInternalEdges = len(fusionTreeNodes) - 1

    graphTree = nx.Graph()

    openEdges = np.unique(np.array(fusionTreeNodes).flatten())
    openEdges = list(openEdges[openEdges < 0])

    # add all the internal edges
    for internalEdge in range(1, numberOfInternalEdges+1):
        addNodes = [node for node in fusionTreeNodes if internalEdge in node]
        assert len(addNodes) == 2
        graphTree.add_edge(str(addNodes[0]), str(addNodes[1]))

    # add nodes to the open edges
    # for openEdge in range(-numberOfOpenEdges, 0):
    for openEdge in openEdges:
        addNodes = [node for node in fusionTreeNodes if openEdge in node]
        assert len(addNodes) == 1
        graphTree.add_edge(str(addNodes[0]), str([openEdge]))

    return graphTree


def get_separationEdge(fusionTreeNodes, fusionTreeDirections):
    """
    Returns the name of the internal edge that separates the splitting and the fusion part of the fusion tree.
    """
    for intEdge in range(1, len(fusionTreeNodes)):
        # get the nodes that are connected by the internal edge
        intNodes = [node for node in fusionTreeNodes if intEdge in node]
        assert len(intNodes) == 2
        intNodes1_idx, intNodes2_idx = fusionTreeNodes.index(intNodes[0]), fusionTreeNodes.index(intNodes[1])
        # the two nodes are not of the same kind (not both splitting or fusion nodes)
        if fusionTreeDirections[intNodes1_idx] != fusionTreeDirections[intNodes2_idx]:
            return intEdge

    # Simple tree consists only of fusion or splitting nodes, e.g. no separation edge
    return None


def metric_isolatedLeg(u, fusionTreeNodes, fusionTreeDirections):
    """
    Calculated how many edges need to be transversed in the shortest path to get from an open leg u to the node that
    has the internal separation leg. If the open leg u is separated, the distance should be 1.
    """
    graphTree = fusion_tree_list_to_graph(fusionTreeNodes)
    u_idx = fusionTreeNodes.index(u)
    u_type = fusionTreeDirections[u_idx]

    # get the node that belongs to the separation edge and is of the same type as
    separationEdge = get_separationEdge(fusionTreeNodes, fusionTreeDirections)
    separationNodes = [node for node in fusionTreeNodes if separationEdge in node]
    target = [node for node in separationNodes if fusionTreeDirections[fusionTreeNodes.index(node)] == u_type]

    return nx.shortest_path_length(graphTree, source=f'[{u}]', target=f'[{target}]')


def find_fMove_order_for_metric(fusionTreeNodes, fusionTreeDirections, metric, metricOptimum, metricArgs):
    """
    Performs a branch and bound search for finding the optimal f-move order to minimize the given metric.
    """
    try_fusionTreeNodes = fusionTreeNodes.copy()
    try_fusionTreeDirections = fusionTreeDirections.copy()
    try_metricOptimum = metric(*metricArgs, fusionTreeNodes, fusionTreeDirections)
    if try_metricOptimum <= metricOptimum:
        print("Find f-move order for metric: Initial tree already optimal.")
        return []

    currentOptimum = try_metricOptimum
    firstLevel = [[try_fusionTreeNodes, try_fusionTreeDirections, try_metricOptimum, []]]
    numberOfInternalEdges = len(fusionTreeNodes) - 1

    # find the internal edge between the splitting and fusion tree. Because we should not apply an f-move to this edge.
    separationEdge = get_separationEdge(fusionTreeNodes, fusionTreeDirections)

    stepCount = 0
    while stepCount <= numberOfInternalEdges:
        secondLevel = []
        stepCount += 1
        for FL_fusionTree, FL_fusionTreeDirections, FL_metric, FL_history in firstLevel:
            for internalEdge in range(1, len(fusionTreeNodes)):
                # we can't apply an f-move to an internal edge twice
                if (internalEdge not in FL_history) and (internalEdge != separationEdge):
                    fNodes = [fnode for fnode in FL_fusionTree if internalEdge in fnode]
                    SL_outputTree, SL_outputTreeDirections, _ = f_move_build_new_fusion_tree(fNodes, FL_fusionTree, FL_fusionTreeDirections)
                    SL_metric = metric(*metricArgs, SL_outputTree, SL_outputTreeDirections)
                    SL_history = FL_history + [internalEdge]

                    if SL_metric <= metricOptimum:
                        print("Find f-move order for metric: Best f-move sequence found.")
                        return SL_history

                    elif SL_metric < currentOptimum:
                        # delete all the previous graphs as they have a worse metric
                        secondLevel = [[SL_outputTree, SL_outputTreeDirections, SL_metric, SL_history]]

                    elif SL_metric == currentOptimum:
                        # add this graph to be used in the next step
                        secondLevel += [[SL_outputTree, SL_outputTreeDirections, SL_metric, SL_history]]

        # the second level now becomes the first level
        firstLevel = secondLevel

    raise RuntimeError("Find f-move order for metric: No f-move order found to minimize metric.")


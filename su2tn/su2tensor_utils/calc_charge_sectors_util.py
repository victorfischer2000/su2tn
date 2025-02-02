import numpy as np
import itertools
from su2tn.util import clebsch_gordan_charge_sectors


def calculate_all_charge_sectors(listOfOpenEdges, nameOrdering, fusionTree, fusionTreeDirections):
    """
    From the list of open edges and the structure of the fusionTree, calculate all possible charge sectors.
    """
    nameOrdering = nameOrdering.copy()
    nameOrdering = [0] + nameOrdering

    numberOfOpenEdges = len(listOfOpenEdges)
    numberOfInternalEdges = len(fusionTree) - 1
    # names of edges, where we already calculated the possible values for the irrep.
    calculated_irreps = [0] + [OpenEdge['edgeName'] for OpenEdge in listOfOpenEdges]

    # list of lists for all irrep values for each open edge
    open_edge_irrep_values = [None for _ in range(numberOfOpenEdges)]
    for openEdge in listOfOpenEdges:
        open_edge_irrep_values[openEdge['edgeNumber'] - 1] = ([j[0] for j in openEdge['edgeIrreps']])

    # list of charge sectors where possible charge sectors will be consecutively added in this function.
    charge_sectors = []

    # save all the combinations of irreps for the open edges and write None values for the internal edges.
    for open_leg_irrep_combination in itertools.product(*[open_edge_irrep_values[k]
                                                          for k in range(numberOfOpenEdges)]):
        charge_sectors.append([0] + [None for _ in range(numberOfInternalEdges)] +
                              [ji for ji in open_leg_irrep_combination])

    nodesToBeCalculated = fusionTree.copy()
    nodesToBeCalculatedDirections = fusionTreeDirections.copy()

    # loop over nodes until, the charge sectors for all nodes have been determined.
    while nodesToBeCalculated:

        for node in nodesToBeCalculated:
            # count the number of legs for each node that already have been calculated.
            already_calculated_legs = [leg for leg in node if leg in calculated_irreps]

            # we know the values for no or just one irrep. So we just go to the next node.
            if len(already_calculated_legs) == 0 or len(already_calculated_legs) == 1:
                pass

            # we know 2 irreps and want to determine the possible values of the third irrep
            elif len(already_calculated_legs) == 2:
                charge_sectors, calculated_irreps, nodesToBeCalculated, nodesToBeCalculatedDirections = (
                    calc_charge_sectors_twoIrrepsKnown(node, already_calculated_legs, calculated_irreps,
                                                       nameOrdering, charge_sectors,
                                                       nodesToBeCalculated, nodesToBeCalculatedDirections))

            # we already calculated all three irreps in the node and now we have to do consistency checks, that the
            # values for the irreps still make sense for this node.
            elif len(already_calculated_legs) == 3:
                charge_sectors, nodesToBeCalculated, nodesToBeCalculatedDirections = (
                    calc_charge_sectors_threeIrrepsKnown(node,
                                                         nameOrdering, charge_sectors,
                                                         nodesToBeCalculated, nodesToBeCalculatedDirections))

    # remove the 0 dummy charge sectors and nameOrdering
    charge_sectors = np.array(charge_sectors)[:, 1:].tolist()

    return charge_sectors


def calc_charge_sectors_twoIrrepsKnown(node, already_calculated_legs, calculated_irreps,
                                       nameOrdering, charge_sectors,
                                       nodesToBeCalculated, nodesToBeCalculatedDirections):
    """
    Function that adds the candidates for the third leg if two legs already have candidate irrep values.
    """
    calc_node_idx = nodesToBeCalculated.index(node)
    calc_node = nodesToBeCalculated.pop(calc_node_idx)
    calc_node_direction = nodesToBeCalculatedDirections.pop(calc_node_idx)

    # get the names and the indices of the two calculated legs and the undetermined leg
    irrep1, irrep2 = already_calculated_legs
    irrep3 = list(set(calc_node) - set(already_calculated_legs))[0]

    irrep1_idx, irrep2_idx = nameOrdering.index(irrep1), nameOrdering.index(irrep2)
    irrep3_idx = nameOrdering.index(irrep3)
    calculated_irreps.append(irrep3)
    # remove the first element in the charge sector list and add the new lists with the possible
    # values for the undetermined irrep added.
    for _ in range(len(charge_sectors.copy())):
        charge_sector = charge_sectors.pop(0)
        # get values of irrep1 and irrep2 in charge sector and add the possible values of j3 that could
        # result in these j1 and j2 values.
        j1, j2 = charge_sector[irrep1_idx], charge_sector[irrep2_idx]
        for j3 in np.arange(np.abs(j1 - j2), j1 + j2 + 1):
            charge_sector[irrep3_idx] = j3
            charge_sectors.append(charge_sector.copy())

    return charge_sectors, calculated_irreps, nodesToBeCalculated, nodesToBeCalculatedDirections


def calc_charge_sectors_threeIrrepsKnown(node,
                                         nameOrdering, charge_sectors,
                                         nodesToBeCalculated, nodesToBeCalculatedDirections):
    """
    Function that remove all charge sectors that violate the selection rules for the current node.
    """
    calc_node_idx = nodesToBeCalculated.index(node)
    calc_node = nodesToBeCalculated.pop(calc_node_idx)
    calc_node_direction = nodesToBeCalculatedDirections.pop(calc_node_idx)
    irrep1, irrep2, irrep3 = calc_node
    irrep1_idx = nameOrdering.index(irrep1)
    irrep2_idx = nameOrdering.index(irrep2)
    irrep3_idx = nameOrdering.index(irrep3)
    if calc_node_direction == -1:
        # get all the values the irreps take without duplicates
        irrep1_values = list(set(np.array(charge_sectors)[:, irrep1_idx]))
        irrep2_values = list(set(np.array(charge_sectors)[:, irrep2_idx]))

        # check if the combination of a value from edge1 and edge2 con combine to a value of edge3.
        # If not, remove the entries where this combination appears

        node_possible_charge_sectors = clebsch_gordan_charge_sectors(irrep1_values, irrep2_values).tolist()
        # go through the charge sectors and check if the combinations of j1, j2 and j3 are possible
        for charge_sector in charge_sectors.copy():
            irrep_combination = [charge_sector[irrep1_idx], charge_sector[irrep2_idx], charge_sector[irrep3_idx]]
            if irrep_combination not in node_possible_charge_sectors:
                charge_sectors.remove(charge_sector)

    elif calc_node_direction == 1:
        # get all the values the irreps take without duplicates
        irrep2_values = list(set(np.array(charge_sectors)[:, irrep2_idx]))
        irrep3_values = list(set(np.array(charge_sectors)[:, irrep3_idx]))

        node_possible_charge_sectors = clebsch_gordan_charge_sectors(irrep2_values,
                                                                     irrep3_values).tolist()
        # go through the charge sectors and check if the combinations of j1, j2 and j3 are possible
        for charge_sector in charge_sectors.copy():
            irrep_combination = [charge_sector[irrep2_idx], charge_sector[irrep3_idx],
                                 charge_sector[irrep1_idx]]
            if irrep_combination not in node_possible_charge_sectors:
                charge_sectors.remove(charge_sector)

    return charge_sectors, nodesToBeCalculated, nodesToBeCalculatedDirections



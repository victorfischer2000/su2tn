import numpy as np
from su2tn.su2_tensor import SU2Tensor
from su2tn.algorithms.scalar_product import conjugate_singe_fusionNode_tensor
from su2tn.algorithms.lanczos_method import eigh_krylov

from su2tn.models.MPS_utils.orthonormalize_left import make_left_orthonormal
from su2tn.models.MPS_utils.orthonormailze_right import make_right_orthonormal


def dmrg_one_site(H, psi, numsweeps, lanczos_numiter, return_initial_energy=False, verify_block_calculation=False):
    """
    Approximate the ground state MPS by left and right sweeps and local two-site optimizations.

    Args:
        H: Hamiltonian as MPO
        psi: initial MPS used for optimization; will be overwritten
        numsweeps: maximum number of left and right sweeps

    Returns:
        numpy.ndarray: array of approximate ground state energies after each iteration
    """

    # number of lattice sites
    L = H.nsites
    assert L == psi.nsites

    # left and right operator blocks
    # initialize leftmost block by 1x1x1 identity
    BR = compute_right_operator_blocks(psi, H)
    BL = [None for _ in range(L)]
    BL[0] = get_left_trivial_tensor()

    en_min = np.zeros(numsweeps)
    en_list = []

    init_en = None
    if return_initial_energy:
        from classical_benchmark.standard_MPO import MPO as standard_MPO
        from classical_benchmark.models.standard_heisenberg import construct_heisenberg_mpo
        from su2tn.models.MPS_utils.MPS_util import get_explizit_tensor_state
        H_mat = standard_MPO(construct_heisenberg_mpo(L=psi.nsites, J=-(1/4), pbc=False)).as_matrix()
        init_vec = np.reshape(get_explizit_tensor_state(psi), int(2 ** psi.nsites))
        init_vec = init_vec / np.linalg.norm(init_vec)
        init_en = init_vec @ H_mat @ init_vec
        print('Initial energy ', init_en)


    # Number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0
        sweep_list = []

        # sweep from left to right (rightmost two lattice sites are handled by right-to-left sweep)
        for i in range(L-1):
            print('step: ', i)
            en, Aloc = eigh_krylov(vstart=psi.A[i], numiter=lanczos_numiter, H=H.A[i], L=BL[i], R=BR[i])
            psi.A[i] = Aloc
            # update the left blocks
            make_left_orthonormal(A_target=psi.A[i], A_right=psi.A[i + 1])
            BL[i + 1] = contract_left_block(psi.A[i], H.A[i], BL[i])

            # construct_local_hamiltonian_explicit(L=BL[i], H=H.A[i], R=BR[i])

            # H = construct_local_hamiltonian(L=BL[i], H=H.A[i], R=BR[i])
            print(en)
            sweep_list.append(en)

        # sweep from right to left
        for i in reversed(range(1, L)):
            print('step: ', i)
            en, Aloc = eigh_krylov(vstart=psi.A[i], numiter=lanczos_numiter, H=H.A[i], L=BL[i], R=BR[i])
            psi.A[i] = Aloc

            make_right_orthonormal(A_left=psi.A[i-1], A_target=psi.A[i])
            # update the right blocks
            BR[i-1] = contract_right_block(psi.A[i], H.A[i], BR[i])

            if verify_block_calculation:
                BR_verify = contract_right_block_verify(psi.A[i], H.A[i], BR[i])
                BR_test = BR[i-1].return_explicit_tensor_blocks()

                # if an key is not in the test key list, then the corresponding tensor has to be zero
                delete_keys = []
                for key in BR_verify.keys():
                    if key not in BR_test.keys():
                        delete_keys.append(key)

                for key in delete_keys:
                    del BR_verify[key]

                assert sorted(list(BR_verify.keys())) == sorted(list(BR_test.keys()))
                for cs in BR_verify.keys():
                    assert np.allclose(BR_verify[cs], BR_test[cs])

                contract_right_block_verify(psi.A[i], H.A[i], BR[i])
            print(en)
            sweep_list.append(en)
        en_list.append(sweep_list)
        # record energy after each sweep
        en_min[n] = en
        print(en)
        print("sweep {} completed, current energy: {}".format(n + 1, en))

    return init_en, en_min, en_list


def get_left_trivial_tensor():
    fusionTree = [[-1, -2, -3]]
    fusionTreeDirections = [+1]

    listOfOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None}
    ]
    listOfDegeneracyTesors = [np.reshape(np.array([1]), (1, 1, 1))]
    listOfChargeSectors = [[0, 0, 0]]

    nameOrdering = [-1, -2, -3]

    return SU2Tensor(fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirections,
                     listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTesors,
                     listOfChargeSectors=listOfChargeSectors, nameOrdering=nameOrdering)


def get_right_trivial_tensor():
    fusionTree = [[-1, -2, -3]]
    fusionTreeDirections = [-1]

    listOfOpenEdges = [
        {'edgeName': -1, 'edgeNumber': 1, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -2, 'edgeNumber': 2, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None},
        {'edgeName': -3, 'edgeNumber': 3, 'edgeIrreps': [(0, 1)], 'isFused': False, 'originalIrreps': None}
    ]
    listOfDegeneracyTesors = [np.reshape(np.array([1]), (1, 1, 1))]
    listOfChargeSectors = [[0, 0, 0]]

    nameOrdering = [-1, -2, -3]

    return SU2Tensor(fusionTree=fusionTree, fusionTreeDirections=fusionTreeDirections,
                     listOfOpenEdges=listOfOpenEdges, listOfDegeneracyTensors=listOfDegeneracyTesors,
                     listOfChargeSectors=listOfChargeSectors, nameOrdering=nameOrdering)


def compute_right_operator_blocks(psi, op):
    """
    Compute all partial contractions from the right.
    """
    L = psi.nsites
    assert L == op.nsites
    BR = [None for _ in range(L)]
    # initialize rightmost dummy block
    BR[-1] = get_right_trivial_tensor()
    for i in reversed(range(1, L)):
        psi.A[i-1], psi.A[i] = make_right_orthonormal(A_left=psi.A[i-1], A_target=psi.A[i])
    for i in reversed(range(L-1)):
        BR[i] = contract_right_block(psi.A[i+1], op.A[i+1], BR[i+1])
    return BR


def contract_left_block(A, H, L):
    """
    Contraction step from left to right, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

      _________           _____
     /         \         /     \
     |       -3|---   ---|1 A 2|---
     |         |         \__0__/
     |         |            |
     |         |
     |         |          __|__
     |         |         /  1  \
     |    L  -2|---   ---|2 W 3|---
     |         |         \__0__/
     |         |            |
     |         |
     |         |          __|__
     |         |         /  0  \
     |       -1|---   ---|1 A*2|---
     \_________/         \_____/
    """
    Astar = conjugate_singe_fusionNode_tensor(A=A)

    output = SU2Tensor.einsum(su2tensor1=L, subscript1=(0, 1, 2), su2tensor2=H, subscript2=(3, 1, 4, 5))

    output.perform_f_move([[-1, 2, -2], [-3, 2, 1]])
    output.perform_f_move([[1, -4, -5], [2, 1, -2]])

    output = output.einsum(su2tensor1=output, subscript1=(0, 1, 2, 3, 4), su2tensor2=A, subscript2=(4, 1, 5))
    output = output.einsum(su2tensor1=Astar, subscript1=(0, 1, 2), su2tensor2=output, subscript2=(2, 1, 3, 4))

    return output


def contract_right_block(A, W, R):
    """
    Contraction step from right to left, with a matrix product operator
    sandwiched in between.

    To-be contracted tensor network::

          _____           _________
         /     \         /         \
      ---|1 A 2|---   ---|0        |
         \__0__/         |         |
            |            |         |
                         |         |
          __|__          |         |
         /  1  \         |         |
      ---|2 W 3|---   ---|1   R    |
         \__0__/         |         |
            |            |         |
                         |         |
          __|__          |         |
         /  0  \         |         |
      ---|1 A*2|---   ---|2        |
         \_____/         \_________/
    """
    Astar = conjugate_singe_fusionNode_tensor(A=A)
    output = SU2Tensor.einsum(su2tensor1=A, subscript1=(0, 1, 2), su2tensor2=R, subscript2=(3, 2, 4))

    output.perform_f_move([node for node in output.fusionTree if 1 in node])

    output = SU2Tensor.einsum(su2tensor1=W, subscript1=(0, 1, 2, 3), su2tensor2=output, subscript2=(3, 4, 2, 5))

    output.perform_f_move([node for node in output.fusionTree if 1 in node])

    Astar.reverse_left_most_leg(reverseIrrepName=-2)
    output.reverse_left_most_leg(reverseIrrepName=-1)

    output = SU2Tensor.einsum(su2tensor1=output, subscript1=(0, 1, 2, 3), su2tensor2=Astar, subscript2=(3, 0, 4))

    for idx, deg in enumerate(output.listOfDegeneracyTensors):
        output.listOfDegeneracyTensors[idx] = (-1) * deg

    return output


def contract_right_block_verify(A, W, R):
    from su2tn.algorithms.lanczos_method import perform_einsum_on_explicit_tensor_blocks
    Astar = conjugate_singe_fusionNode_tensor(A=A)

    A_explicit = A.return_explicit_tensor_blocks()
    W_explicit = W.return_explicit_tensor_blocks()
    Astar_explicit = Astar.return_explicit_tensor_blocks()
    R_explicit = R.return_explicit_tensor_blocks()

    output = perform_einsum_on_explicit_tensor_blocks(A1=A_explicit, subscript1=(0, 1, 2),
                                                      A2=R_explicit, subscript2=(3, 2, 4))

    output = perform_einsum_on_explicit_tensor_blocks(A1=W_explicit, subscript1=(0, 1, 2, 3),
                                                      A2=output, subscript2=(3, 4, 2, 5))

    output = perform_einsum_on_explicit_tensor_blocks(A1=output, subscript1=(0, 1, 2, 3),
                                                      A2=Astar_explicit, subscript2=(3, 0, 4))

    return output


def construct_local_hamiltonian(L, H, R):
    """
    Construct the two-site local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor of degree 8)::

                    6       4               5        7
      _________     |          |        |    _________
     /         \    |          |        |   /         \
     |       -3|---/           |        \---|-2       |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |             __|__          |         |
     |         |            / -4  \         |         |
     |    L  -2|---      ---|-2W-3|---   ---|-1  R    |
     |         |            \_-1__/         |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |       -1|---\           |        /---|-3       |
     \_________/    |          |       |    \_________/
                    |          |       |
                    2       0               1       3
    """
    output = SU2Tensor.einsum(su2tensor1=L, subscript1=(0, 1, 2), su2tensor2=H, subscript2=(3, 1, 4, 5))

    output.perform_f_move([[-1, 2, -2], [-3, 2, 1]])

    R.perform_permutation(irrep1Name=-1, irrep2Name=-2)

    output = SU2Tensor.einsum(su2tensor1=output, subscript1=(0, 1, 2, 3, 4), su2tensor2=R, subscript2=(5, 4, 6))

    output.perform_f_move([[-5, 3, -6], [2, 3, -4]])

    output.perform_f_move([[-5, 2, 3], [1, 2, -2]])

    output.perform_permutation(irrep1Name=-5, irrep2Name=-3)
    output.perform_permutation(irrep1Name=-5, irrep2Name=-1)

    output.perform_permutation(irrep1Name=-6, irrep2Name=-4)
    output.perform_permutation(irrep1Name=-6, irrep2Name=-2)

    output.perform_f_move([node for node in output.fusionTree if 1 in node])

    output.reverse_right_most_leg(reverseIrrepName=-1)

    output.perform_permutation(irrep1Name=-6, irrep2Name=-1)

    output.perform_f_move([node for node in output.fusionTree if 3 in node])

    output.reverse_right_most_leg(reverseIrrepName=-6)

    output.fuse_neighboring_legs(irrep1Name=-3, irrep2Name=-1)
    output.fuse_neighboring_legs(irrep1Name=-3, irrep2Name=-6)

    output.fuse_neighboring_legs(irrep1Name=-2, irrep2Name=-4)
    output.fuse_neighboring_legs(irrep1Name=-5, irrep2Name=-2)

    return output


def construct_local_hamiltonian_explicit(L:SU2Tensor, H: SU2Tensor, R: SU2Tensor):
    """
    Construct the two-site local Hamiltonian operator.

    To-be contracted tensor network (the indices at the open legs
    show the ordering for the output tensor of degree 8)::

                    6       4               5        7
      _________     |          |        |    _________
     /         \    |          |        |   /         \
     |       -3|---/           |        \---|-2       |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |             __|__          |         |
     |         |            / -4  \         |         |
     |    L  -2|---      ---|-2W-3|---   ---|-1  R    |
     |         |            \_-1__/         |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |         |               |            |         |
     |       -1|---\           |        /---|-3       |
     \_________/    |          |       |    \_________/
                    |          |       |
                    2       0               1       3
    """
    L_exp = L.return_explicit_tensor_blocks()
    H_exp = H.return_explicit_tensor_blocks()
    R_exp = R.return_explicit_tensor_blocks()

    comb_cs_list = []
    comb_tensor_list = []
    for L_cs, L_tensor in L_exp.items():
        for H_cs, H_tensor in H_exp.items():
            if H_cs[1] == L_cs[1]:
                comb_cs_list.append([L_cs[0], L_cs[2]] + [H_cs[0], H_cs[2], H_cs[3]])
                comb_tensor = np.einsum(L_tensor, (0, 1, 2), H_tensor, (3, 1, 4, 5))
                comb_tensor_list.append(comb_tensor)

    new_cs_list = []
    new_tensor_list = []
    for comb_cs, comb_tensor in zip(comb_cs_list, comb_tensor_list):
        new_cs_part = comb_cs.copy()
        new_cs_part.pop(3)
        for R_cs, R_tensor in R_exp.items():
            if R_cs[0] == comb_cs[3]:
                new_cs = new_cs_part + [R_cs[1], R_cs[2]]
                new_cs = [new_cs[i] for i in (0, 2, 5, 1, 3, 4)]
                # make sure that the hamiltonian is a square matrix
                if (new_cs[0] == new_cs[3]) and (new_cs[1] == new_cs[4]) and (new_cs[2] == new_cs[5]):
                    new_cs_list.append(new_cs)
                    new_tensor = np.einsum(comb_tensor, (0, 1, 2, 3, 4), R_tensor, (3, 5, 6), (0, 2, 6, 1, 4, 5))
                    new_tensor_list.append(new_tensor)

    for idx, tensor in enumerate(new_tensor_list):
        tensor = np.reshape(tensor,
                            (tensor.shape[0]*tensor.shape[1]*tensor.shape[2],
                             tensor.shape[3]*tensor.shape[4]*tensor.shape[5]))

        assert np.allclose(tensor, np.transpose(np.conj(tensor)))
    return {tuple(new_cs_list[i]): new_tensor_list[i] for i in range(len(new_cs_list))}


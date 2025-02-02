import numpy as np
from su2tn.algorithms.two_site_dmrg_util import merge_two_MPS_tensors, split_MPS_tensor, merge_two_MPO_tensors

from su2tn.algorithms.dmrg import (get_left_trivial_tensor,  compute_right_operator_blocks,
                                   contract_left_block, contract_right_block)

from su2tn.algorithms.lanczos_method import eigh_krylov


def two_site_dmrg(H, psi, numsweeps, lanczos_numiter, return_initial_energy=True, tol=1e-4, abort_condition=1e-4, verify_block_calculation=False):
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
        from su2tn.classical_benchmark.standard_MPO import MPO as standard_MPO
        from su2tn.classical_benchmark.standard_heisenberg import construct_heisenberg_mpo
        from su2tn.models.MPS_utils.MPS_util import get_explizit_tensor_state
        H_mat = standard_MPO(construct_heisenberg_mpo(L=psi.nsites, J=-(1 / 4), pbc=False)).as_matrix()
        init_vec = np.reshape(get_explizit_tensor_state(psi), int(2 ** psi.nsites))
        print(np.linalg.norm(init_vec))
        init_vec = init_vec / np.linalg.norm(init_vec)
        init_en = init_vec @ H_mat @ init_vec
        print('Initial energy ', init_en)

    # Number of iterations should be determined by tolerance and some convergence measure
    for n in range(numsweeps):
        en = 0
        sweep_list = []

        # sweep from left to right (rightmost two lattice sites are handled by right-to-left sweep)
        for i in range(L - 1):
            print('step: ', i, 'optimize', i, i+1)
            # Apremege = SU2Tensor.einsum(psi.A[i], (0, 1, 2), psi.A[i+1], (3, 2, 4))
            # print('Apremege')
            # for deg in Apremege.listOfDegeneracyTensors:
            #     print(np.round(np.real(deg), decimals=3))

            Hloc = merge_two_MPO_tensors(H1=H.A[i], H2=H.A[i+1])
            MPSmerge, fuse_info = merge_two_MPS_tensors(A1=psi.A[i], A2=psi.A[i+1])
            en, Aloc = eigh_krylov(vstart=MPSmerge, numiter=lanczos_numiter, H=Hloc, L=BL[i], R=BR[i+1])
            # aloctest = deepcopy(Aloc)
            psi.A[i], psi.A[i+1] = split_MPS_tensor(Aloc, mode='left', fuse_info=fuse_info, tol=tol)
            #verify_split_two_MPS_tensors(Ainitial=aloctest, Aleft=psi.A[i], Aright=psi.A[i+1])

            # update the left blocks
            BL[i+1] = contract_left_block(psi.A[i], H.A[i], BL[i])
            # if i == L-3:
            #     BL[L-1] = contract_left_block(psi.A[L-2], H.A[L-2], BL[L - 2])

            print(en)
            sweep_list.append(en)
            # test = np.reshape(get_explizit_tensor_state(psi), int(2 ** L))
            # print(np.linalg.norm(test))

        # sweep from right to left
        for i in reversed(range(1, L)):
            print('step: ', i, 'optimize', i, i-1)

            Hloc = merge_two_MPO_tensors(H1=H.A[i-1], H2=H.A[i])
            MPSmerge, fuse_info = merge_two_MPS_tensors(A1=psi.A[i-1], A2=psi.A[i])
            en, Aloc = eigh_krylov(vstart=MPSmerge, numiter=lanczos_numiter, H=Hloc, L=BL[i-1], R=BR[i])
            psi.A[i-1], psi.A[i] = split_MPS_tensor(Aloc, mode='right', fuse_info=fuse_info, tol=tol)

            # update the right blocks
            BR[i-1] = contract_right_block(psi.A[i], H.A[i], BR[i])
            print(en)
            sweep_list.append(en)

            # test = np.reshape(get_explizit_tensor_state(psi), int(2 ** L))
            # print(np.linalg.norm(test))

        en_list.append(sweep_list)
        # record energy after each sweep
        en_min[n] = en

        if abort_condition is None:
            pass
        elif n >= 1 and en_min[-2] - en_min[-1] < abort_condition:
            print('Algorithm converged')
            break
        print("sweep {} completed, current energy: {}".format(n + 1, en))

    return init_en, en_min, en_list

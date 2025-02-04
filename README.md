# su2tn - A Python-based framework for $SU(2)$-symmetric tensors
Exploiting symmetries in tensor networks and their benefits have been broadly demonstrated for the numerical simulation of strongly correlated quantum systems. Abelian symmetries are widely used to decrease computational cost and saving capacities due to their simpler structure, which can be straightforwardly implemented into many existing algorithms. On the other hand, non-abelian symmetries feature a richer internal structure where many subtleties arise when introducing them to tensor networks. Therefore, this package gives insights into where and why these subtleties occur and how tensor network algorithms can be adjusted to them. For this, this thesis focuses mainly on the global $SU(2)$ symmetry, an example of a non-trivial, non-abelian symmetry that is quite common in nature. Many of the concepts explained here can nevertheless be used for other, more exotic symmetries. 

The main benefit of using symmetries stems from decomposing a tensor into blocks where each block is described by certain quantum numbers of the symmetry, like particle number or spin. In addition, non-abelian symmetries allow us to further split each block into a degeneracy part that holds the degrees of freedom and a non-trivial structural part that is completely identified by the symmetry. As the structural tensors can quickly become larger tensor objects, we eliminate the need to save them explicitly in favor of a tree structure. 

We will not restrict ourselves to a specific tensor network algorithm but rather aim to implement a generic data structure for symmetric tensors that also scales for higher-degree tensors. In addition to the data structure, we implement an elementary set of tensor operations that are the building blocks of most tensor network algorithms.

Installation
------------
To install *su2tn*, clone this repository and install it in development mode via


    python3 -m pip install -e <path/to/repo>


References
----------
- | Sukhwinder Singh, Guifre Vidal
  | Tensor network states and algorithms in the presence of a global SU(2) symmetry
  | Phys. Rev. B 86, 195114 (2012)
  | `Phys. Rev. B 86, 195114 (2012) <https://doi.org/10.1103/PhysRevB.86.195114>`_
- | Philipp Schmoll, Sukhbinder Singh, Matteo Rizzi, Román Orús
  | A programming guide for tensor networks with global SU(2) symmetry
  | `Ann. Phys. 419, 168232 (2020) <https://doi.org/10.1016/j.aop.2020.168232>`_
- | A. I. Tóth, C. P. Moca, Ö. Legeza, G. Zaránd
  | Density matrix numerical renormalization group for non-Abelian symmetries
  | `Phys. Rev. B 78, 245109 (2008) <https://doi.org/10.1103/PhysRevB.78.245109>`_

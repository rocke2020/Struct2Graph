import networkx as nx
from rdkit import Chem
import numpy as np
from collections import Counter
import os
import sys

sys.path.append(os.path.abspath("."))
from utils_comm.log_util import ic, logger


def smiles_to_nx(smiles, verbose=0):
    mol = Chem.MolFromSmiles(smiles)
    G = nx.Graph()
    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(), label=atom.GetSymbol())
        if verbose:
            ic(atom.GetIdx(), atom.GetSymbol())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        if verbose:
            ic(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return G


def weisfeiler_lehman_step(G):
    new_labels = {}
    for node in G.nodes():
        # node: 16, type(node): <class 'int'>, G.nodes[node]["label"]: 'C'
        ic(node, type(node), G.nodes[node]["label"])
        neighbors_labels = [
            G.nodes[neighbor]["label"] for neighbor in G.neighbors(node)
        ]
        new_label = hash((G.nodes[node]["label"], tuple(sorted(neighbors_labels))))
        new_labels[node] = new_label
    # ic(Counter([G.nodes[node]["label"] for node in G.nodes()]))
    # replace the existent node labels with the new ones
    nx.set_node_attributes(G, new_labels, "label")


def weisfeiler_lehman_hash(G, n_iter=5):
    for _ in range(n_iter):
        weisfeiler_lehman_step(G)
    result = Counter([G.nodes[node]["label"] for node in G.nodes()])
    ic(result)
    return result


def wl_kernel_normalized(graphs, n_iter=5, verbose=1):
    wl_hashes = [weisfeiler_lehman_hash(G, n_iter) for G in graphs]
    n = len(graphs)
    kernel_matrix = np.zeros((n, n))
    normalized_matrix = np.zeros_like(kernel_matrix)
    if verbose:
        ic(n)
    for i in range(n):
        for j in range(n):
            if verbose and ((i == 0 and j == 1) or (i == 1 and j == 0)):
                ic(i, j)
                r = wl_hashes[i] & wl_hashes[j]
                ic(wl_hashes[i], wl_hashes[j], r)
                ic(len(wl_hashes[i]), len(wl_hashes[j]), len(r))
            # __and__ is the intersection of two Counter objects, which is the minimum 
            # of the two counts
            kernel_matrix[i, j] = sum((wl_hashes[i] & wl_hashes[j]).values())

    # Normalize the kernel matrix
    for i in range(n):
        for j in range(n):
            norm_factor = np.sqrt(kernel_matrix[i, i] * kernel_matrix[j, j])
            normalized_matrix[i, j] = (
                kernel_matrix[i, j] / norm_factor if norm_factor != 0 else 0
            )
    return normalized_matrix


smiles_list = [
    "CC(C)CCCC(C)C1CCC2C1(CCCC2=CC=C3CC(CCC3=C)O)C",
    "CCC(C)C(=O)OC1CC(C=C2C1C(C(C=C2)C)CCC3CC(CC(=O)O3)O)C",
    "CC(C)C1=C(C(=C(N1CCC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",
    "O=C4CC[C@@]3([C@@H]1[C@H]([C@H]2[C@](CC1)(CCC2)C)CCC3=C4)C",
]

graphs = [smiles_to_nx(smiles) for smiles in smiles_list]
km_norm = wl_kernel_normalized(graphs, n_iter=2)
# ## Print the matrix
ic(km_norm)

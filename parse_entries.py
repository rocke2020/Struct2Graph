# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:25:40 2020
@author: mayank
"""

import logging
import os
import pickle
from collections import defaultdict
from os import walk
from pathlib import Path

import numpy as np
from Bio import SeqIO
from numpy import linalg as LA

from utils_comm.log_util import DATE_FORMAT, FMT, ic


def get_logger(name=None, log_file=None, log_level=logging.DEBUG):
    """default log level DEBUG"""
    _logger = logging.getLogger(name)
    logging.basicConfig(format=FMT, datefmt=DATE_FORMAT)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, "w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FMT, DATE_FORMAT))
        _logger.addHandler(fh)
    _logger.setLevel(log_level)
    return _logger


logger = get_logger()

all_amino = []

max_residues = 2000

amino_list = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "PYL",
    "SER",
    "SEC",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASX",
    "GLX",
    "XAA",
    "XLE",
]
logger.info("len(amino_list) %s", len(amino_list))
amino_short = {}
amino_short["ALA"] = "A"
amino_short["ARG"] = "R"
amino_short["ASN"] = "N"
amino_short["ASP"] = "D"
amino_short["CYS"] = "C"
amino_short["GLN"] = "Q"
amino_short["GLU"] = "E"
amino_short["GLY"] = "G"
amino_short["HIS"] = "H"
amino_short["ILE"] = "I"
amino_short["LEU"] = "L"
amino_short["LYS"] = "K"
amino_short["MET"] = "M"
amino_short["PHE"] = "F"
amino_short["PRO"] = "P"
amino_short["PYL"] = "O"
amino_short["SER"] = "S"
amino_short["SEC"] = "U"
amino_short["THR"] = "T"
amino_short["TRP"] = "W"
amino_short["TYR"] = "Y"
amino_short["VAL"] = "V"
amino_short["ASX"] = "B"
amino_short["GLX"] = "Z"
amino_short["XAA"] = "X"
amino_short["XLE"] = "J"


def create_fingerprints(amino_acids, adjacency, radius):
    """Extract r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using WeisfeilerLehman-like algorithm."""
    fingerprints = []
    if (len(amino_acids) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in amino_acids]
    else:
        logger.info(
            "len(amino_acids) %s, %s, %s",
            len(amino_acids),
            amino_acids[:5],
            amino_acids[-5:],
        )
        log_existent_fingerprint = True
        for i in range(len(amino_acids)):
            vertex = amino_acids[i]
            # Most adjacency value is > 0.0001.
            neighbors = sorted(amino_acids[np.where(adjacency[i] > 0.0001)[0]])
            # 14, [0, 1, 2, 8, 9, 9, 20, 20, 20, 20, 20, 20, 20, 20]
            if i == 0:
                logger.debug("%s, %s", len(neighbors), neighbors[:20])
            neighbors = tuple(set(neighbors))
            # 6, (0, 1, 2, 8, 9, 20)
            fingerprint = (vertex, neighbors)
            if i == 0:
                logger.debug("%s, %s", len(neighbors), neighbors[:20])
                # fingerprint (6, (0, 1, 2, 3, 6, 7, 8, 13, 14, 18)) in existent dict
            if log_existent_fingerprint and fingerprint in fingerprint_dict:
                logger.debug("fingerprint %s in existent dict", fingerprint)
                log_existent_fingerprint = False
            fingerprints.append(fingerprint_dict[fingerprint])
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # logger.info('%s', fingerprints[:20])
    return np.array(fingerprints)


def create_amino_acids(acids):
    aa_indices = [
        (
            acid_dict[acid_name]
            if acid_name in amino_list
            else acid_dict["MET"] if acid_name == "FME" else acid_dict["TMP"]
        )
        for acid_name in acids
    ]
    aa_indices = np.array(aa_indices)
    # ic(len(aa_indices), len(np.unique(aa_indices)))
    return aa_indices


def dump_dictionary(dictionary, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(dict(dictionary), f)


def is_empty_pdb(pdb_id):

    empty = False

    with open(pdb_id + ".pdb", "r") as f:
        for ln in f:
            if ln.startswith("<html>"):
                empty = True
                break
    return empty


def replace_pdb(pdb_id):
    """TODO In my test, wrong to replace DBREF1, DBREF2, DBREF3, DBREF4, as DBREF
    Such as 3jcm.pdb.
    '"""
    with open(pdb_id + ".pdb", "r") as f:
        filedata = f.read()
        filedata = filedata.replace("DBREF1", "DBREF")
        filedata = filedata.replace("DBREF2", "DBREF")
        filedata = filedata.replace("DBREF3", "DBREF")
        filedata = filedata.replace("DBREF4", "DBREF")
        filedata = filedata.replace("DBREF5", "DBREF")
        filedata = filedata.replace("\\'", "P")

    with open(pdb_id + ".pdb", "w") as f:
        f.write(filedata)


def parse_PDB(pdb_name, uniprot_id, user_chain):

    without_chain = False
    # Bad logic to use try except here!
    try:
        if not user_chain == "0":
            for record in SeqIO.parse(pdb_name + ".pdb", "pdb-seqres"):
                pdb_id = record.id.strip().split(":")[0]
                chain = record.annotations["chain"]
                # some pdb has no DBREF row and so no dbxrefs
                # if len(record.dbxrefs):
                _, UNP_id = record.dbxrefs[0].strip().split(":")

                if UNP_id == uniprot_id:
                    if chain == user_chain:
                        break

            if not chain:
                chain = user_chain
        else:
            chain = user_chain
            without_chain = True
    except Exception as e:
        logger.warning("Error in %s, %s, %s\n%s", pdb_name, uniprot_id, user_chain, e)
        # raise RuntimeError() from e
        chain = user_chain

    with open(pdb_name + ".pdb", "r") as fi:
        mdl = False
        for ln in fi:
            if ln.startswith("NUMMDL"):
                mdl = True
                break

    with open(pdb_name + ".pdb", "r") as fi:
        id = []

        if mdl:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)
                elif ln.startswith("ENDMDL"):
                    break
        else:
            for ln in fi:
                if ln.startswith("ATOM") or ln.startswith("HETATM"):
                    id.append(ln)

    count = 0
    seq = {}
    seq["type_atm"], seq["ind"], seq["amino"], seq["group"], seq["coords"] = (
        [],
        [],
        [],
        [],
        [],
    )

    for element in id:
        type_atm = element[0:6].strip().split()[0]
        ind = int(element[6:12].strip().split()[0])
        atom = element[12:17].strip().split()[0]
        amino = element[17:21].strip().split()[0]
        chain_id = element[21]
        group_id = int(element[22:26].strip().split()[0])
        x_coord = float(element[30:38].strip().split()[0])
        y_coord = float(element[38:46].strip().split()[0])
        z_coord = float(element[46:54].strip().split()[0])

        coords = np.array([x_coord, y_coord, z_coord])

        if not without_chain:
            if chain_id == chain:
                seq["type_atm"].append(type_atm)
                seq["ind"].append(int(ind))
                seq["amino"].append(amino)
                seq["group"].append(int(group_id))
                seq["coords"].append(coords)

                count += 1
        else:
            seq["type_atm"].append(type_atm)
            seq["ind"].append(int(ind))
            seq["amino"].append(amino)
            seq["group"].append(int(group_id))
            seq["coords"].append(coords)

            count += 1

    return seq["type_atm"], seq["amino"], seq["group"], seq["coords"], chain


def unique(list1):
    x = np.array(list1)
    return np.unique(x)


def group_by_coords(group, amino, coords):
    """
    Returns:
        group_coords: 3D coordinates of the groups, (n, 3) ndarray
        group_amino:  amino acid of the groups
    """
    uniq_group = np.unique(group)
    group_coords = np.zeros((uniq_group.shape[0], 3))

    group_amino = []

    np_group = np.array(group)

    for i, e in enumerate(uniq_group):
        inds = np.where(np_group == e)[0]
        group_coords[i, :] = np.mean(np.array(coords)[inds], axis=0)
        group_amino.append(amino[inds[0]])

    return group_coords, group_amino


def get_graph_from_struct(group_coords, group_amino):
    """

    Returns:
        residue_type: list of amino acid types

        adjacency: (n, n) ndarray, the adjacency matrix of the graph
        [0.33333333 0.16666667 0.12909944 0.12909944 0.12309149 0. 0. 0. 0. 0. ]
        ...
        [0. 0. 0. 0. 0. 0.07106691 0.09534626 0.07537784 0.07106691 0.18181818]
    """
    num_residues = group_coords.shape[0]

    if num_residues > max_residues:
        num_residues = max_residues

    residues = group_amino[:num_residues]

    retval = [[0 for i in range(0, num_residues)] for j in range(0, num_residues)]

    residue_type = []
    for i in range(0, num_residues):
        if residues[i] == "FME":
            residues[i] = "MET"
        elif residues[i] not in amino_list:
            residues[i] = "TMP"

        residue_type.append(residues[i])

        for j in range(i + 1, num_residues):
            x, y = group_coords[i], group_coords[j]
            retval[i][j] = LA.norm(x - y)
            retval[j][i] = retval[i][j]

    retval = np.array(retval)

    threshold = 9.5

    for i in range(0, num_residues):
        for j in range(0, num_residues):
            if retval[i, j] <= threshold:
                retval[i, j] = 1
            else:
                retval[i, j] = 0

    n = retval.shape[0]
    adjacency = retval + np.eye(n)
    degree = sum(adjacency)  # bad code to read
    # ic(degree.shape, degree[:10], degree[-10:])
    # degree = np.sum(adjacency, axis=0)
    d_half = np.sqrt(np.diag(degree))
    # ic(d_half.shape, d_half[0][:10], d_half[-1][-10:])
    d_half_inv = np.linalg.inv(d_half)
    # ic(d_half_inv.shape, d_half_inv[0][:10], d_half_inv[-1][-10:])

    adjacency = np.matmul(d_half_inv, np.matmul(adjacency, d_half_inv))
    logger.debug(
        "adjacency[0]...[-1]\n%s\n...\n%s",
        adjacency[0][:10],
        adjacency[-1][-10:],
    )
    return residue_type, np.array(adjacency)


with open("list_of_prots.txt", "r") as f:
    data_list = f.read().strip().split("\n")

pdb_to_uniprot_dict, pdb_to_chain_dict = {}, {}
for data in data_list:
    pdb = data.strip().split("\t")[1]
    pdb_to_uniprot_dict[pdb] = data.strip().split("\t")[0]
    pdb_to_chain_dict[pdb] = data.strip().split("\t")[2]


radius = 1

atom_dict = defaultdict(lambda: len(atom_dict))
acid_dict = defaultdict(lambda: len(acid_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))

filepath = "pdb_files/"
pdb_files = list(Path(filepath).glob("*.pdb"))
ic(len(pdb_files))

_dir_input = Path(filepath) / ("input" + str(radius))
_dir_input.mkdir(exist_ok=True)
dir_input = "input" + str(radius) + "/"


f = []
for dirpath, dirnames, filenames in walk(filepath):
    ic(dirpath, dirnames, filenames[:5], filenames[-5:], len(filenames))
    f.extend(filenames)
    break
os.chdir(filepath)

pdb_ids = []
for data in f:
    pdb_ids.append(data.strip().split(".")[0])
num_prots = len(pdb_ids)

count = 0
q_count = 0

adjacencies, proteins, pnames, pseqs = [], [], [], []
total_aas_num = 0

for n in range(num_prots):
    # if n+1 < 1334: continue
    if is_empty_pdb(pdb_ids[n]):
        continue
    replace_pdb(pdb_ids[n])
    pdb_name = pdb_ids[n]
    uniprot_id = pdb_to_uniprot_dict[pdb_ids[n]]
    user_chain = pdb_to_chain_dict[pdb_ids[n]]

    logger.info("pdb_name %s, %s", pdb_name, "/".join(map(str, [n + 1, num_prots])))

    try:
        type_atm, amino, group, coords, chain = parse_PDB(
            pdb_name, uniprot_id, user_chain
        )
        group_coords, group_amino = group_by_coords(group, amino, coords)
        residue_type, adjacency = get_graph_from_struct(group_coords, group_amino)
        amino_acids = create_amino_acids(residue_type)
        total_aas_num += len(amino_acids)
        fingerprints = create_fingerprints(amino_acids, adjacency, radius)
        # break  # debug
        adjacencies.append(adjacency)
        proteins.append(fingerprints)
        pnames.append(pdb_name)

        d_seq = {}
        for no, g in enumerate(group):
            if g not in d_seq.keys():
                d_seq[g] = amino[no]

        seq_pr = ""
        for k in d_seq:
            if d_seq[k] in amino_list:
                seq_pr += amino_short[d_seq[k]]

        pseqs.append(seq_pr)

        count += 1
        if count % 10 == 0 or n == num_prots - 1:
            # {'ASP': 0, 'THR': 1, 'GLU': 2, 'ARG': 3, 'ALA': 4, 'TRP': 5, 'LEU': 6,
            # 'ASN': 7, 'LYS': 8, 'VAL': 9, 'HIS': 10, 'MET': 11, 'PRO': 12,
            # 'PHE': 13, 'ILE': 14, 'CYS': 15, 'GLN': 16, 'GLY': 17, 'SER': 18,
            # 'TYR': 19, 'TMP': 20}
            if count < 100:
                logger.debug("acid_dict: %s", acid_dict)
            proteins = np.asarray(proteins, dtype=object)
            adjacencies = np.asarray(adjacencies, dtype=object)
            pnames = np.asarray(pnames, dtype=object)
            pseqs = np.asarray(pseqs, dtype=object)
            np.save(
                dir_input
                + "proteins_"
                + str(10 * q_count + 1)
                + "_"
                + str(10 * (q_count + 1)),
                proteins,
            )
            np.save(
                dir_input
                + "adjacencies_"
                + str(10 * q_count + 1)
                + "_"
                + str(10 * (q_count + 1)),
                adjacencies,
            )
            np.save(
                dir_input
                + "names_"
                + str(10 * q_count + 1)
                + "_"
                + str(10 * (q_count + 1)),
                pnames,
            )
            np.save(
                dir_input
                + "seqs_"
                + str(10 * q_count + 1)
                + "_"
                + str(10 * (q_count + 1)),
                pseqs,
            )
            adjacencies, proteins, pnames, pseqs = [], [], [], []
            q_count += 1

    except Exception as identifier:
        logger.warning(
            "Error in %s, %s, %s\n%s", pdb_name, uniprot_id, user_chain, identifier
        )
        raise RuntimeError() from identifier
logger.info("total_aas_num %s", total_aas_num)
dump_dictionary(fingerprint_dict, dir_input + "fingerprint_dict.pickle")
logger.info("Length of fingerprint dictionary: " + str(len(fingerprint_dict)))

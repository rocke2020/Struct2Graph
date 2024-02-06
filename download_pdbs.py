# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:03:59 2020

@author: mayank
"""
import shutil
from pathlib import Path
import requests
import time
import os
from utils_comm.log_util import ic, logger


def fetch_pdb_file(pdb_id):
    the_url = "https://files.rcsb.org/download/" + pdb_id
    page = requests.get(the_url)
    pdb_file = str(page.content)
    # pdb_file = page.content.decode('utf-8')
    pdb_file = pdb_file.replace("\\n", "\n")
    return pdb_file

# unique pdb file num is 3684
with open("list_of_prots.txt", "r") as f:
    data_list = f.read().strip().split("\n")

pdb_list = []

for data in data_list:
    pdb_list.append(data.strip().split("\t")[1])

all_downloade_pdb_dir = Path("/mnt/nas/bio_drug_corpus/pdb/pdb_files")
all_known_pdb_files = [file.name for file in all_downloade_pdb_dir.glob("*.pdb")]
logger.info(
    "len(all_known_pdb_files) %s, %s",
    len(all_known_pdb_files),
    all_known_pdb_files[:10],
)
# os.makedirs('pdb_files/', exist_ok=True)
ctr = 0
for pdb_id in pdb_list:
    pdb_filename = pdb_id + ".pdb"
    file_path = "pdb_files/" + pdb_filename
    if Path(file_path).is_file():
        logger.info("Already downloaded " + pdb_filename)
        continue
    if pdb_filename.lower() in all_known_pdb_files:
        shutil.copyfile(all_downloade_pdb_dir / pdb_filename.lower(), file_path)
        logger.info("Copy " + pdb_filename)
    else:
        pdbfile = fetch_pdb_file(pdb_filename)

        logger.info("Writing " + file_path)
        with open(file_path, "w") as fd:
            fd.write(pdbfile)
    ctr += 1
    if ctr % 1000 == 0:
        time.sleep(60)

logger.info("All done!")

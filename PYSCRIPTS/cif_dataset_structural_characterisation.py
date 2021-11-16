import sys
sys.path.append('../CrystalAnalyser')
from Core import *

import os
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cif_directory',type=str)
    parser.add_argument('--output_directory',type=str)
    parser.add_argument('--occupancy_tolerance',type=int,default=1)
    parser.add_argument('--minimum_molecules_supercell',type=int,default=100)
    parser.add_argument('--n_splits',type=int,default=1)
    parser.add_argument('--split_index',type=1,default=1)
    args = parser.parse_args()
    cif_directory = args.cif_directory
    output_directory = args.output_directory
    occupancy_tolerance = args.occupancy_tolerance
    minimum_molecules_supercell = args.minimum_molecules_supercell
    n_splits = args.n_splits
    split_index =args.split_index
    cifs = os.listdir(cif_directory)
    batch_size = int(len(cifs)/n_splits)
    if split_index < n_splits:
        cifs = cifs[batch_size*(split_index-1):batch_size*split_index]
    else:
        cifs = cifs[batch_size*(split_index-1):]
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    if not os.path.exists('tempdir'):
        os.mkdir('tempdir')
    for n in tqdm(range(1,len(cifs)+1),total=len(cifs)):
        cif = cifs[n-1]
        identifier = cif.split('.')[0]
        cif_reader = CifReader(os.path.join(cif_directory,cif),occupancy_tolerance=occupancy_tolerance)
        n_molecules = 0
        sc_size = 2
        while n_molecules < minimum_molecules_supercell:
            supercell = [[sc_size,0,0],[0,sc_size,0],[0,0,sc_size]]
            cif_reader.supercell_to_mol2('tempdir/supercell',supercell)
            


    

if __name__ == '__main__':
    main()

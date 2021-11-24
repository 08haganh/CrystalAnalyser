# CrystalAnalyser Python Package
# CONFIG
from CONFIG import CONFIG, vdw_radii, atomic_mass, atomic_number, rdkit_bond_types
# DEPENDENCIES
import re
import numpy as np
import pandas as pd
import networkx as nx
import numpy.linalg as la
from openbabel import openbabel
from pymatgen.io.cif import CifParser
from pymatgen.io.xyz import XYZ
from rdkit import Chem

############################################# CIFREADER #############################################
class CifReader(CifParser):

    def __init__(self,filename,occupancy_tolerance=1,site_tolerance=0.0001):
        super().__init__(filename,occupancy_tolerance,site_tolerance)
        self.identifier = filename.split('/')[-1].split('.')[0]
        self.cif_dict = self.as_dict()[self.identifier]
        
    def supercell_to_mol2(self,fname,supercell_size,preserve_labelling=True):
        # Can have issues with not writing any bonds to mol2 file
        # however this does not occur often
        name = fname.split('.')[0]
        struc = self.get_structures()[0]
        struc.make_supercell(supercell_size, to_unit_cell=False)
        labels = self.get_new_labels(struc,supercell_size)
        xyzrep = XYZ(struc)
        xyzrep.write_file(f"{name}.xyz")  # write supercell to file
        # Convert supercell to Mol2 format
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "mol2")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, f"{name}.xyz")   # Open Babel will uncompress automatically
        mol.AddHydrogens()
        obConversion.WriteFile(mol, f'{name}.mol2')
        if preserve_labelling:
            self.change_mol2_atom_labels(f'{name}.mol2',labels)
            
    def supercell_to_xyz(self,fname,supercell_size):
        name = fname.split('.')[0]
        struc = self.get_structures()[0]
        struc.make_supercell(supercell_size, to_unit_cell=False)
        xyzrep = XYZ(struc)
        xyzrep.write_file(f"{name}.xyz")  # write supercell to file
        
    def change_mol2_atom_labels(self,filename,new_labels):
        old_file = open(filename,'r').readlines()
        new_file = open(filename,'w')
        atoms = False
        i=0
        for line in old_file:
            stripped = re.sub("\s+", ",", line.strip())
            split = stripped.split(',')
            arr = np.array(split)
            if arr[0] == '@<TRIPOS>ATOM':
                atoms = True
                new_file.write('@<TRIPOS>ATOM\n')
                continue
            if arr[0] == '@<TRIPOS>BOND':
                atoms = False
                new_file.write('@<TRIPOS>BOND\n')
                continue
            if atoms:
                new_arr = arr
                new_arr[1] = new_labels[i]
                i+=1
            else:
                new_arr = arr
            for elem in new_arr:
                new_file.write(f'{elem} ')
            new_file.write('\n')
        new_file.close()
        
    def get_new_labels(self,struc,supercell_size):
        atom_counter = {}
        new_labels = []
        site_dict = struc.as_dict()['sites']
        symops_len = len(self.cif_dict['_symmetry_equiv_pos_site_id'])
        sc_len = supercell_size[0][0]*supercell_size[1][1]*supercell_size[2][2]
        multiplier = symops_len*sc_len
        for i in range(0,int(len(site_dict)/multiplier)):
            label = site_dict[i*multiplier]['label']
            if label not in atom_counter.keys():
                atom_counter[label] = 1
            new_labels.append([f'{label}{atom_counter[label]}']*multiplier)
            atom_counter[label] += 1
        
        return np.array(new_labels).reshape(-1)

############################################# MOL2READER #############################################
class Mol2Reader():
    def __init__(self,path,n_atoms=False,add_rings_as_atoms=False,complete_molecules=False):
        self.path = path
        self.file = open(self.path,'r')
        self.n_atoms = n_atoms
        self.atoms = []
        self.bonds = []
        self.molecules = []
        self.generate_molecules(add_rings_as_atoms,complete_molecules)
        
    def generate_molecules(self,add_rings_as_atoms,complete_molecules):
        tripos_atom = False
        tripos_bond = False
        for line in self.file.readlines():
            arr = self.line_to_array(line)
            if arr[0] == '@<TRIPOS>ATOM':
                tripos_atom = True
                continue
            if arr[0] == '@<TRIPOS>BOND':
                tripos_atom = False
                tripos_bond = True
                continue
            if tripos_atom:
                atom_number = (int(arr[0]))
                atom_label = (str(arr[1]))
                x = (float(arr[2]))
                y = (float(arr[3]))
                z = (float(arr[4]))
                atom_type = (str(arr[5]))
                atom_coordinates = np.array([x,y,z])
                atom_symbol = re.sub("\d+", "",atom_label)
                self.atoms.append(Atom(atom_label,atom_coordinates,atom_symbol,atom_type,atom_number))
            if tripos_bond:
                bond_number = (int(arr[0]))
                bond_atom_number_1 = (int(arr[1]))
                bond_atom_number_2 = (int(arr[2]))
                bond_type = (str(arr[3]))
                bond_atom1 = self.atoms[bond_atom_number_1-1]
                bond_atom2 = self.atoms[bond_atom_number_2-1]
                self.atoms[bond_atom_number_1-1].neighbours.append(self.atoms[bond_atom_number_2-1])
                self.atoms[bond_atom_number_2-1].neighbours.append(self.atoms[bond_atom_number_1-1])
                self.bonds.append(Bond(bond_atom1,bond_atom2,bond_type,bond_number))
    
        #supermolecule = Molecule(self.atoms,self.bonds,add_rings = False,add_rings_as_atoms=False)
        supergraph = nx.Graph()
        supergraph.add_nodes_from(self.atoms)
        supergraph.add_edges_from([(bond.atom1,bond.atom2,{'type':bond.type}) for bond in self.bonds])
        subgraphs = [supergraph.subgraph(c) for c in nx.connected_components(supergraph)]
        # Using n_atoms potentially buggy
        # Will have to have a think as to how to load co-crystals
        if self.n_atoms:
            pass
        else:  
            n_atoms = max([len(subgraph.nodes) for subgraph in subgraphs])
        if not complete_molecules:
            subgraphs = [subgraph for subgraph in subgraphs if len(subgraph.nodes) == n_atoms]
        else:
            subgraphs = subgraphs
        for graph in subgraphs:
            bonds = []
            for edge in graph.edges:
                bonds.append(Bond(edge[0],edge[1],supergraph[edge[0]][edge[1]]['type']))
            mol = Molecule(list(graph.nodes),bonds,add_rings_as_atoms=add_rings_as_atoms)
            self.molecules.append(mol)
        for mol in self.molecules:
            for atom in mol.atoms:
                atom.add_interaction_dict()
         
    def line_to_array(self,line):
        stripped = re.sub("\s+", ",", line.strip())
        split = stripped.split(',')
        arr = np.array(split)
        return arr

############################################# ATOM #############################################
class Atom():
    def __init__(self,atom_label,atom_coordinates,atom_symbol='',atom_type='',atom_index=np.nan):
        self.label = atom_label
        self.coordinates = atom_coordinates
        self.symbol = atom_symbol
        self.type = atom_type
        self.number = atom_index
        self.interaction = False
        self.in_ring = False
        self.neighbours = []
        try:
            self.weight = atomic_mass[self.symbol]
        except:
            self.weight = 0
        try:
            self.number = atomic_number[self.symbol]
        except:
            self.number = 0
        try:
            self.vdw_radii = vdw_radii[self.symbol]
        except:
            self.vdw_radii = 0
        # self.number
    def add_interaction_dict(self):
        self.interaction = InteractionDict(self)   

############################################# RING CENTROID ###########################################        
class RingCentroid(Atom):
    def __init__(self,label,coordinates,symbol='',atom_type='',atom_number=np.nan,plane=False):
        super().__init__(label,coordinates,symbol,atom_type,atom_number)
        self.plane = plane

############################################# BOND #############################################
class Bond():
    def __init__(self,atom1,atom2,bond_type='',bond_number=np.nan):
        self.atom1 = atom1
        self.atom2 = atom2
        self.type = bond_type
        self.atoms = [self.atom1,self.atom2]
        self.in_ring = False
        
    def length(self):
        c1 = self.atom1.coordinates
        c2 = self.atom2.coordinates
        disp = c2 - c1
        return np.sqrt(np.dot(disp,disp))

############################################# MOLECULE #######################################################
class Molecule():
    def __init__(self,atoms,bonds,add_rings=True,add_cogs=True,add_planes=True,add_rings_as_atoms=False,
                 canonicalise_atom_order=True):
        self.atoms = atoms
        self.bonds = bonds
        self.plane = False
        self.cog = False
        self.ring_systems = False
        self.peripheries = False
        self.rings = False
        if add_rings:
            self.add_rings()
        if add_cogs:
            self.add_centre_of_geometry()
        if add_planes:
            self.add_plane()
        if add_rings_as_atoms:
            self.add_rings_as_atoms()
        if canonicalise_atom_order:
            self.canonicalise_atom_order()
              
    ############################################### Cool stuff ###########################################
    def add_rings(self):
        self.rings = []
        self.ring_atoms = nx.cycle_basis(self.to_networkx())
        self.ring_bonds = []
        for ring in self.ring_atoms:
            temp = []
            for bond in self.bonds:
                if np.sum(np.isin(bond.atoms,ring)) == 2:
                    temp.append(bond)
                else:
                    continue 
            self.ring_bonds.append(temp)
        for ring_atoms, ring_bonds in zip(self.ring_atoms,self.ring_bonds):
            for atom in ring_atoms:
                atom.in_ring = True
            for bond in ring_bonds:
                bond.in_ring = True
            ring = Ring(ring_atoms,ring_bonds)
            self.rings.append(ring)
            
    def add_rings_as_atoms(self):
        if not self.rings:
            self.add_rings()
        for ring in self.rings:
            atom_number = len(self.atoms)
            label = f'ring{atom_number}'
            self.atoms.append(ring.to_atom(label,atom_number))
            
    def add_centre_of_geometry(self):
        self.cog = np.average([atom.coordinates for atom in self.atoms],axis=0)
    
    def centre_of_geometry(self):
        return np.average([atom.coordinates for atom in self.atoms],axis=0)
    
    def add_plane(self):
        self.plane = Plane(np.array([atom.coordinates for atom in self.atoms]))
        
    def plane(self):
        return Plane(np.array([atom.coordinates for atom in self.atoms]))
    
    def add_ring_systems(self):
        self.ring_systems = Molecule([atom for ring in self.ring_atoms for atom in ring],
                                          [bond for ring in self.ring_bonds for bond in ring],add_rings_as_atoms=False)
        self.ring_systems = self.ring_systems.get_components()
    
    def add_peripheries(self):
        self.peripheries = Molecule([atom for atom in self.atoms if (not atom.in_ring)],
                                    [bond for bond in self.bonds if (not bond.in_ring)])
        self.peripheries = self.peripheries.get_components()
    
    def add_atom_neighbours(self):
        g = self.to_networkx()
        for atom in self.atoms:
            atom.neighbours = [n for n in g.neighbors(atom)]
            
    def test_planarity(self):
        mol_plane = Plane(np.array(atom.coordinates for atom in self.atoms))
        devs = [mol_plane.point_distance(atom) for atom in self.atoms]
        if np.mean(devs) > 1:
            return False
        else:
            return True

    def get_components(self):
        g = self.to_networkx()
        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        components = []
        for graph in subgraphs:
            bonds = []
            for edge in graph.edges:
                bonds.append(Bond(edge[0],edge[1],g[edge[0]][edge[1]]['type']))
            mol = Molecule(list(graph.nodes),bonds)
            components.append(mol)
        self.components = components
        
        return self.components
        
    def get_unique_components(self):
        g = self.to_networkx()
        subgraphs = [g.subgraph(c) for c in nx.connected_components(g)]
        unique = []
        for i, graph in enumerate(subgraphs):
            if i == 0:
                unique.append(graph)
                continue
            else:
                for un in unique:
                    if nx.isomorphic(un, graph):
                        continue
                    else:
                        unique.append(graph)
        return unique
    
    def canonicalise_atom_order(self):
        atom_labels = np.array([atom.label for atom in self.atoms])
        order = np.argsort(atom_labels)
        self.atoms = np.array(self.atoms)[order].tolist()
    ############################################### Boring IO stuff ###########################################
    def to_edgelist(self):
        # atom1,atom2,edge_attribute
        pass
    
    def to_bond_dataframe(self):
        # bond,atom1,atom2
        bond_dataframe = []
        for bond in self.bonds:
            bond_dataframe.append({'bond':bond,'atom1':bond.atom1,'atom2':bond.atom2})
        return pd.DataFrame(bond_dataframe)
    
    def to_mol2(self):
        pass
    
    def to_xyz(self,fname):
        split = fname.split('.')
        name = split[0] if len(split) == 1 else split[:-1]
        file = open(name+'.xyz','w')
        n_atoms = len([atom for atom in self.atoms])
        file.write(f'{n_atoms}\n')
        for atom in self.atoms:
            x, y, z = atom.coordinates
            if 'ring' in atom.symbol:
                file.write(f'Ti {x} {y} {z}\n')
            else:
                file.write(f'{atom.symbol} {x} {y} {z}\n')
        file.close()

    def to_rdkit(self):
        # adapted from https://github.com/maxhodak/keras-molecules
        mol = Chem.RWMol()
        node_to_idx = {}
        for atom in self.atoms:
            a = Chem.Atom(atom.number)
            idx = mol.AddAtom(a)
            node_to_idx[atom] = idx
        for bond in self.bonds:
            ifirst = node_to_idx[bond.atom1]
            isecond = node_to_idx[bond.atom2]
            print(bond.type, type(bond.type))
            bond_type = rdkit_bond_types[bond.type]
            mol.AddBond(ifirst,isecond,bond_type)

        Chem.SanitizeMol(mol)
        return mol.GetMol()
    
    def to_networkx(self):
        G = nx.Graph()
        G.add_nodes_from(self.atoms)
        G.add_edges_from([(bond.atom1,bond.atom2,{'type':bond.type}) for bond in self.bonds])
        return G
    
################################################# RING ######################################################
class Ring():
    def __init__(self,atoms,bonds):
        self.atoms = atoms
        self.bonds = bonds
        self.type = self.check_aromaticity()
        self.plane = False
        
    def check_aromaticity(self):
        lengths = [bond.length() for bond in self.bonds]
        if np.average(lengths,axis=0) < 1.45:
            return 'aromatic'
        else:
            return 'aliphatic'
    
    def to_atom(self,label,atom_number=np.nan):
        coordinates = np.average([atom.coordinates for atom in self.atoms],axis=0)
        if not self.plane:
            self.add_plane()
        if self.type == 'aromatic':  
            symbol = 'aromatic_ring'
        else:
            symbol = 'aliphatic_ring'
        atom_type = self.type
        return RingCentroid(label,coordinates,symbol,atom_type,atom_number,self.plane)
    
    def add_plane(self):
        self.plane = Plane(np.array([atom.coordinates for atom in self.atoms]))
        
    def plane(self):
        return Plane(np.array([atom.coordinates for atom in self.atoms]))

################################################# SUPERCELL ######################################################
class Supercell():
    def __init__(self,molecules):
        self.molecules = molecules
        self.atom_interactions = pd.DataFrame()
        self.combined_atom_interactions = pd.DataFrame()
        self.geometric_interactions = pd.DataFrame()
        self.molecule_interactions = pd.DataFrame()
    
    def add_atom_interactions(self,central_only=True,atom_distance_cutoff=5.5):
        # Calculates the atomic interactions between atoms in the supercell that are within a cutoff distance
        # this is a very time consuming function, as such only atom interactions around the central molecule is mapped
        # for structures with more than one unique molecular species, central only should be set to False
        if central_only:
            mol1_idxs = []
            mol2_idxs = []
            dists = []
            central_molecule, central_idx = self.get_central_molecule(return_idx=True)
            central_atom_coords = np.array([atom.coordinates for atom in central_molecule.atoms])
            all_atom_coords = []
            for mol in self.molecules:
                all_atom_coords.append(np.array([atom.coordinates for atom in mol.atoms]))
            all_atom_coords = np.array(all_atom_coords)
            for i, mol_coords in enumerate(all_atom_coords):
                temp_dist = []
                for x in range(len(mol_coords)):
                    mol1_idxs += [central_idx]*len(mol_coords)
                    mol2_idxs += [i]*len(mol_coords)
                    disp = mol_coords - central_atom_coords # shape = (n_atoms,3)
                    dist2 = disp[:,0] * disp[:,0] + disp[:,1] * disp[:,1] + disp[:,2] * disp[:,2]
                    dist = np.sqrt(dist2) # shape = (n_atoms)
                    temp_dist.append(dist)
                    mol_coords = np.roll(mol_coords,-1,axis=0)
                dists.append(temp_dist)
            dists = np.array(dists) # shape = (n_molecules,x_atoms,y_atoms) | where y in y_atoms = dist(atom_x_central - atom_y_mol_n)
            # Put distances in order of atom indices
            in_atom_order = np.array([dist.flatten('F') for dist in dists]).reshape(-1)
            d1 = dists.shape[0]
            d2 = dists.shape[1]
            arange = np.arange(d2)
            atom1s = np.concatenate([[x]*d2 for x in range(d2)]*d1)
            atom2s = np.concatenate([np.roll(arange,-x) for x in range(d2)]*d1)
            #atom2s = np.concatenate([[x for x in range(d2)]*d2]*d1)
            # Turn Atom Distances to DataFrame
            data_dict= {'mol1_idx':mol1_idxs,'mol2_idx':mol2_idxs,'atom1_idx':atom1s,'atom2_idx':atom2s,
                        'distances':in_atom_order}
            distances = pd.DataFrame(data_dict)
            distances = distances[distances.mol1_idx != distances.mol2_idx]
            distances = distances.loc[distances.distances <= atom_distance_cutoff]
            distances = distances.iloc[:,:-1].values
            for row in distances:
                mol1_idx = row[0]
                mol2_idx = row[1]
                atom1_idx = row[2]
                atom2_idx = row[3]
                atom1 = self.molecules[mol1_idx].atoms[atom1_idx]
                atom2 = self.molecules[mol2_idx].atoms[atom2_idx]
                interaction = Interaction(atom1,atom2,mol1_idx,mol2_idx,atom1_idx,atom2_idx).to_dict()
                self.atom_interactions = self.atom_interactions.append(pd.DataFrame(interaction,index=[0]))
            self.atom_interactions.set_index(['mol1_idx','mol2_idx'],inplace=True)    
        else:
            mol1_idxs = []
            mol2_idxs = []
            dists = []
            all_atom_coords = []
            for mol in self.molecules:
                all_atom_coords.append(np.array([atom.coordinates for atom in mol.atoms]))
            all_atom_coords = np.array(all_atom_coords)
            for i, mol_coords1 in enumerate(all_atom_coords):
                for j, mol_coords2 in enumerate(all_atom_coords):
                    temp_dist = []
                    for x in range(len(mol_coords2)):
                        mol1_idxs += [i]*len(mol_coords1)
                        mol2_idxs += [j]*len(mol_coords2)
                        disp = mol_coords2 - mol_coords1 # shape = (n_atoms,3)
                        dist2 = disp[:,0] * disp[:,0] + disp[:,1] * disp[:,1] + disp[:,2] * disp[:,2]
                        dist = np.sqrt(dist2) # shape = (n_atoms)
                        temp_dist.append(dist)
                        mol_coords2 = np.roll(mol_coords2,-1,axis=0)
                    dists.append(temp_dist)
            dists = np.array(dists) # shape = (n_molecules,x_atoms,y_atoms) | where y in y_atoms = dist(atom_x_central - atom_y_mol_n)
            # Put distances in order of atom indices
            in_atom_order = np.array([dist.flatten('F') for dist in dists]).reshape(-1)
            d1 = dists.shape[0]
            d2 = dists.shape[1]
            arange = np.arange(d2)
            atom1s = np.concatenate([[x]*d2 for x in range(d2)]*d1)
            atom2s = np.concatenate([np.roll(arange,-x) for x in range(d2)]*d1)
            #atom2s = np.concatenate([[x for x in range(d2)]*d2]*d1)
            # Turn Atom Distances to DataFrame
            data_dict= {'mol1_idx':mol1_idxs,'mol2_idx':mol2_idxs,'atom1_idx':atom1s,'atom2_idx':atom2s,
                        'distances':in_atom_order}
            distances = pd.DataFrame(data_dict)
            distances = distances[distances.mol1_idx != distances.mol2_idx]
            distances = distances.loc[distances.distances <= atom_distance_cutoff]
            distances = distances.iloc[:,:-1].values
            for row in distances:
                mol1_idx = row[0]
                mol2_idx = row[1]
                atom1_idx = row[2]
                atom2_idx = row[3]
                atom1 = self.molecules[mol1_idx].atoms[atom1_idx]
                atom2 = self.molecules[mol2_idx].atoms[atom2_idx]
                interaction = Interaction(atom1,atom2,mol1_idx,mol2_idx,atom1_idx,atom2_idx).to_dict()
                self.atom_interactions = self.atom_interactions.append(pd.DataFrame(interaction,index=[0]))
            self.atom_interactions.set_index(['mol1_idx','mol2_idx'],inplace=True)    
                    
    def add_geometric_interactions(self,functions=[]):
        # populates geometric interactions between all pairs of molecules in the supercell
        # must pass the list of functions 
        for i, mol1 in enumerate(self.molecules[:-1],0):
            for j, mol2 in enumerate(self.molecules[i+1:],i+1):
                info = pd.Series(dtype=object)
                for function in functions:
                    info = info.append(function(i,j))
                info = pd.DataFrame(info).T
                info['mol1_idx'] = i
                info['mol2_idx'] = j
                info.set_index(['mol1_idx','mol2_idx'],inplace=True)
                info = np.round(info,5)
                self.geometric_interactions = self.geometric_interactions.append(info)
    
    def get_central_molecule(self,return_idx=False):
        # returns the molecules closest to the centre of geometry of the supercell
        mol_cogs = [mol.cog for mol in self.molecules]
        cog = self.centre_of_geometry()
        disps = mol_cogs - cog
        distances = np.sqrt(disps[:,0]*disps[:,0] + disps[:,1]*disps[:,1] + disps[:,2]*disps[:,2])
        central_idx = np.argsort(distances)[0]
        central_molecule = self.molecules[central_idx]
        if return_idx:
            return central_molecule, central_idx
        else:
            return central_molecule
                
    def centroid_distance(self,mol1_idx,mol2_idx):
        # calculates the centroid distance between two molecules in the supercell
        info = pd.Series(dtype=object)
        cog1 = self.molecules[mol1_idx].cog
        cog2 = self.molecules[mol2_idx].cog
        disp = cog2 - cog1
        info['x'] = disp[0]
        info['y'] = disp[1]
        info['z'] = disp[2]
        info['centroid_distance'] = np.sqrt(np.dot(disp,disp))
        
        return info
    
    def interplanar_angle(self,mol1_idx,mol2_idx):
        # calculates intperplanar angle between two molecules in the supercell
        info = pd.Series(dtype=object)
        plane1 = self.molecules[mol1_idx].plane
        plane2 = self.molecules[mol2_idx].plane
        angle = plane1.plane_angle(plane2)
        info['interplanar_angle'] = angle
        
        return info
    
    def planar_offset(self,mol1_idx,mol2_idx):
        # calculates projection vector, and vertical and horizontal planar offsets between two molecules
        # in the supercell
        info = pd.Series(dtype=object)
        cog1 = self.molecules[mol1_idx].cog
        cog2 = self.molecules[mol2_idx].cog
        plane1 = self.molecules[mol1_idx].plane
        disp = cog2 - cog1
        distance = np.sqrt(np.dot(disp,disp))
        scaled_disp = disp / distance
        vec_angle = np.radians(vector_angle(disp, np.array([plane1.a,plane1.b,plane1.c])))
        v_offset = distance*np.cos(vec_angle)
        h_offset = distance*np.sin(vec_angle)
        projection = np.dot(plane1.unit_normal(),scaled_disp)
        info['projection'] = np.abs(projection)
        info['vertical_offset'] = np.abs(v_offset)
        info['horizontal_offset'] = np.abs(h_offset)
        
        return info
    
    def quaternion(self):
        # calculates quaturnion between two molecules in the supercell
        pass
        
    def combine_atom_interactions(self,only_unique=False):
        # combines single atom interactions to molecular level
        forms_bonds = ((self.atom_interactions.hydrogen_bond > 0) |
                   (self.atom_interactions.pi_bond > 0) |
                   (self.atom_interactions.halogen_bond > 0) |
                   (self.atom_interactions.ch_pi_bond > 0) |
                   (self.atom_interactions.hydrophobic > 0))
        filtered = self.atom_interactions.loc[forms_bonds]
        filtered = filtered[['vdw_contact','hydrogen_bond','pi_bond','halogen_bond','ch_pi_bond',
                             'hydrophobic']]
        combined_dfs = []
        for idx in filtered.index.unique():
            temp = filtered.loc[idx]
            temp = pd.DataFrame(temp.sum(axis=0)).T 
            index = pd.MultiIndex.from_tuples([idx], names=['mol1_idx','mol2_idx'])
            temp.index = index
            combined_dfs.append(temp)

        self.combined_interactions = pd.concat(combined_dfs)
        full_index = self.combined_interactions.index.to_numpy()
        swap = (self.combined_interactions.index.get_level_values(0) > 
                self.combined_interactions.index.get_level_values(1))
        changed_idx = self.combined_interactions.loc[swap].swaplevel().index.rename(['mol1_idx','mol2_idx']).to_numpy()
        full_index[swap] = changed_idx
        self.combined_interactions.index = pd.MultiIndex.from_tuples(full_index,names=['mol1_idx','mol2_idx'])
        self.combined_interactions.sort_index(inplace=True)

        if only_unique:
            self.combined_interactions.drop_duplicates(inplace=True,keep='first')
            
    def add_molecule_interactions(self):
        # populates geometric interactions with combined interactions
        # matched by centroid distance
        temp_atoms = self.combined_interactions.copy()
        temp_geometric = self.geometric_interactions.copy()
        cds = pd.DataFrame(temp_geometric.centroid_distance)
        temp_atoms = temp_atoms.join(cds)
        left = temp_geometric.reset_index().set_index('centroid_distance')
        right = temp_atoms.set_index('centroid_distance')
        self.molecule_interactions = left.join(right).reset_index().set_index(['mol1_idx','mol2_idx'])
        self.molecule_interactions = self.molecule_interactions.sort_index()
        self.molecule_interactions.fillna(0,inplace=True)
        
    def centre_of_geometry(self):
        # centre of geometry of all molecules in the supercell
        mol_cogs = []
        for mol in self.molecules:
            mol_cogs.append(mol.cog)
        mol_cogs = np.array(mol_cogs)
        return np.average(mol_cogs,axis=0)
    
    def sanitise_interactions(self,interaction_type='geometric',inplace=False):
        # Removes small numerical differences between interactions by replacing with first incident
        # required for matching interactions for joins, unique interactions, etc
        if interaction_type == 'geometric':
            interactions = self.geometric_interactions.copy()
        if interaction_type == 'atom':
            interactions = self.atom_interactions.copy()
        length = len(interactions.columns)
        seen = np.array([np.zeros(shape=(length))])
        new_values = []
        mask = []
        for idx in interactions.index:
            values = interactions.loc[idx].values
            if list(values) in seen.tolist():
                mask.append(False)
            else:
                if np.sum((np.sum(np.isclose(values,seen,atol=0.05),axis=1) == length),axis=0)>0:
                    mask.append(True)
                    new_values.append(seen[np.sum(np.isclose(values,seen,atol=0.05),axis=1) == length][0])   
                else:
                    mask.append(False)
                    seen = np.append(seen,[values], axis=0)
        interactions[mask] = new_values
        if inplace:
            if interaction_type == 'geometric':
                self.geometric_interactions = interactions.copy()
            if interaction_type == 'atom':
                self.atom_interactions = interactions.copy()

    def to_xyz(self,fname,for_point_cloud=False):
        split = fname.split('.')
        name = split[0] if len(split) == 1 else split[:-1]
        file = open(name+'.xyz', 'w')
        if not for_point_cloud:
            atom_count = len([atom for mol in self.molecules for atom in mol.atoms])
            file.write(f'{atom_count}\n')
            for mol in self.molecules:
                for atom in mol.atoms:
                    x, y, z = atom.coordinates
                    if 'ring' in atom.symbol:
                        file.write(f'Ti {x} {y} {z}\n')
                    else:
                        file.write(f'{atom.symbol} {x} {y} {z}\n')
            file.close()
        else:
            for mol in self.molecules:
                for atom in mol.atoms:
                    x, y, z = atom.coordinates
                    file.write(f'{x} {y} {z}\n')
            file.close()

    def to_mol2(self,fname,add_interactions=False):
        # remember to add residue number as well so you can create a pymol script to turn the dummy bonds
        # to dashes
        pass

    def unique_dimers_to_mol2(self,fname,add_interactions=False):
        pass

################################################# INTERACTION DICT ######################################################
class InteractionDict():
    def __init__(self,atom):
        self.atom = atom
        self.check_hydrogen_bond_donor()
        self.check_hydrogen_bond_acceptor()
        self.check_halogen_bond_donor()
        self.check_halogen_bond_acceptor()
        self.check_pi_bond_donor()
        self.check_pi_bond_acceptor()
        self.check_ch_pi_bond_donor()
        self.check_ch_pi_bond_acceptor()
        self.check_hydrophobic()
        
    def check_hydrogen_bond_donor(self):
        if self.atom.symbol == 'H':
            neighbours = [atom.symbol for atom in self.atom.neighbours]
            assert len(neighbours) > 0
            if  np.sum(np.isin(np.array(neighbours),np.array(CONFIG['HYDROGEN_BOND']['DONORS']))) > 0:
                self.hydrogen_bond_donor = True 
            else:
                self.hydrogen_bond_donor = False
        else:
            self.hydrogen_bond_donor = False
        
    def check_hydrogen_bond_acceptor(self):
        if self.atom.symbol in CONFIG['HYDROGEN_BOND']['ACCEPTORS']:
            self.hydrogen_bond_acceptor = True 
        else:
            self.hydrogen_bond_acceptor = False
            
    def check_halogen_bond_donor(self):
        if self.atom.symbol in CONFIG['HALOGEN_BOND']['DONORS']:
            self.halogen_bond_donor = True
        else:
            self.halogen_bond_donor = False
            
    def check_halogen_bond_acceptor(self):
        if self.atom.symbol in CONFIG['HALOGEN_BOND']['ACCEPTORS']:
            self.halogen_bond_acceptor = True
        else:
            self.halogen_bond_acceptor = False
        
    def check_pi_bond_donor(self):
        if self.atom.symbol in CONFIG['PIPI_BOND']['DONORS']:
            self.pi_bond_donor = True
        else:
            self.pi_bond_donor = False 
            
    def check_pi_bond_acceptor(self):
        if self.atom.symbol in CONFIG['PIPI_BOND']['ACCEPTORS']:
            self.pi_bond_acceptor = True
        else:
            self.pi_bond_acceptor = False 
            
    def check_ch_pi_bond_donor(self):
        if self.atom.symbol in CONFIG['CHPI_BOND']['DONORS']:
            neighbours = neighbours = [atom.symbol for atom in self.atom.neighbours]
            assert len(neighbours) > 0
            if  np.sum(np.isin(np.array(neighbours),np.array(['C']))) > 0:
                self.ch_pi_bond_donor = True
            else:
                self.ch_pi_bond_donor = False
        else:
            self.ch_pi_bond_donor = False
    
    def check_ch_pi_bond_acceptor(self):
        if self.atom.symbol in CONFIG['CHPI_BOND']['ACCEPTORS']:
            self.ch_pi_bond_acceptor = True
        else:
            self.ch_pi_bond_acceptor = False
            
    def check_hydrophobic(self):
        if self.atom.symbol == 'C':
            neighbours = neighbours = [atom.symbol for atom in self.atom.neighbours]
            assert len(neighbours) > 0
            if  np.sum(np.isin(np.array(neighbours),np.array(['C','H']),invert=True)) == 0:
                self.hydrophobic = True
            else:
                self.hydrophobic = False
        else:
            self.hydrophobic = False

################################################# INTERACTION ######################################################
class Interaction():
    def __init__(self,atom1,atom2,mol1_idx=np.nan,mol2_idx=np.nan,atom1_idx=np.nan,atom2_idx=np.nan):
        self.atom1 = atom1
        self.atom2 = atom2
        self.mol1_idx = mol1_idx
        self.mol2_idx = mol2_idx
        self.atom1_idx = atom1_idx
        self.atom2_idx = atom2_idx
        self.displacement = self.atom2.coordinates - self.atom1.coordinates
        self.distance = np.sqrt(np.dot(self.displacement,self.displacement))
        self.vdw_sum = self.atom2.vdw_radii + self.atom1.vdw_radii
        self.vdw_distance = self.distance - self.vdw_sum
        if self.vdw_distance <= 0:
            self.vdw_contact = True
        else:
            self.vdw_contact = False
        self.angle = np.nan
        self.theta1 = np.nan
        self.theta2 = np.nan
        self.vertical_offset = np.nan
        self.horizontal_offset = np.nan
        self.hydrogen_bond_type = np.nan
        self.halogen_bond_type = np.nan
        self.hydrogen_bond = self.check_hydrogen_bond()
        self.halogen_bond = self.check_halogen_bond()
        self.pi_bond = self.check_pi_bond()
        self.ch_pi_bond = self.check_ch_pi_bond()
        self.hydrophobic = self.check_hydrophobic()
        
    def check_hydrogen_bond(self):
        case1 = self.atom1.interaction.hydrogen_bond_donor & self.atom2.interaction.hydrogen_bond_acceptor
        case2 = self.atom2.interaction.hydrogen_bond_donor & self.atom1.interaction.hydrogen_bond_acceptor
        within_distance = ((self.distance < CONFIG['HYDROGEN_BOND']['MAX_DISTANCE']) & 
                            (self.distance > CONFIG['HYDROGEN_BOND']['MIN_DISTANCE']))
        if case1 & within_distance:
            neighbour = self.atom1.neighbours[0]
            angle = bond_angle(neighbour,self.atom1,self.atom2)
            neigh_symbol = neighbour.symbol 
            if ((angle > CONFIG['HYDROGEN_BOND']['MIN_ANGLE']) & 
                (angle < CONFIG['HYDROGEN_BOND']['MAX_ANGLE'])):
                self.angle = angle
                self.hydrogen_bond_type = neigh_symbol
                return True
            else:
                return False
        elif case2 & within_distance:
            neighbour = self.atom2.neighbours[0]
            angle = bond_angle(neighbour,self.atom2,self.atom1)
            neigh_symbol = neighbour.symbol 
            if ((angle > CONFIG['HYDROGEN_BOND']['MIN_ANGLE']) & 
                (angle < CONFIG['HYDROGEN_BOND']['MAX_ANGLE'])):
                self.angle = angle
                self.hydrogen_bond_type = neigh_symbol
                return True
            else:
                return False
        else:
            return False
        
    def check_halogen_bond(self):
        # Assign whether halogen bond
        case1 = self.atom1.interaction.halogen_bond_donor & self.atom2.interaction.halogen_bond_acceptor
        case2 = self.atom2.interaction.halogen_bond_donor & self.atom1.interaction.halogen_bond_acceptor
        within_distance = ((self.distance < CONFIG['HALOGEN_BOND']['MAX_DISTANCE']) &
                           (self.distance > CONFIG['HALOGEN_BOND']['MIN_DISTANCE']))
        if (case1 | case2) & within_distance:
            n1 = self.atom1.neighbours[0]
            n2 = self.atom2.neighbours[1]
            theta1 = bond_angle(n1,self.atom1,self.atom2)
            self.theta1 = theta1
            theta2 = bond_angle(n2,self.atom2,self.atom1)
            self.theta2 = theta2
            if ((np.abs(theta2 - theta1) > CONFIG['HALOGEN_BOND']['TYPE1_BOND_DIFFERENCE_MIN']) & 
                (np.abs(theta2 - theta1) < CONFIG['HALOGEN_BOND']['TYPE1_BOND_DIFFERENCE_MAX'])):
                self.halogen_bond_type = 1
            elif ((np.abs(theta2 - theta1) > CONFIG['HALOGEN_BOND']['TYPE1X2_BOND_DIFFERENCE_MIN']) & 
                (np.abs(theta2 - theta1) < CONFIG['HALOGEN_BOND']['TYPE1X2_BOND_DIFFERENCE_MAX'])):
                self.halogen_bond_type = 1.5
            elif ((np.abs(theta2 - theta1) > CONFIG['HALOGEN_BOND']['TYPE2_BOND_DIFFERENCE_MIN']) & 
                (np.abs(theta2 - theta1) < CONFIG['HALOGEN_BOND']['TYPE2_BOND_DIFFERENCE_MAX'])):
                self.halogen_bond_type = 2
            else:
                pass
            return True
        else:
            return False
        
    def check_pi_bond(self):
        # Assign whether pi-pi bond
        case1 = self.atom1.interaction.pi_bond_donor & self.atom2.interaction.pi_bond_acceptor
        case2 = self.atom2.interaction.pi_bond_donor & self.atom1.interaction.pi_bond_acceptor
        within_distance = ((self.distance < CONFIG['PIPI_BOND']['MAX_DISTANCE']) &
                              (self.distance > CONFIG['PIPI_BOND']['MIN_DISTANCE']))
        if (case1 | case2) & within_distance:
            # Calculate bond angle
            # Angle between pi-pi bond and plane of ring1
            pi_plane1 = self.atom1.plane
            pi_plane2 = self.atom2.plane
            pi_bond_angle = pi_plane1.plane_angle(pi_plane2)
            # Calculating offset
            disp = self.atom2.coordinates - self.atom1.coordinates
            vec_angle = np.radians(vector_angle(disp, np.array([pi_plane1.a,pi_plane1.b,pi_plane1.c])))
            h_offset = self.distance*np.sin(vec_angle)
            v_offset = self.distance*np.cos(vec_angle)
            if h_offset < CONFIG['PIPI_BOND']['MAX_OFFSET']:
                if pi_bond_angle > 90:
                    pi_bond_angle = 180 - pi_bond_angle
                within_angle = ((pi_bond_angle > CONFIG['PIPI_BOND']['MIN_ANGLE']) & 
                                (pi_bond_angle < CONFIG['PIPI_BOND']['MAX_ANGLE']))
                if within_angle:
                    self.angle = pi_bond_angle
                    self.horizontal_offset = h_offset
                    self.vertical_offset = v_offset
                    return True
        else:
            return False
    
    def check_ch_pi_bond(self):
        # Assign whether CH-pi bond
        case1 = self.atom1.interaction.ch_pi_bond_donor & self.atom2.interaction.ch_pi_bond_acceptor
        case2 = self.atom2.interaction.ch_pi_bond_donor & self.atom1.interaction.ch_pi_bond_acceptor
        within_distance = ((self.distance < CONFIG['CHPI_BOND']['MAX_DISTANCE']) & 
                           (self.distance > CONFIG['CHPI_BOND']['MIN_DISTANCE']))
        if case1 & within_distance:
            pi_plane = self.atom2.plane
            pi_norm = np.array([pi_plane.a,pi_plane.b,pi_plane.c])
            disp = self.atom2.coordinates - self.atom1.coordinates
            pi_bond_angle = np.degrees(np.arccos(disp.dot(pi_norm)/(np.sqrt(disp.dot(disp))*np.sqrt(pi_norm.dot(pi_norm)))))
            if pi_bond_angle > 90:
                pi_bond_angle = 180 - pi_bond_angle
            pi_within_angle = ((pi_bond_angle > CONFIG['CHPI_BOND']['MIN_ANGLE']) & (pi_bond_angle < CONFIG['CHPI_BOND']['MAX_ANGLE']))
            if pi_within_angle:
                self.angle = pi_bond_angle
                return True
        elif case2 & within_distance:
            pi_plane = self.atom1.plane
            pi_norm = np.array([pi_plane.a,pi_plane.b,pi_plane.c])
            disp = self.atom2.coordinates - self.atom1.coordinates
            pi_bond_angle = np.degrees(np.arccos(disp.dot(pi_norm)/(np.sqrt(disp.dot(disp))*np.sqrt(pi_norm.dot(pi_norm)))))
            if pi_bond_angle > 90:
                pi_bond_angle = 180 - pi_bond_angle
            pi_within_angle = ((pi_bond_angle > CONFIG['CHPI_BOND']['MIN_ANGLE']) & (pi_bond_angle < CONFIG['CHPI_BOND']['MAX_ANGLE']))
            if pi_within_angle:
                self.angle = pi_bond_angle
                return True
        else:
            return False
        
    def check_hydrophobic(self):
        # Hydrophobic Interactions
        case1 = self.atom1.interaction.hydrophobic & self.atom2.interaction.hydrophobic
        case2 = case1
        within_distance = ((self.distance < CONFIG['CC_HYDROPHOBIC_BOND']['MAX_DISTANCE']) & 
                                       (self.distance > CONFIG['CHPI_BOND']['MIN_DISTANCE']))
        if (case1 | case2) & within_distance:
            return True
        else:
            return False
        
    def to_dict(self):
        info = {
            'mol1_idx':self.mol1_idx,
            'mol2_idx':self.mol2_idx,
            'atom1_idx':self.atom1_idx,
            'atom2_idx':self.atom2_idx,
            'atom1_symbol':self.atom1.symbol,
            'atom2_symbol':self.atom2.symbol,
            'atom1_type':self.atom1.type,
            'atom2_type':self.atom2.type,
            'a':self.displacement[0],
            'b':self.displacement[1],
            'c':self.displacement[2],
            'distance':self.distance,
            'vdw_sum':self.vdw_sum,
            'vdw_distance':self.vdw_distance,
            'vdw_contact':self.vdw_contact,
            'hydrogen_bond':self.hydrogen_bond,
            'halogen_bond':self.halogen_bond,
            'pi_bond':self.pi_bond,
            'ch_pi_bond':self.ch_pi_bond,
            'hydrophobic':self.hydrophobic,
            'angle':self.angle,
            'theta1':self.theta1,
            'theta2':self.theta2,
            'horizontal_offset':self.horizontal_offset,
            'vertical_offset':self.vertical_offset,
            'hydrogen_bond_type':self.hydrogen_bond_type,
            'halogen_bond_type':self.halogen_bond_type}
        
        return info    

################################################# GEOMETRY ######################################################
class Point():
    pass 

class Vector():
    pass

class Plane():
    # https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
    def __init__(self,points):
        """
        p, n = planeFit(points)

        Given an array, points, of shape (d,...)
        representing points in d-dimensional space,
        fit an d-dimensional plane to the points.
        Return a point, p, on the plane (the point-cloud centroid),
        and the normal, n.
        """
        if len(points) == 2:
            centre = np.average(points,axis=0)
            print(centre,points)
            points = np.concatenate([points,centre],axis=0)
        if points.shape[0] >= points.shape[1]:
            points = np.vstack([points[:,0],points[:,1],points[:,2]])
        points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
        assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
        self.ctr = points.mean(axis=1)
        x = points - self.ctr[:,np.newaxis]
        M = np.dot(x, x.T) # Could also use np.cov(x) here.
        vect = la.svd(M)[0][:,-1]
        self.a, self.b, self.c = vect
        # ax + by + cz + d = 0
        self.d = (points[0,0]*self.a + points[1,0]*self.b + points[2,0]*self.c)*-1

    def plane_angle(self, plane):
        a1,b1,c1 = self.a,self.b, self.c
        a2,b2,c2 = plane.a,plane.b, plane.c
            
        d = ( a1 * a2 + b1 * b2 + c1 * c2 )
        e1 = np.sqrt( a1 * a1 + b1 * b1 + c1 * c1)
        e2 = np.sqrt( a2 * a2 + b2 * b2 + c2 * c2)
        d = d / (e1 * e2)
        A = np.degrees(np.arccos(d))
        if A > 90:
            A = 180 - A
        return A

    def unit_normal(self):
        mag = np.sqrt(self.a**2 + self.b**2 + self.c**2)
        unit_norm = np.array([self.a,self.b,self.c]) / mag
        return unit_norm
    
    def point_distance(self,coordinates): 
        x1, y1, z1 = coordinates[0], coordinates[1], coordinates[2]
        d = np.abs((self.a * x1 + self.b * y1 + self.c * z1 + self.d)) 
        e = (np.sqrt(self.a * self.a + self.b * self.b + self.c * self.c))
        return d/e

    def test_planarity(self,atoms = None):
        if atoms == None:
            devs = [self.point_distance(atom) for atom in self.atoms]
            if len(np.where(np.array(devs)>2)[0]) >= 1:
                return False
            else:
                return True
        else:
            devs = [self.point_distance(atom) for atom in atoms]
            if len(np.where(np.array(devs)>2)[0]) >= 1:
                return False
            else:
                return True

    def get_planar_basis(self):
        normal = np.array(self.a,self.b,self.c)
        
class Ellipsoid():
    '''
    https://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    Python implementation of the MATLAB function MinVolEllipse, based on the Khachiyan algorithm
    for both 
    A is a matrix containing the information regarding the shape of the ellipsoid 
    to get radii from A you have to do SVD on it, giving U Q and V
    1 / sqrt(Q) gives the radii of the ellipsoid
    problems arise for planar motifs. add two extra points at centroid of +/- 0.00001*plane_normal to overcome
    
    Find the minimum volume ellipse around a set of atom objects.
    Return A, c where the equation for the ellipse given in "center form" is
    (x-c).T * A * (x-c) = 1
    [U Q V] = svd(A); 
    where r = 1/sqrt(Q)
    V is rotation matrix
    U is ??? 
    '''
    def __init__(self,points,tol = 0.00001):
        self.points = points
        points_asarray = np.array(self.points)
        points = np.asmatrix(points_asarray)
        N, d = points.shape
        Q = np.column_stack((points, np.ones(N))).T
        err = tol+1.0
        u = np.ones(N)/N
        try:
            while err > tol:
                # assert u.sum() == 1 # invariant
                X = Q * np.diag(u) * Q.T
                M = np.diag(Q.T * la.inv(X) * Q)
                jdx = np.argmax(M)
                step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
                new_u = (1-step_size)*u
                new_u[jdx] += step_size
                err = la.norm(new_u-u)
                u = new_u
            c = u*points
            A = la.inv(points.T*np.diag(u)*points - c.T*c)/d    
        except: # For singular matrix errors i.e. motif is ellipse rather than ellipsoid
            centroid = np.average(points_asarray,axis=0)
            points = np.array([atom.coordinates for atom in self.atoms])
            plane = Plane(points)
            normal = np.array([plane.a,plane.b,plane.c])
            norm_mag = np.sqrt(np.dot(normal,normal))
            for i, norm in enumerate(normal):
                normal[i] = norm * 1 / norm_mag
            centroid = np.average(points,axis=0).reshape(-1,3)
            p1 = centroid + normal*0.00001
            p2 = centroid - normal*0.00001
            points_asarray = np.concatenate([points_asarray,p1,p2],axis=0)
            points = np.asmatrix(points_asarray)
            N, d = points.shape
            Q = np.column_stack((points, np.ones(N))).T
            err = tol+1.0
            u = np.ones(N)/N
            while err > tol:
                # assert u.sum() == 1 # invariant
                X = Q * np.diag(u) * Q.T
                M = np.diag(Q.T * la.inv(X) * Q)
                jdx = np.argmax(M)
                step_size = (M[jdx]-d-1.0)/((d+1)*(M[jdx]-1.0))
                new_u = (1-step_size)*u
                new_u[jdx] += step_size
                err = la.norm(new_u-u)
                u = new_u
            c = u*points
            A = la.inv(points.T*np.diag(u)*points - c.T*c)/d   
            
        self.matrix = np.asarray(A)
        self.centre = np.squeeze(np.asarray(c))
        U, D, V = la.svd(self.matrix)    
        self.rx, self.ry, self.rz = 1./np.sqrt(D)
        self.axes = np.array([self.rx,self.ry,self.rz])

# Old
def bond_angle(atom1,atom2,atom3):
    a = atom1.coordinates
    b = atom2.coordinates
    c = atom3.coordinates

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def torsional_angle(atom1,atom2,atom3,atom4):
    # returns interplanar angle between planes defined by atom1, atom2, atom3, and atom2, atom3, atom4
    pass
def vector(atom1,atom2, as_angstrom=False):
    # returns the vector defined by the position between two atoms
    pass
def vector_angle(v1,v2):
    theta = np.arccos((v1.dot(v2))/(np.sqrt(v1.dot(v1))*np.sqrt(v2.dot(v2))))
    return np.degrees(theta)
def vector_plane_angle(vector, plane):
    # returns the angle made between a vector and a plane
    pass

def ellipse(rx,ry,rz):
    u, v = np.mgrid[0:2*np.pi:20j, -np.pi/2:np.pi/2:10j]
    x = rx*np.cos(u)*np.cos(v)
    y = ry*np.sin(u)*np.cos(v)
    z = rz*np.sin(v)
    return x,y,z

def generate_ellipsoids(crystal,mol_pairs,atom_pairs,tol = 0.00001):

    ellipsoid_info = []
    for molecule_pair, atom_pair in zip(mol_pairs,atom_pairs):
        molecules = [crystal.molecules[molecule_pair[0]],crystal.molecules[molecule_pair[1]]]
        atoms = [[molecules[0].atoms[pair[0]],molecules[1].atoms[pair[1]]] for pair in atom_pair]
        atoms = np.reshape(atoms,-1)
        A, centroid = mvee(atoms,tol=tol)    
        ellipsoid_info.append(dict(matrix=A,centre=centroid))
    
    return ellipsoid_info

def CalcSVDRotation(mol1, mol2):
    A = np.array([atom.coordinates for atom in mol1.atoms]).T
    B = np.array([atom.coordinates for atom in mol2.atoms]).T

    disp = mol1.centre_of_geometry() - mol2.centre_of_geometry()
    
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    reflected = np.linalg.det(R) < 0
    if reflected:
        print("det(R) < 0, reflection detected!, correcting for it ...")
        Vt[2,:] = Vt[2,:]*-1
        R = Vt.T @ U.T
    
    t = -R @ centroid_A + centroid_B
    t = t.reshape(-1)
    # Account for 180 degrees about an axis
    #if not reflected:
    t = np.where(np.abs(disp*-1 - t) < 0.001, t*-1,t)

    return R, t

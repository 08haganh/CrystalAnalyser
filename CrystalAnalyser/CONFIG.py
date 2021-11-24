# CONFIG FILE
from rdkit import Chem

# HYDROGEN BONDING
HYDROGEN_BOND_DONORS = ['O','N','F','S','C']
HYDROGEN_BOND_ACCEPTORS = ['O','N','F','S','P']
HYDROGEN_BOND_DISTANCE_MIN = 0.05
HYDROGEN_BOND_DISTANCE_MAX = 4.1 # Max. distance between hydrogen bond donor and acceptor (Hubbard & Haider, 2001) + 0.6 A
HYDROGEN_BOND_ANGLE_MIN = 100 # Min. angle at the hydrogen bond donor (Hubbard & Haider, 2001) + 10
HYDROGEN_BOND_ANGLE_MAX = 180

# HALOGEN BONDING
# Definition - (Cavallo et al. 2016, The Halogen Bond)
## A halogen bond occurs when there is evidence of a net attractive interaction between an electrophilic region associated with a 
## halogen atom in a molecular entity and a nucleophilic region in another, or the same, molecular entity.
# parameter ref ()
HALOGEN_BOND_DONORS = ['F','CL','Cl','BR','Br','I'] # (Cavallo et al. 2016, The Halogen Bond)
HALOGEN_BOND_ACCEPTORS = ['O','N','S','Se','P','F','Cl','CL','Br','BR','I'] # (Cavallo et al. 2016, The Halogen Bond)
HALOGEN_BOND_DISTANCE_MIN = 0.05
HALOGEN_BOND_DISTANCE_MAX = 4.0 # (Halogen bonds in biological molecules., Auffinger)+0.5
TYPE1_BOND_DIFFERENCE_MIN = 0 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)
TYPE1_BOND_DIFFERENCE_MAX = 15 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)
TYPE1X2_BOND_DIFFERENCE_MIN = 15 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)
TYPE1X2_BOND_DIFFERENCE_MAX = 30 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)
TYPE2_BOND_DIFFERENCE_MIN = 30 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)
TYPE2_BOND_DIFFERENCE_MAX = 180 # (Mukherjee et al. 2014, Halogen bonds in Crystal Engineering: Like Hydrogen Bonds yet different)

# PI STACKING
PI_PI_BOND_DONORS = ['aromatic_ring']
PI_PI_BOND_ACCEPTORS = ['aromatic_ring']
PI_STACKING_BOND_DISTANCE_MIN = 0.05
PI_STACKING_BOND_DISTANCE_MAX = 5.5 # Max. distance for parallel or offset pistacking (McGaughey, 1998)
PI_STACKING_BOND_ANGLE_MIN = 0 
PI_STACKING_BOND_ANGLE_MAX = 30  # Max. Deviation from parallel or perpendicular orientation (in degrees)
PI_STACKING_OFFSET_MAX = 4.0 # Maximum offset of the two rings (corresponds to the radius of benzene + 0.5 A)

# CH-PI BONDING
CHPI_BOND_DONORS = ['H']
CHPI_BOND_ACCEPTORS = ['aromatic_ring']
CHPI_BOND_DISTANCE_MIN = 0.05
CHPI_BOND_DISTANCE_MAX = 4.0
CHPI_BOND_ANGLE_MIN = 0
CHPI_BOND_ANGLE_MAX = 50

# CC HYDROPHOBIC INTERACTIONS
CC_BOND_DISTANCE_MIN = 0.05
CC_BOND_DISTANCE_MAX = 4.0  # 4.0 # Distance cutoff for detection of hydrophobic contacts

# DEFINE CONFIG DICTIONARY
CONFIG = {'HYDROGEN_BOND':{'ACCEPTORS':HYDROGEN_BOND_ACCEPTORS,
			    'DONORS':HYDROGEN_BOND_DONORS,
			    'MIN_DISTANCE':HYDROGEN_BOND_DISTANCE_MIN,
			    'MAX_DISTANCE':HYDROGEN_BOND_DISTANCE_MAX,
			    'MIN_ANGLE':HYDROGEN_BOND_ANGLE_MIN,
			    'MAX_ANGLE':HYDROGEN_BOND_ANGLE_MAX},
	  'HALOGEN_BOND':{'ACCEPTORS':HALOGEN_BOND_ACCEPTORS,
			   'DONORS':HALOGEN_BOND_DONORS,
			   'MIN_DISTANCE':HALOGEN_BOND_DISTANCE_MIN,
			   'MAX_DISTANCE':HALOGEN_BOND_DISTANCE_MAX,
			   'TYPE1_BOND_DIFFERENCE_MIN':TYPE1_BOND_DIFFERENCE_MIN,
			   'TYPE1_BOND_DIFFERENCE_MAX':TYPE1_BOND_DIFFERENCE_MAX,
			   'TYPE1X2_BOND_DIFFERENCE_MIN':TYPE1X2_BOND_DIFFERENCE_MIN,
			   'TYPE1X2_BOND_DIFFERENCE_MAX':TYPE1X2_BOND_DIFFERENCE_MAX,
			   'TYPE2_BOND_DIFFERENCE_MIN':TYPE2_BOND_DIFFERENCE_MIN,
			   'TYPE2_BOND_DIFFERENCE_MAX':TYPE2_BOND_DIFFERENCE_MAX},
	  'PIPI_BOND':{'ACCEPTORS':PI_PI_BOND_ACCEPTORS,
	  	    'DONORS':PI_PI_BOND_DONORS,
		    'MIN_DISTANCE':PI_STACKING_BOND_DISTANCE_MIN,
			'MAX_DISTANCE':PI_STACKING_BOND_DISTANCE_MAX,
			'MIN_ANGLE':PI_STACKING_BOND_ANGLE_MIN,
			'MAX_ANGLE':PI_STACKING_BOND_ANGLE_MAX,
			'MAX_OFFSET':PI_STACKING_OFFSET_MAX},
	  'CHPI_BOND':{'ACCEPTORS':CHPI_BOND_ACCEPTORS,
	  	    'DONORS':CHPI_BOND_DONORS,
			'MIN_DISTANCE':CHPI_BOND_DISTANCE_MIN,
			'MAX_DISTANCE':CHPI_BOND_DISTANCE_MAX,
			'MIN_ANGLE':CHPI_BOND_ANGLE_MIN,
			'MAX_ANGLE':CHPI_BOND_ANGLE_MAX},
	  'CC_HYDROPHOBIC_BOND':{'MIN_DISTANCE':CC_BOND_DISTANCE_MIN,
			         'MAX_DISTANCE':CC_BOND_DISTANCE_MAX}
}

vdw_radii = {
              'Al': 2, 'Sb': 2, 'Ar': 1.88, 'As': 1.85, 'Ba': 2,
              'Be': 2, 'Bi': 2, 'B': 2, 'Br': 1.85, 'Cd': 1.58,
              'Cs': 2, 'Ca': 2, 'C': 1.7, 'Ce': 2, 'Cl': 1.75,
              'Cr': 2, 'Co': 2, 'Cu': 1.4, 'Dy': 2, 'Er': 2,
              'Eu': 2, 'F':  1.47, 'Gd': 2, 'Ga': 1.87, 'Ge': 2,
              'Au': 1.66, 'Hf': 2, 'He': 1.4, 'Ho': 2, 'H': 1.09,
              'In': 1.93, 'I': 1.98, 'Ir': 2, 'Fe': 2, 'Kr': 2.02,
              'La': 2, 'Pb': 2.02, 'Li': 1.82, 'Lu': 2, 'Mg': 1.73,
              'Mn': 2, 'Hg': 1.55, 'Mo': 2, 'Nd': 2, 'Ne': 1.54,
              'Ni': 1.63, 'Nb': 2, 'N':  1.55, 'Npl':  1.55, 'Os': 2,
              'O': 1.52,
              'Pd': 1.63, 'P': 1.8, 'Pt': 1.72, 'K': 2.75, 'Pr': 2,
              'Pa': 2, 'Re': 2, 'Rh': 2, 'Rb': 2, 'Ru': 2, 'Sm': 2,
              'Sc': 2, 'Se': 1.9, 'Si': 2.1, 'Ag': 1.72, 'Na': 2.27,
              'Sr': 2, 'S': 1.8, 'Ta': 2, 'Te': 2.06, 'Tb': 2,
              'Tl': 1.96, 'Th': 2, 'Tm': 2, 'Sn': 2.17, 'Ti': 2,
              'W': 2, 'U':  1.86, 'V':  2, 'Xe': 2.16, 'Yb': 2,
              'Y': 2, 'Zn': 1.29, 'Zr': 2, 'X':  1.0, 'D':  1.0,
              'O2': 1.52,'ring':0,
              'AL': 2, 'SB': 2, 'AR': 1.88, 'AS': 1.85, 'BA': 2,
              'BE': 2, 'BI': 2, 'B': 2, 'BR': 1.85, 'CD': 1.58,
              'CS': 2, 'CA': 2, 'C': 1.7, 'CE': 2, 'CL': 1.75,
              'CR': 2, 'CO': 2, 'CU': 1.4, 'DY': 2, 'ER': 2,
              'EU': 2, 'F':  1.47, 'GD': 2, 'GA': 1.87, 'GE': 2,
              'AU': 1.66, 'HF': 2, 'HE': 1.4, 'HL': 2, 'H': 1.09,
              'IN': 1.93, 'I': 1.98, 'IR': 2, 'FE': 2, 'KR': 2.02,
              'LA': 2, 'PB': 2.02, 'LI': 1.82, 'LU': 2, 'MG': 1.73,
              'MN': 2, 'HG': 1.55, 'MO': 2, 'ND': 2, 'NE': 1.54,
              'NI': 1.63, 'NB': 2, 'N':  1.55, 'NPL':  1.55, 'OS': 2,
              'O': 1.52,
              'PD': 1.63, 'P': 1.8, 'PT': 1.72, 'K': 2.75, 'PR': 2,
              'PA': 2, 'RE': 2, 'RH': 2, 'RB': 2, 'RU': 2, 'SM': 2,
              'SC': 2, 'SE': 1.9, 'SI': 2.1, 'AG': 1.72, 'NA': 2.27,
              'SR': 2, 'S': 1.8, 'TA': 2, 'TE': 2.06, 'TB': 2,
              'TL': 1.96, 'TH': 2, 'TM': 2, 'SN': 2.17, 'TI': 2,
              'W': 2, 'U':  1.86, 'V':  2, 'XE': 2.16, 'YB': 2,
              'Y': 2, 'ZN': 1.29, 'ZR': 2, 'X':  1.0, 'D':  1.0,
              'O2': 1.52,'ring':0,'aromatic_ring':0,'aliphatic_ring':0
                 }

atomic_mass = {
                'H':1.0079, 'He':4.0026, 'Li':6.941, 'Be':9.0122, 'B':10.811,
                'C':12.0107, 'N': 14.0067, 'O':15.9994, 'F':18.9984, 'Ne':20.1797,
                'Na':22.9897, 'Mg':24.305, 'Al':26.9815, 'Si':28.0855, 'P':30.9738,
                'S':32.065, 'Cl':35.453, 'K':39.0983, 'Ar':39.948, 'Ca':40.078,
                'Sc':44.9559, 'Ti':47.867, 'V':50.9415, 'Cr':51.9961, 'Mn':54.938,
                'Fe':55.845, 'Ni':58.6934, 'Co':58.9332, 'Cu':63.546, 'Zn':65.39
                }

atomic_number = {
                'H':1, 'He':2, 'Li':3, 'Be':4, 'B':5,
                'C':6, 'N':7, 'O':8, 'F':9, 'Ne':10,
                'Na':11, 'Mg':12, 'Al':13, 'Si':14, 'P':15,
                'S':16, 'Cl':17, 'K':19, 'Ar':18, 'Ca':20,
                'Sc':21, 'Ti':22, 'V':23, 'Cr':24, 'Mn':25,
                'Fe':26, 'Ni':28, 'Co':27, 'Cu':29, 'Zn':30
}

rdkit_bond_types = {
                    'ar':Chem.rdchem.BondType.AROMATIC,
                    '1':Chem.rdchem.BondType.SINGLE,
                    '2':Chem.rdchem.BondType.DOUBLE,
                    '3':Chem.rdchem.BondType.TRIPLE

}
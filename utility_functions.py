import os
# Uncomment if you don't want to use GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

from openff.units import unit
import numpy as np
from chargecraft.storage.storage import MoleculePropRecord, MoleculePropStore
from openff.toolkit.topology import Molecule
from ChargeAPI.API_infrastructure.charge_request import module_version
from ChargeAPI.API_infrastructure.esp_request.module_version_esp import handle_esp_request

from rdkit import Chem
# from espaloma_charge.openff_wrapper import EspalomaChargeToolkitWrapper
import json
from rdkit.Chem import Draw
from typing import Literal, Optional
# from MultipoleNet import load_model, build_graph_batched, D_Q
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R
from openff.units import unit
from openff.toolkit.topology import Molecule
from typing import Union, Optional
from openff.toolkit.utils.nagl_wrapper import NAGLToolkitWrapper
# from openff.nagl import GNNModel

import matplotlib.pyplot as plt
import numpy as np
# import tensorflow as tf


# toolkit_registry = EspalomaChargeToolkitWrapper()


VACUUM_PERMITTIVITY = 8.8541878128e-12  # Farads per Metre
ELECTRON_CHARGE = 1.602176634e-19  # Coulombs
PI = 3.141592653589793  # Pi
E_TO_C=1.602176634e-19
M_TO_ANGS = 1e10  # Metres to Angstroms
ANGS_TO_M = 1e-10 #Angstroms to Metres
AVOGADRO = 6.02214179e23  # Particles in 1 Mole
J_TO_KCAL = 0.0002390057  # Joules to kilocalories
KCAL_P_MOL_TO_HA = 0.00159360164  # Kilocalories per mole to Hartrees
HA_TO_KCAL_P_MOL = 627.509391  # Hartrees to kilocalories per mole
J_TO_KCAL_P_MOL = J_TO_KCAL * AVOGADRO  # Joules to kilocalories per mole
COULOMB_CONSTANT = 1/(4*PI*VACUUM_PERMITTIVITY)

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

OPEN_FF_AVAILABLE_CHARGE_MODELS = ['zeros','formal_charge','espaloma-am1bcc']

#TODO automate this
CHARGE_API_AVAILABLE_MODELS = ['EEM','MBIS','MBIS_CHARGE']

NAGL_GNN_MODELS = []

def scan_toolkit_registry():
    """Scan the toolkit registry of openff
    
    """
    from openff.toolkit.utils import toolkits

    toolkits = str(toolkits.GLOBAL_TOOLKIT_REGISTRY)

    if 'RDKIT' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['gasteiger', 'mmff94'])
    elif 'AmberTools' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['am1bcc', 'am1-mulliken', 'gasteiger'])
    elif 'OpenEye' in toolkits:
        OPEN_FF_AVAILABLE_CHARGE_MODELS.extend(['am1bccnosymspt','am1elf10','am1bccelf10'])
    try:
        model_small = GNNModel.load('/Users/localadmin/Documents/projects/Comparison_with_QM/RinnickerTest/nagl-small/nagl-small.pt')
        model_small_weighted = GNNModel.load('/Users/localadmin/Documents/projects/Comparison_with_QM/RinnickerTest/nagl-small-weighted/nagl-small-weighted.pt')
        NAGL_GNN_MODELS.extend(['nagl-small','nagl-small-weighted'])
    except Exception as e:
        print("could not load GNN nagl models")

scan_toolkit_registry()

if len(NAGL_GNN_MODELS) > 0:
    model_small = GNNModel.load('/Users/localadmin/Documents/projects/Comparison_with_QM/RinnickerTest/nagl-small/nagl-small.pt')
    model_small_weighted = GNNModel.load('/Users/localadmin/Documents/projects/Comparison_with_QM/RinnickerTest/nagl-small-weighted/nagl-small-weighted.pt')


def calculate_total_esp(monopoles: np.ndarray,
                        dipoles: np.ndarray, 
                        quadrupoles: np.ndarray, 
                        grid_coordinates: unit.Quantity,  # N x 3
                        atom_coordinates: unit.Quantity,  # M x 3
                        ) -> unit.Quantity:
            """Calculate ESP from grid"""
            monopole_esp = calculate_esp_monopole_au(grid_coordinates=grid_coordinates,
                                                atom_coordinates=atom_coordinates,
                                                charges = monopoles)
            dipole_esp = calculate_esp_dipole_au(grid_coordinates=grid_coordinates,
                                            atom_coordinates=atom_coordinates,
                                            dipoles= dipoles)
            quadrupole_esp = calculate_esp_quadropole_au(grid_coordinates=grid_coordinates,
                                            atom_coordinates=atom_coordinates,
                                            quadrupoles= quadrupoles)
            return (monopole_esp + dipole_esp + quadrupole_esp)

def calculate_rinnicker_esp(smiles: str, 
                conformer_no: int,
                database: MoleculePropStore,
                grid_coords: Optional[np.ndarray] = None):
    
    # grid_coords = database.retrieve(smiles)[conformer_no].grid_coordinates_quantity 
    print('building esp for')
    print(grid_coords)
    coords = database.retrieve(smiles)[conformer_no].conformer_quantity 
    mapped_smiles = database.retrieve(smiles)[conformer_no].tagged_smiles
    # print(coords)
    openff_mol = Molecule.from_mapped_smiles(mapped_smiles=mapped_smiles, allow_undefined_stereo=True)
    openff_mol.add_conformer(coordinates=coords)
    print('coords')
    print(coords)
    rdkit_mol = openff_mol.to_rdkit()
    mol_block = Chem.rdmolfiles.MolToMolBlock(rdkit_mol)
    rdkit_conf = rdkit_mol.GetConformer()
    for atom_idx in range(rdkit_mol.GetNumAtoms()):
        pos = rdkit_conf.GetAtomPosition(atom_idx)
        print(f"Atom {atom_idx}: {pos.x}, {pos.y}, {pos.z}")
    esp_req =  handle_esp_request(
        charge_model = "RIN",
        conformer_mol = mol_block,
        broken_up = True,
        grid = grid_coords
    )
    # print(esp_req)
    monopoles, dipoles, quadropoles = esp_req['monopole'], esp_req['dipole'], esp_req['quadropole']
    # monopoles_quantity, dipoles_quantity, quadropoles_quantity = rinnicker_multipoles(smiles=smiles,
    #                                                                     conformer_no=conformer_no,
    #                                                                     database=database)
    print(f'grid of shape  used is')
    print(esp_req['grid'])
    print(f'grid in database is')
    print(grid_coords)
    print('monopoles are')
    print(json.loads(monopoles))
    esp =  ((json.loads(monopoles) *AU_ESP)
            + (json.loads(dipoles) *AU_ESP)
            + (json.loads(quadropoles) *AU_ESP))
    # print(esp)
    print('length of esp:')
    print(len(esp))
    return esp
                                                                
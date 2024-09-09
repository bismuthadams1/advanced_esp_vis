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



# #Function to calculate the ESP from on-atom charges. Taken from Lily Wang's script
# def calculate_esp(
#     grid_coordinates: unit.Quantity,  # N x 3
#     atom_coordinates: unit.Quantity,  # M x 3
#     charges: unit.Quantity,  # M
#     with_units: bool = False,
# ) -> unit.Quantity:
#     """Calculate ESP from grid"""
#     ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
    
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B
#     distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M B, a0
#     inv_distance = 1 / distance #1/B

#     esp = ke * (inv_distance @ charges)  # N  (1/vacuum_permittivity) * 1/B * elementary_charge, 

#     esp_q = esp.m_as(AU_ESP)
#     if not with_units:
#         return esp_q
#     return esp

# def calculate_esp_SI(
#     grid_coordinates: unit.Quantity,  # N x 3
#     atom_coordinates: unit.Quantity,  # M x 3
#     charges: unit.Quantity,  # M
# ) -> unit.Quantity:
#     """Calculate ESP from grid in SI"""
    
#     #ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
    
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).magnitude*ANGS_TO_M  #M
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).magnitude*ANGS_TO_M   #M
#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 M
#     distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M M
#     inv_distance = 1 / distance #1/N

#     esp = COULOMB_CONSTANT * (inv_distance @ charges.magnitude*E_TO_C) * ELECTRON_CHARGE*1/(AVOGADRO) *10e-3  #MV/C * 1/M * C * eC * 1/An Mol

#     return esp   #kJ/mol


# def calculate_and_store_charges(smiles: str, 
#                       database: MoleculePropStore,
#                       charge_model: str,
#                       extra_sites: Optional[str] = None,
#                       conformer_number: Optional[int]= None,
#                       implicit_solvent: Optional[Literal["PCM","DDX"]] = None,
#                       solvent: Union[str,float] = None) -> Union[np.ndarray,tuple[np.ndarray,np.ndarray]]:
#     # Get mapped smiles and create molecule
#     mapped_smiles = database.retrieve(smiles)[conformer_number].tagged_smiles
#     charge_mol = Molecule.from_mapped_smiles(mapped_smiles)

#     #check if charges already exist in database
#     if conformer_number is not None:
#         charges_dict = database.retrieve_partial(smiles = mapped_smiles,
#                                                 conformer = database.retrieve(smiles)[conformer_number].conformer)
#     elif solvent is not None and implicit_solvent is not None:
#         if isinstance(solvent, (int, float)):
#             charges_dict = database.retrieve_partial(smiles = mapped_smiles,
#                                                 conformer = database.retrieve(smiles = smiles, 
#                                                                               implicit_solvent=implicit_solvent)[conformer_number].conformer,
#                                                 implicit_solvent = implicit_solvent,
#                                                 solvent_epsilon = solvent
#                                                 )
#         else:
#              charges_dict = database.retrieve_partial(smiles = mapped_smiles,
#                                                 conformer = database.retrieve(smiles,
#                                                                               implicit_solvent=implicit_solvent)[conformer_number].conformer,
#                                                 implicit_solvent = implicit_solvent,
#                                                 solvent_type = solvent
#                                                 )
#     else:
#         raise ValueError("please provide a conformer or solvent")

#     if charges_dict:
#         if extra_sites:
#             try:
#                 result = charges_dict[charge_model], charges_dict[extra_sites]
#                 return result
#             except KeyError:
#                 try:
#                     result = charges_dict[charge_model]
#                     return result
#                 except KeyError:
#                     pass
#             finally:
#                 pass
#         else:
#             try:
#                result = charges_dict[charge_model]
#                return result
#             except KeyError:
#                 pass

#     # Assign charges based on the provided charge model
#     if charge_model in OPEN_FF_AVAILABLE_CHARGE_MODELS:
#         if charge_model == 'espaloma-am1bcc':
#             charge_mol.assign_partial_charges(charge_model, use_conformers=[database.retrieve(smiles)[conformer_number].conformer_quantity], toolkit_registry= EspalomaChargeToolkitWrapper())
#             charges_list = charge_mol.partial_charges.magnitude
#         else:
#             charge_mol.assign_partial_charges(charge_model, use_conformers=[database.retrieve(smiles)[conformer_number].conformer_quantity])
#             charges_list = charge_mol.partial_charges.magnitude


#     elif charge_model in CHARGE_API_AVAILABLE_MODELS:
#         rdkit_mol = data_to_rdkit_molecule(mapped_smiles, database.retrieve(smiles)[conformer_number].conformer_quantity)
#         mol_block = Chem.rdmolfiles.MolToMolBlock(rdkit_mol)
#         result_json = module_version.handle_charge_request(charge_model = charge_model, 
#                                                     conformer_mol = mol_block,
#                                                     batched=False)
#         print(result_json)
        
#         result = json.loads(result_json['charge_result'])
#         charges_list = result
    
#     elif charge_model in NAGL_GNN_MODELS:
#         if charge_model == 'nagl-small':
#             charge_mol.add_conformer(database.retrieve(smiles)[conformer_number].conformer_quantity)
#             charges_list = model_small.compute_property(charge_mol).flatten().tolist()
#         elif charge_model == 'nagl-small-weighted':
#             charge_mol.add_conformer(database.retrieve(smiles)[conformer_number].conformer_quantity)
#             charges_list = model_small_weighted.compute_property(charge_mol).flatten().tolist()
#         else:
#             return None
#     else:
#         return None
    
#     # Store charges in the property store
#     database.store_partial(smiles=mapped_smiles,
#                     conformer=database.retrieve(smiles)[conformer_number].conformer,
#                     charge_model=charge_model,
#                     charges=charges_list)
    
#     return charges_list

# def data_to_openff_molecule(tagged_smiles: str, conformer_quantity: unit.Quantity) -> Molecule:
#     """Creates an openff molecule from database information
    
#     """

#     new_molecule = Molecule.from_mapped_smiles(tagged_smiles)
#     new_molecule.add_conformer(conformer_quantity)

#     return new_molecule

# def data_to_rdkit_molecule(tagged_smiles: str, conformer_quantity: unit.Quantity) -> Molecule:
#     """Creates an rdkit molecule from database information
    
#     """

#     openff_molecule = data_to_openff_molecule(tagged_smiles, conformer_quantity)
#     rdkit_molecule = Molecule.to_rdkit(openff_molecule)

#     return rdkit_molecule

# def calculate_dipole_magnitude(charges: np.ndarray, conformer: np.ndarray) -> float:
#     """Calculate dipole magnitude
     
#     """
#     reshaped_charges = np.reshape(charges,(-1,1))
#     dipole_vector = np.sum(conformer * reshaped_charges,axis=0)
#     dipole_magnitude = np.linalg.norm(dipole_vector)

#     return dipole_magnitude

# def generate_molecule_images_with_dipoles(smiles, conformer, prop_store, df_dipole):
#     charge_models = ['QM','QM', 'am1-mulliken', 'am1bcc', 'MBIS', 'MBIS_CHARGE', 'espaloma-am1bcc', 'EEM']
#     dipoles_names = ['QM_HF_Dipoles','QM_MBIS_dipoles', 'am1_dipoles', 'am1bcc_dipoles', 'NAGL_MBIS_dipoles', 'NAGL_MBIS_CHARGE_dipoles', 'espaloma_dipoles', 'EEM2015bn dipoles']
#     labels = ['QM MBIS paritioning (HF Dipole)','QM MBIS paritioning (dipole from on-atom charges)', 'am1-mulliken', 'am1bcc', 'Nagl-MBIS-dipole', 'Nagl-MBIS_CHARGE', 'espaloma-am1bcc', 'EEM']

#     tagged_smiles = prop_store.retrieve(smiles)[0].tagged_smiles
#     molecule_openff = Molecule.from_mapped_smiles(tagged_smiles)
    
#     images = []
#     dipoles = []
    
#     for charges_model, dipole_name in zip(charge_models, dipoles_names):
#         if charges_model == 'QM':
#             qm_charges = prop_store.retrieve(smiles)[conformer].mbis_charges.flatten().tolist()
#             molecule_rdkit = molecule_openff.to_rdkit()
#             for i, atom in enumerate(molecule_rdkit.GetAtoms()):
#                 lbl = '%.3f' % (qm_charges[i])
#                 atom.SetProp('atomNote', lbl)
#             images.append(molecule_rdkit)
        
#         else:
#             model_charges = calculate_and_store_charges(smiles=smiles, charge_model=charges_model, conformer_number=conformer, database=prop_store)  # .flatten().tolist()
#             molecule_rdkit = molecule_openff.to_rdkit()
#             for i, atom in enumerate(molecule_rdkit.GetAtoms()):
#                 lbl = '%.3f' % (model_charges[i])
#                 atom.SetProp('atomNote', lbl)
#             images.append(molecule_rdkit)

#         smiles_match = df_dipole['SMILES'] == smiles
#         conformer_match = df_dipole['Conformer'] == conformer
#         dipole = df_dipole.loc[smiles_match & conformer_match, dipole_name]
#         dipoles.append(dipole)

#     legend_labels = [legend_item + f" dipole: {round(float(dipole), 2)}ea_0" for (legend_item, dipole) in zip(labels, dipoles)]
#     print(legend_labels)
#     img = Draw.MolsToGridImage(images, molsPerRow=4, subImgSize=(300, 300), legends=legend_labels)
#     return img

# def general_rmse_function(x, y): 
#     x = x.to_numpy()
#     y = y.to_numpy()
#     m =(np.sum((x - y)**2, axis=0)/(y.shape[0]))**0.5
#     return m, x, y


# def rinnicker_multipoles(smiles: str, 
#                         conformer_no: int,
#                         database: MoleculePropStore) -> tuple[unit.Quantity, unit.Quantity, unit.Quantity]:
#     """
#     Produce monopole, dipole and quadropole in units e, e.A, e.A**2
#     """
    
#     # Get mapped smiles and create molecule
#     mapped_smiles = database.retrieve(smiles)[conformer_no].tagged_smiles
#     confomer_coords = database.retrieve(smiles)[conformer_no].conformer_quantity
#     openff_mol = Molecule.from_mapped_smiles(mapped_smiles)
#     openff_mol.add_conformer(confomer_coords)
#     rdkit_mol = openff_mol.to_rdkit()
#     dtype = np.float32 
#     model = load_model()
#     elements = [a.GetSymbol() for a in rdkit_mol.GetAtoms()]
#     # Generate a conformation
#     coordinates = rdkit_mol.GetConformer(0).GetPositions().astype(dtype)
#     monopoles, dipoles, quadrupoles = model.predict(coordinates, elements)
#     #Convert multipoles ot correct units.
#     monopoles_quantity = monopoles.numpy()*unit.e
#     dipoles_quantity = dipoles.numpy()*unit.e*unit.angstrom
#     quadropoles_quantity = quadrupoles.numpy()*unit.e*unit.angstrom*unit.angstrom
#     return monopoles_quantity, dipoles_quantity, quadropoles_quantity

# #Function to calculate the ESP from on-atom charges. Taken from Lily Wang's script
# def calculate_esp_monopole(
#     grid_coordinates: unit.Quantity,  # N x 3
#     atom_coordinates: unit.Quantity,  # M x 3
#     charges: unit.Quantity,  # M
#     with_units: bool = False,
#     ) -> unit.Quantity:
#     """Calculate ESP from grid"""

#     ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
#     print(ke)
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B
#     distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M B, a0
#     inv_distance = 1 / distance #1/B

#     esp = ke * (inv_distance @ charges)  # N  (1/vacuum_permittivity) * 1/B * elementary_charge, 

#     esp_q = esp.m_as(AU_ESP)
#     if not with_units:
#         return esp_q
#     return esp

# def calculate_esp_monopole_au(
#     grid_coordinates: unit.Quantity,  # N x 3
#     atom_coordinates: unit.Quantity,  # M x 3
#     charges: unit.Quantity,  # M
#     ):
#     #prefactor
#     ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

#     #Ensure everything is in AU and correct dimensions
#     charges = charges.flatten()
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
#     #displacement and distance
#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B
#     distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M
#     inv_distance = 1 / distance  #N, M

#     esp = ke*np.sum(inv_distance * charges[None,:], axis=1)  # (N,M)*(1,M) -> (N,M) numpy broadcasts all charges. Over all atoms  =  Sum over M (atoms), resulting shape: (N,) charges broadcast over each N
    
#     return esp.to(AU_ESP)

# def calculate_esp_dipole_au(
#   grid_coordinates: unit.Quantity,  # N , 3
#   atom_coordinates: unit.Quantity,  # M , 3
#   dipoles: unit.Quantity,  # M , 3       
#   ) -> unit.Quantity:
#     #prefactor
#     ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)

#     #Ensure everything is in AU
#     dipoles = dipoles.to(unit.e*unit.bohr)
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
#     distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
#     inv_distance_cubed = 1 / distance**3 #1/B
#     #Hadamard product for element-wise multiplication
#     dipole_dot = np.sum(displacement * dipoles[None,:,:], axis=-1) # dimless * e.a

#     esp = ke*np.sum(inv_distance_cubed* dipole_dot,axis=1) # e.a/a**2 

#     return esp.to(AU_ESP)

# def calculate_esp_quadropole_au(
#     grid_coordinates: unit.Quantity,  # N x 3
#     atom_coordinates: unit.Quantity,  # M x 3
#     quadrupoles: unit.Quantity,  # M N 
#     ) -> unit.Quantity:
#     #prefactor
#     ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
#     #Ensure everything is in AU
#     quadrupoles = quadrupoles.to(unit.e*unit.bohr*unit.bohr)    
#     grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr
#     atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr

#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N , M , 3 
#     distance = np.linalg.norm(displacement.m, axis=-1)*unit.bohr # N, M 
#     inv_distance = 1 / distance #1/B

#     quadrupole_dot_1 = np.sum(quadrupoles[None,:,:] * displacement[:,:,None],axis=-1)
#     quadrupole_dot_2 = np.sum(quadrupole_dot_1*displacement,axis=-1)
#     esp = ke*np.sum((3*quadrupole_dot_2*(1/2 * inv_distance**5)),axis=-1)

#     return esp.to(AU_ESP)

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
                                                                
from esp_from_conformer import ESPProcessor
import sys
import numpy as np

sys.path.append('/Users/localadmin/Documents/projects/QM_ESP_Psi4')

from source.storage.storage import MoleculePropRecord, MoleculePropStore
from point_charge_esp import calculate_esp
from openff.toolkit.topology import Molecule
from openff.units import unit
from molesp.gui import launch

"""
Smiles list in prop store:
['OCC(O)CO',
 'C#CC',
 'C1CN1',
 'C1COC1',
 'CC#N',
 'CC(=O)[O-]',
 'CC(C)=O',
 'CCCC',
 'CCNCC',
 'CN(C)C',
 'CN=[N+]=[N-]',
 'CNC',
 'COC',
 'CSC',
 'Fc1ccccc1',
 'NCO',
 'Nc1ccccc1',
 'O=[NH+][O-]',
 'Oc1ccccc1',
 'CCl',
 'CF',
 'CO',
 'CS',
 'C1COCO1',
 'c1ccccc1',
 'c1ccncc1',
 'c1ccsc1']
"""

MOLECULE_STR = 'CC(=O)[O-]'
CONFORMER = 0
prop_store = MoleculePropStore("/Users/localadmin/Documents/projects/QM_ESP_Psi4/properties_store.db")
conformer = prop_store.retrieve(MOLECULE_STR)[CONFORMER]
mapped_smiles = conformer.tagged_smiles
partial = prop_store.retrieve_partial(smiles=mapped_smiles,
                           conformer=conformer.conformer)
charges_names = list(partial.keys())
charges = list(partial.values())
charges.extend([conformer.mulliken_charges,conformer.lowdin_charges,conformer.mbis_charges])
charges_names.extend(['mulliken','lowdin','mbis'])
qm_esp = ESPProcessor(prop_store_path = '/Users/localadmin/Documents/projects/QM_ESP_Psi4/properties_store.db', port = 8100, molecule = MOLECULE_STR, conformer = CONFORMER) 
esp, grid, esp_molecule = qm_esp.process_and_launch_qm_esp()
print(esp)
# retrieve on atom charges
# conformer = prop_store.retrieve(MOLECULE_STR)[CONFORMER]
# mapped_smiles = conformer.tagged_smiles
# charge_mol_am1 = Molecule.from_mapped_smiles(mapped_smiles) 
# charge_mol_am1.assign_partial_charges('mmff94', use_conformers=[conformer.conformer_quantity])
# partial_charges = [charge_mol_am1.partial_charges]

qm_esp.add_on_atom_esp(charges,charges_names)








from esp_from_conformer import ESPProcessor
import sys
import numpy as np

sys.path.append('/Users/localadmin/Documents/projects/QM_ESP_Psi4')

from source.storage.storage import MoleculePropRecord, MoleculePropStore
from point_charge_esp import calculate_esp
from openff.toolkit.topology import Molecule
from openff.units import unit

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

qm_esp = ESPProcessor(prop_store_path = '/Users/localadmin/Documents/projects/QM_ESP_Psi4/examples/prop_test_2.db', port = 8000) 
esp, grid = qm_esp.process_and_launch_qm_esp(molecule = MOLECULE_STR, conformer = CONFORMER)

#generate on atom charges for AM1-Muliken on-atom charges
prop_store = MoleculePropStore("/Users/localadmin/Documents/projects/QM_ESP_Psi4/examples/prop_test_2.db")
conformer = prop_store.retrieve(MOLECULE_STR)[CONFORMER]
mapped_smiles = conformer.tagged_smiles
charge_mol_am1 = Molecule.from_mapped_smiles(mapped_smiles) 
charge_mol_am1.assign_partial_charges('am1-mulliken', use_conformers=[conformer.conformer_quantity])
am1_esp_quantity = calculate_esp(grid, conformer.conformer_quantity, charge_mol_am1.partial_charges, with_units= True).to(unit.hartree/unit.e)  
#feed the on atom esp into the ESP visualizer 
am1_esp = ESPProcessor(prop_store_path = '/Users/localadmin/Documents/projects/QM_ESP_Psi4/examples/prop_test_2.db', port = 8100)
am1_esp.on_atom_esp(am1_esp_quantity, MOLECULE_STR, CONFORMER)





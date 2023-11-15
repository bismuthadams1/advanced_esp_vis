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
#The acetate had the highest RMSEv of all the charges
MOLECULE_STR = 'CC(=O)[O-]'
CONFORMER = 0
#load the prop_store so we can 
prop_store = MoleculePropStore("properties_store.db")
#retrieve the first conformer
conformer = prop_store.retrieve(MOLECULE_STR)[CONFORMER]
#retrieve the mapped smiles
mapped_smiles = conformer.tagged_smiles
partial = prop_store.retrieve_partial(smiles=mapped_smiles,
                           conformer=conformer.conformer)
#add all the partial charges into two lists
charges_names = list(partial.keys())
charges = list(partial.values())
#add the charge partitioning charges to the list
charges.extend([conformer.mulliken_charges,conformer.lowdin_charges,conformer.mbis_charges])
charges_names.extend(['mulliken','lowdin','mbis'])
#start the ESPProcesser class
qm_esp = ESPProcessor(prop_store_path = 'properties_store.db', port = 8100, molecule = MOLECULE_STR, conformer = CONFORMER) 
#this will first generate the qm esp, ctrl + c to break the subprocess to generate the on-atom esps and refresh the localhost:8100
esp, grid, esp_molecule = qm_esp.process_and_launch_qm_esp()
qm_esp.add_on_atom_esp(charges,charges_names)








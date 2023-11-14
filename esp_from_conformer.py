import pickle
import sys
import os 
import numpy as np
import asyncio


sys.path.append('/Users/localadmin/Documents/projects/QM_ESP_Psi4')

from source.storage.storage import MoleculePropRecord, MoleculePropStore

from openff.recharge.esp import ESPSettings
from openff.recharge.grids import MSKGridSettings
from openff.recharge.esp.psi4 import Psi4ESPGenerator
from openff.toolkit.topology import Molecule
from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii
from openff.units import unit
from openff.utilities import temporary_cd

from molesp.models import ESPMolecule, Surface
from molesp.gui import launch
from molesp.cli._cli import compute_surface

PORT = 8000
"""
prop_store = MoleculePropStore("/Users/localadmin/Documents/projects/QM_ESP_Psi4/examples/prop_test_2.db")

smiles_list = prop_store.list()

test_mol = prop_store.retrieve(smiles_list[5])[0]
tagged_smiles = test_mol.tagged_smiles
openff_molecule = Molecule.from_mapped_smiles(tagged_smiles)
test_mol.conformer_quantity

vdw_radii = compute_vdw_radii(openff_molecule, radii_type=VdWRadiiType.Bondi)

radii = (
            np.array([[radii] for radii in vdw_radii.m_as(unit.angstrom)])
            * unit.angstrom
        )

vertices, indices = compute_surface(molecule = openff_molecule,
                                    conformer = test_mol.conformer_quantity, 
                                    radii = radii,
                                    radii_scale = 1.4,
                                    spacing = 0.2 * unit.angstrom )


esp_settings = ESPSettings(
            basis="6-31G*", method="hf", grid_settings=MSKGridSettings()
        )

with temporary_cd():

    _, esp, _ = Psi4ESPGenerator._generate(
        openff_molecule,
        test_mol.conformer_quantity,
        vertices * unit.angstrom,
        esp_settings,
        "",
        minimize=False,
        compute_esp=True,
        compute_field=False,
    )

esp_molecule =  ESPMolecule(
    atomic_numbers = [atom.atomic_number for atom in openff_molecule.atoms],
    conformer = test_mol.conformer.flatten().tolist(),
    surface = Surface(
            vertices=vertices.flatten().tolist(), indices=indices.flatten().tolist()
        ),
        esp={"QC ESP": esp.m_as(unit.hartree / unit.e).flatten().tolist()},
    )


launch(esp_molecule, port=PORT)
"""

class ESPProcessor:

    def __init__(self, 
                prop_store_path: str,
                port: int = 8000) -> None:
        self._port = port
        self._prop_store = MoleculePropStore(prop_store_path)
        self.esp_settings = ESPSettings(
            basis="6-31G*", method="hf", grid_settings=MSKGridSettings()
        )
        
    def _get_conformer(self, 
                       molecule: str,
                       conformer: int) -> list[MoleculePropRecord]:
        """
           retrieve molecule properties from the database
        Parameters
        ----------
        molecule
            The smiles string of the molecule
        conformer
            The index of conformer 
            
        Returns
        -------
            list of Molecule Properties.
        
        """
        molecule_props = self._prop_store.retrieve(molecule)[conformer]
        return molecule_props

    def _compute_vdw_radii(self, 
                           openff_molecule: Molecule) -> np.ndarray:
        """
            compute the VdW radii  
        Parameters
        ----------
        openff_molecule
            openff molecule from tagged smiles in db
        Returns
        -------
           Returns VdW radii array.
        """
        vdw_radii = compute_vdw_radii(openff_molecule, radii_type=VdWRadiiType.Bondi) 
        radii = np.array([[r] for r in vdw_radii.m_as(unit.angstrom)]) * unit.angstrom
        return radii

    def _compute_surface(self, 
                         openff_molecule: Molecule, 
                         molecule_props: list[MoleculePropRecord], 
                         radii: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        """"
            Compute the surface to project the ESP on
        Parameters
        ----------
        openff_molecule
            openff molecule from tagged smiles in db
        molecule_props
            list of molecule properties associated with openff_molecule
        radii
            VdW radii 
        Returns
        -------
            Tuple of vertices and indices associated with the surface.
        """

        vertices, indices = compute_surface(
            molecule=openff_molecule,
            conformer=molecule_props.conformer_quantity,
            radii=radii,
            radii_scale=1.4,
            spacing=0.2 * unit.angstrom,
        )
        return vertices, indices

    def _generate_esp(self, 
                      openff_molecule: Molecule,
                      molecule_props: list[MoleculePropRecord],
                      esp_settings: ESPSettings,
                      vertices: np.ndarray) -> tuple[unit.Quantity, unit.Quantity]:
        """
           generate the esp using Psi4ESPGenerator 
        Parameters
        ----------
        openff_molecule
            openff molecule from tagged smiles in db
        molecule_props
            list of molecule properties associated with openff_molecule
        esp_settings
            openff.recharge ESPSettings object containing method and grid settings
        vertices
            vertices associated with ESP surface
        Returns
        -------
            ESP array of units of Angrstrom.
        """

        with temporary_cd():
            _, esp, _  = Psi4ESPGenerator._generate(
                openff_molecule,
                molecule_props.conformer_quantity,
                vertices * unit.angstrom,
                esp_settings,
                "",
                minimize=False,
                compute_esp=True,
                compute_field=False,
            )
        
        grid = vertices * unit.angstrom

        return esp, grid

    def _create_esp_molecule(self, 
                             openff_molecule: Molecule, 
                             molecule_props: list[MoleculePropRecord], 
                             vertices: np.ndarray, 
                             indices: np.ndarray, 
                             esp: unit.Quantity) -> ESPMolecule:
        """
        creates an ESPMolecule class containing all the visualisation information
        Parameter
        ---------
        openff_molecule
            openff molecule from tagged smiles in db
        molecule_props
            list of molecule properties associated with openff_molecule
        vertices
            vertices associated with ESP surface
        indices
            indices associated with ESP surface
        esp
            ESP associated with the molecule
        Returns
        -------
            ESP molecule object
        """
        esp_molecule = ESPMolecule(
            atomic_numbers=[atom.atomic_number for atom in openff_molecule.atoms],
            conformer=molecule_props.conformer.flatten().tolist(),
            surface=Surface(
                vertices=vertices.flatten().tolist(),
                indices=indices.flatten().tolist(),
            ),
            esp={"QC ESP": esp.m_as(unit.hartree / unit.e).flatten().tolist()},
        )
        return esp_molecule

    def process_and_launch_qm_esp(self,
                           molecule: str,
                           conformer: int) -> None:
        conf = self._get_conformer(molecule, conformer)
        openff_molecule = Molecule.from_mapped_smiles(conf.tagged_smiles)

        radii = self._compute_vdw_radii(openff_molecule)

        vertices, indices = self._compute_surface(openff_molecule, conf, radii)

        esp, grid = self._generate_esp(openff_molecule, conf, self.esp_settings, vertices)

        esp_molecule = self._create_esp_molecule(
            openff_molecule, conf, vertices, indices, esp
        )

        launch(esp_molecule, port=self._port)

        return esp, grid, esp_molecule
    
    def on_atom_esp(self,
                    esp: unit.Quantity,
                    molecule: str,
                    conformer: int) -> None:
        
        conf = self._get_conformer(molecule, conformer)
        openff_molecule = Molecule.from_mapped_smiles(conf.tagged_smiles)

        radii = self._compute_vdw_radii(openff_molecule)

        vertices, indices = self._compute_surface(openff_molecule, conf, radii)
        
        esp_molecule2 = self._create_esp_molecule(
            openff_molecule, conf, vertices, indices, esp
        )

        launch(esp_molecule2, port=self._port)

        return esp, esp_molecule2

        

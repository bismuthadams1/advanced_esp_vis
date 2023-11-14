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
from point_charge_esp import calculate_esp

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
                molecule: str,
                conformer: int,
                port: int = 8000,
                grid: unit.Quantity | None =  None,
                qmesp: unit.Quantity | None = None,
                vertices: np.ndarray | None = None,
                indices: np.ndarray | None = None) -> None:
        self._port = port
        self._prop_store = MoleculePropStore(prop_store_path)
        self.esp_settings = ESPSettings(
            basis="6-31G*", method="hf", grid_settings=MSKGridSettings()
        )
        self.conformer = self._get_conformer(molecule, conformer)  
        self.openff_molecule = Molecule.from_mapped_smiles(self.conformer.tagged_smiles)
        self.grid = grid
        self.qmesp = qmesp
        self.vertices = vertices
        self.indices = indices
        self.esp_molecule = None

    @property
    def grid(self):
        return self._grid 
    
    @grid.setter
    def grid(self, value: unit.Quantity) -> None:
        if value is None:
            self._grid = None
        else:
            self._grid = value
    
    @property
    def qmesp(self):
        return self._qmesp
    
    @qmesp.setter
    def qmesp(self, value: unit.Quantity) -> None:
        if value is None:
            self._qmesp = None
        else:
            self._qmesp = value

    @property
    def vertices(self):
        return self._vertices
    
    @vertices.setter
    def vertices(self, value: np.ndarray | None) -> None:
        self._vertices = value

    @property
    def indices(self):
        return self._indices
    
    @indices.setter
    def indices(self, value: np.ndarray | None) -> None:
        self._indices = value

        
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

    def _compute_vdw_radii(self) -> np.ndarray:
        """
            compute the VdW radii  
        Parameters
        ----------
   
        Returns
        -------
           Returns VdW radii array.
        """
        vdw_radii = compute_vdw_radii(self.openff_molecule, radii_type=VdWRadiiType.Bondi) 
        radii = np.array([[r] for r in vdw_radii.m_as(unit.angstrom)]) * unit.angstrom
        return radii

    def _compute_surface(self, 
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
            molecule=self.openff_molecule,
            conformer=self.conformer.conformer_quantity,
            radii=radii,
            radii_scale=1.4,
            spacing=0.2 * unit.angstrom,
        )
        return vertices, indices

    def _generate_esp(self) -> tuple[unit.Quantity, unit.Quantity]:
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
                self.openff_molecule,
                self.conformer.conformer_quantity,
                self.vertices * unit.angstrom,
                self.esp_settings,
                "",
                minimize=False,
                compute_esp=True,
                compute_field=False,
            )
        
        self.grid = self.vertices * unit.angstrom

        return esp, self.grid

    def _create_esp_molecule(self, 
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
            atomic_numbers=[atom.atomic_number for atom in self.openff_molecule.atoms],
            conformer=self.conformer.conformer.flatten().tolist(),
            surface=Surface(
                vertices=self.vertices.flatten().tolist(),
                indices=self.indices.flatten().tolist(),
            ),
            esp={"QC ESP": np.round(esp,7).m_as(unit.hartree / unit.e).flatten().tolist()},
        )
        return esp_molecule

    def process_and_launch_qm_esp(self) -> None:
        """
        Produce QM ESP using the supplied ESPSettings, Molecule, Conformer
        Paramters
        ---------
        molecule 
        """

        radii = self._compute_vdw_radii()

        vertices, indices = self._compute_surface(radii)

        self.vertices = vertices
        self.indices = indices

        esp, grid = self._generate_esp()

        self.qmesp = esp
        self.grid = grid

        #create esp molecule object for visualization
        esp_molecule = self._create_esp_molecule(esp)
        self.esp_molecule = esp_molecule
        
        launch(esp_molecule, port = self._port)

        return esp, grid, self.esp_molecule
    
    def add_on_atom_esp(self,
                        on_atom_charges: list[unit.Quantity],
                        labels: list[str]) -> tuple[unit.Quantity, ESPMolecule]:
        
        for charge_list, label in zip(on_atom_charges, labels):
            on_atom_esp = self._generate_on_atom_esp(charge_list)
            #ensure the on atom esp is at 7dp as visualisation crashes otherwise
            self.esp_molecule.esp[label] = np.round(on_atom_esp,7).m_as(unit.hartree / unit.e).flatten().tolist()
        
        launch(self.esp_molecule, port = self._port + 100)

        return self.esp_molecule
        
    def _generate_on_atom_esp(self,
                              on_atom_charges: list[float]) -> unit.Quantity:
        """"
        takes in a list of on atom charges and produces an ESP for them
        Parameters
        ----------
        on_atom_charges
            list of on atom charges
        Returns
        -------
        on_atom_esp
            on atom esp formed from the conformer and on atom chargers
        """
        
        on_atom_esp =  calculate_esp(self.grid,
                             self.conformer.conformer_quantity,
                             on_atom_charges,
                             with_units= True)

        on_atom_esp.to(unit.hartree/unit.e)
        on_atom_esp = on_atom_esp.magnitude.reshape(-1, 1)
        on_atom_esp = on_atom_esp * unit.hartree/unit.e

        return on_atom_esp

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

        

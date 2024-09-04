from openff.units import unit
import numpy as np

AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

#Function to calculate the ESP from on-atom charges. Taken from Lily Wang's script
def calculate_esp(
    grid_coordinates: unit.Quantity,  # N x 3
    atom_coordinates: unit.Quantity,  # M x 3
    charges: unit.Quantity,  # M
    with_units: bool = False,
) -> unit.Quantity:
    """Calculate ESP from grid"""
    ke = 1 / (4 * np.pi * unit.epsilon_0) # 1/vacuum_permittivity, 1/(e**2 * a0 *Eh)
    
    grid_coordinates = grid_coordinates.reshape((-1, 3)).to(unit.bohr)  #Å to Bohr 
    atom_coordinates = atom_coordinates.reshape((-1, 3)).to(unit.bohr)    #Å to Bohr
    #performs subtraction operation between all grid points and each atom,  provides the displacement vectores. None ensures broadcasting
    displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B, 
    #find L2 norm, ie the distance associated with the displacement vector.
    distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M B, a0
    inv_distance = 1 / distance #1/B

    esp = ke * (inv_distance @ charges)  # N  (1/vacuum_permittivity) * 1/B * elementary_charge, 

    esp_q = esp.m_as(AU_ESP)
    if not with_units:
        return esp_q
    return esp
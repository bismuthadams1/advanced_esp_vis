# from openff.units import unit
# import numpy as np

# AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

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
#     #performs subtraction operation between all grid points and each atom,  provides the displacement vectores. None ensures broadcasting
#     displacement = grid_coordinates[:, None, :] - atom_coordinates[None, :, :]  # N x M x 3 B, 
#     #find L2 norm, ie the distance associated with the displacement vector.
#     distance = (displacement ** 2).sum(axis=-1) ** 0.5  # N x M B, a0
#     inv_distance = 1 / distance #1/B

#     esp = ke * (inv_distance @ charges)  # N  (1/vacuum_permittivity) * 1/B * elementary_charge, 

#     esp_q = esp.m_as(AU_ESP)
#     if not with_units:
#         return esp_q
#     return esp

import jax
import jax.numpy as jnp
from openff.units import unit

# This defines the "atomic units of ESP" just like your original code:
AU_ESP = unit.atomic_unit_of_energy / unit.elementary_charge

# Convert ke = 1/(4*pi*eps_0) into a pure float in the units of [Hartree·Bohr / (e^2)].
# Reasoning: (1/Bohr) * (charge in e) => e/Bohr, multiplied by [Hartree·Bohr/e^2] => Hartree/e
KE_DIMLESS = (1 / (4 * jnp.pi * unit.epsilon_0)).m_as("hartree * bohr / elementary_charge^2")

@jax.jit
def _calculate_esp_jax(grid_coords_bohr, atom_coords_bohr, charges_e):
    """
    JAX-compiled core that expects:
      grid_coords_bohr : (N, 3) float array (already in Bohr, dimensionless)
      atom_coords_bohr : (M, 3) float array (already in Bohr, dimensionless)
      charges_e        : (M,)   float array of charges in e (dimensionless)

    Returns a (N,) array of ESP values in Hartree/e.
    """
    # Broadcasted displacements: shape (N, M, 3)
    displacement = grid_coords_bohr[:, None, :] - atom_coords_bohr[None, :, :]

    # Distance: shape (N, M)
    distance = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    inv_distance = 1.0 / distance

    # Sum over M atoms => shape (N,)
    # This is effectively (inv_distance @ charges_e).
    partial_esp = jnp.einsum("nm,m->n", inv_distance, charges_e)

    # Multiply by the dimensionless ke => final is in (Hartree/e).
    return KE_DIMLESS * partial_esp

def calculate_esp(
    grid_coordinates: unit.Quantity,  # (N, 3)
    atom_coordinates: unit.Quantity,  # (M, 3)
    charges: unit.Quantity,           # (M,)
    with_units: bool = False
):
    """
    Calculate the ESP on a set of grid points (grid_coordinates)
    from point charges at atom_coordinates.

    Returns either:
      - a plain JAX array in Hartree/e (if with_units=False),
      - or an openff.units.Quantity in the same unit (if with_units=True).
    """

    # Convert to Bohr, e, but keep them as NumPy arrays first
    # (You could also go directly to jnp.array(...) if you like.)
    grid_coords_bohr = jnp.array(grid_coordinates.to(unit.bohr).magnitude)
    atom_coords_bohr = jnp.array(atom_coordinates.to(unit.bohr).magnitude)
    charges_e        = jnp.array(charges.to(unit.elementary_charge).magnitude)

    # Call the JAX-compiled core
    esp_hartree_per_e = _calculate_esp_jax(grid_coords_bohr, atom_coords_bohr, charges_e)

    if not with_units:
        # Return dimensionless array in "Hartree/e"
        return esp_hartree_per_e
    else:
        # Wrap back into openff.units; treat the array as in "Hartree/e"
        return esp_hartree_per_e * AU_ESP

import numpy as np
from openff.recharge.grids import MSKGridSettings, GridGenerator
from openff.recharge.utilities.toolkits import VdWRadiiType, compute_vdw_radii
from openff.units import unit
from openff.toolkit.topology import Molecule

def _generate_connolly_sphere(radius: float, density: float) -> np.ndarray:
    """Generates a set of points on a sphere with a given radius and density
    according to the method described by M. Connolly.

    Parameters
    ----------
    radius
        The radius [Angstrom] of the sphere.
    density
        The density [/Angstrom^2] of the points on the sphere.

    Returns
    -------
        The coordinates of the points on the sphere
    """

    # Estimate the number of points according to `surface area * density`
    n_points = int(4 * np.pi * radius * radius * density)

    n_equatorial = int(np.sqrt(n_points * np.pi))  # sqrt (density) * 2 * pi
    n_latitudinal = int(n_equatorial / 2)  # 0 to 180 def so 1/2 points

    phi_per_latitude = np.pi * np.arange(n_latitudinal + 1) / n_latitudinal

    sin_phi_per_latitude = np.sin(phi_per_latitude)
    cos_phi_per_latitude = np.cos(phi_per_latitude)

    n_longitudinal_per_latitude = np.maximum(
        (n_equatorial * sin_phi_per_latitude).astype(int), 1
    )

    sin_phi = np.repeat(sin_phi_per_latitude, n_longitudinal_per_latitude)
    cos_phi = np.repeat(cos_phi_per_latitude, n_longitudinal_per_latitude)

    theta = np.concatenate(
        [
            2 * np.pi * np.arange(1, n_longitudinal + 1) / n_longitudinal
            for n_longitudinal in n_longitudinal_per_latitude
        ]
    )

    x = radius * np.cos(theta) * sin_phi
    y = radius * np.sin(theta) * sin_phi
    z = radius * cos_phi

    return np.stack([x, y, z]).T

def _generate_msk_shells(
   conformer: np.ndarray, radii: np.ndarray, settings: MSKGridSettings
) -> np.ndarray:
    """Generates a grid of points according to the algorithm proposed by Connolly
    using the settings proposed by Merz-Singh-Kollman.

    Parameters
    ----------
    conformer
        The conformer [Angstrom] of the molecule with shape=(n_atoms, 3).
    radii
        The radii [Angstrom] of each atom in the molecule with shape=(n_atoms, 1).
    settings
        The settings that describe how the grid should be generated.

    Returns
    -------
        The coordinates [Angstrom] of the grid with shape=(n_grid_points, 3).
    """

    shells = []

    scale = 1
    atom_spheres = [
        coordinate
        + _generate_connolly_sphere(radius.item() * scale, settings.density)
        for radius, coordinate in zip(radii, conformer)
    ]
    shell = np.vstack(atom_spheres)

    n_grid_points = len(shell)
    n_atoms = len(radii)

    # Build a mask to ensure that grid points generated around an atom aren't
    # accidentally culled due to precision issues.
    exclusion_mask = np.zeros((n_atoms, n_grid_points), dtype=bool)

    offset = 0

    for atom_index, atom_sphere in enumerate(atom_spheres):
        exclusion_mask[atom_index, offset : offset + len(atom_sphere)] = True
        offset += len(atom_sphere)

    shells.append(
        _cull_points(
            conformer, shell, radii * scale, exclusion_mask=exclusion_mask
        )
    )

    return np.vstack(shells)

def _cull_points(
    conformer: np.ndarray,
    grid: np.ndarray,
    inner_radii: np.ndarray,
    outer_radii: np.ndarray | None = None,
    exclusion_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Removes all points that are either within or outside a vdW shell around a
    given conformer.
    """

    from scipy.spatial.distance import cdist

    distances = cdist(conformer, grid)
    exclusion_mask = False if exclusion_mask is None else exclusion_mask

    is_within_shell = np.any(
        (distances < inner_radii) & (~exclusion_mask), axis=0
    )
    is_outside_shell = False

    if outer_radii is not None:
        is_outside_shell = np.all(
            (distances > outer_radii) & (~exclusion_mask), axis=0
        )

    discard_point =  np.logical_or(is_within_shell, is_outside_shell)

    return grid[~discard_point]

def generate(
    molecule: Molecule,
    conformer: unit.Quantity,
    settings: MSKGridSettings,
) -> unit.Quantity:
    """Generates a grid of points in a shell around a specified
    molecule in a given conformer according a set of settings.

    Parameters
    ----------
    molecule
        The molecule to generate the grid around.
    conformer
        The conformer of the molecule with shape=(n_atoms, 3).
    settings
        The settings which describe how the grid should
        be generated.

    Returns
    -------
        The coordinates of the grid with shape=(n_grid_points, 3).
    """

    conformer = conformer.to(unit.angstrom).m

    vdw_radii = compute_vdw_radii(molecule, radii_type=VdWRadiiType.Bondi)
    radii_array = np.array([[radii] for radii in vdw_radii.m_as(unit.angstrom)])

    coordinates = _generate_msk_shells(conformer, radii_array, settings)
    print(coordinates)
    return coordinates * unit.angstrom
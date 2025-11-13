import numpy as np
import spglib

a = 5.43
lattice = np.array([
    [0.0, 0.5*a, 0.5*a],
    [0.5*a, 0.0, 0.5*a],
    [0.5*a, 0.5*a, 0.0]
])
positions = [[0,0,0],[0.25,0.25,0.25]]
numbers = [14,14]  # Si
reciprocal_lattice = 2 * np.pi * np.linalg.inv(lattice).T
print("Reciprocal lattice vectors (in bohr^-1):")
for vec in reciprocal_lattice:
    print(f"[{vec[0]:.4f}, {vec[1]:.4f}, {vec[2]:.4f}]")

cell = (lattice, positions, numbers)
mesh = [2,2,2]
is_shift = [0,0,0]

mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, cell, is_shift=is_shift)
unique_ids, counts = np.unique(mapping, return_counts=True)
total = mesh[0]*mesh[1]*mesh[2]

print("All k-points mapping:")
for i, gid in enumerate(mapping):
    k_frac = grid[i] / np.array(mesh)
    k_cart = np.dot(k_frac, reciprocal_lattice)
    
    # Get irreducible k-point coordinates
    irreducible_k_frac = grid[gid] / np.array(mesh)
    irreducible_k_cart = np.dot(irreducible_k_frac, reciprocal_lattice)
    
    print(f"k{i+1}: [{k_frac[0]:.4f}, {k_frac[1]:.4f}, {k_frac[2]:.4f}] -> irreducible k{gid}: [{irreducible_k_frac[0]:.4f}, {irreducible_k_frac[1]:.4f}, {irreducible_k_frac[2]:.4f}] bohr^-1")

print(f"\nTotal k-points: {total}")
print(f"Irreducible k-points: {len(unique_ids)}")
print(f"Reduction ratio: {len(unique_ids)/total:.4f}")

print(f"\nIrreducible k-points and weights:")
for i, (uid, cnt) in enumerate(zip(unique_ids, counts)):
    k_frac = grid[uid] / np.array(mesh)
    weight = cnt / total
    print(f"k{i+1}: [{k_frac[0]:.4f}, {k_frac[1]:.4f}, {k_frac[2]:.4f}] weight: {weight:.8f} (mult: {cnt})")

print(f"\nCrystal structure analysis:")
print(f"Space group: {spglib.get_spacegroup(cell)}")
print(f"Lattice volume: {np.linalg.det(lattice):.2f} bohr³")
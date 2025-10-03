"""Tool for loading checkpoint and computing network wavefunction values."""

import os
import kfac_jax
import jax
import jax.numpy as jnp
import numpy as np
from ferminet import base_config
import ml_collections
from ferminet import networks
from ferminet import envelopes
from ferminet import psiformer
from ferminet.utils import system
import ferminet.pbc.envelopes as pbc_envelopes
import ferminet.pbc.feature_layer as pbc_feature_layer

def input_params():
    cfg = base_config.default()
    cfg.network.full_det = False

    cfg.checkpoint_path = "qmcjax_ckpt_000010.npz"
    cfg.system.particles = (3, 3, 1)
    cfg.system.charges = (-1., -1., 1)
    cfg.system.masses = (1., 1., 1)

    a = 6.63  # Lattice constant in bohr
    cfg.system.molecule = [
        system.Atom('Li', (0, 0, 0)),
        system.Atom('Li', (0.5*a, 0.5*a, 0.5*a))
    ]

    cfg.system.pbc.apply_pbc = True
    cfg.system.pbc.lattice_vectors = a * np.eye(3)

    return cfg

def check_params(params, cfg):
    assert(len(params['envelope']) == len(cfg.system.particles))
    nk_read = params['envelope'][0]['sigma'].shape[0] // 2
    assert(nk_read == sum(cfg.system.particles))
    cfg.network.determinants = params['envelope'][0]['sigma'].shape[1] // cfg.system.particles[0]

    return cfg

def read_npz(file_path):
    """Read npz checkpoint file"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    ckpt_data = np.load(file_path, allow_pickle=True)  # Use np.load instead of jnp.load
    params = ckpt_data['params'].item()
    data = networks.FermiNetData(**ckpt_data['data'].item())
    return params, data


def create_network_from_config(cfg: ml_collections.ConfigDict):
    """Create network structure from configuration"""
    if cfg.system.pyscf_mol:
        cfg.update(
            system.pyscf_mol_to_internal_representation(cfg.system.pyscf_mol))
    
    atoms = jnp.stack([jnp.array(atom.coords) for atom in cfg.system.molecule])
    charges = jnp.array([atom.charge for atom in cfg.system.molecule])
    use_complex = cfg.network.get('complex', False)

    # Create feature layer
    if not cfg.system.pbc.apply_pbc:
        feature_layer = networks.make_ferminet_features(
            natoms=charges.shape[0],
            nspins=cfg.system.particles,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
        )
    else:
        if len(cfg.system.molecule) == 1 and cfg.system.molecule[0].charge == 0:
            include_r_ae = False
        else:
            include_r_ae = True
        feature_layer = pbc_feature_layer.make_pbc_feature_layer(
            natoms=charges.shape[0],
            nspins=cfg.system.particles,
            ndim=cfg.system.ndim,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            lattice=cfg.system.pbc.lattice_vectors,
            include_r_ae=include_r_ae
        )
    
    # Create envelope function
    if not cfg.system.pbc.apply_pbc:
        envelope = envelopes.make_isotropic_envelope()
    else:
        kpoints = pbc_envelopes.make_kpoints(
            cfg.system.pbc.lattice_vectors,
            cfg.system.particles,
            cfg.system.pbc.min_kpoints
        )
        envelope = pbc_envelopes.make_multiwave_envelope(kpoints)
    
    # Create network
    if cfg.network.network_type == 'ferminet':
        network = networks.make_fermi_net(
            cfg.system.particles,
            charges,  # Use local charges array, not cfg.system.charges
            ndim=cfg.system.ndim,
            particle_masses=cfg.system.masses,
            particle_charges=cfg.system.charges,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            full_det=cfg.network.full_det,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=use_complex,
            **cfg.network.ferminet,
        )
    elif cfg.network.network_type == 'psiformer':
        network = psiformer.make_fermi_net(
            cfg.system.particles,
            charges,  # Use local charges array, not cfg.system.charges
            ndim=cfg.system.ndim,
            particle_masses=cfg.system.masses,
            particle_charges=cfg.system.charges,
            determinants=cfg.network.determinants,
            states=cfg.system.states,
            envelope=envelope,
            feature_layer=feature_layer,
            jastrow=cfg.network.get('jastrow', 'default'),
            bias_orbitals=cfg.network.bias_orbitals,
            rescale_inputs=cfg.network.get('rescale_inputs', False),
            complex_output=use_complex,
            **cfg.network.psiformer,
        )
    else:
        raise ValueError(f"Unknown network type: {cfg.network.network_type}")
    
    return network, atoms, charges


def compute_wavefunction_value(network, params, positions, spins, atoms, charges):
    """Compute wavefunction value"""
    
    output = network.apply(params, positions, spins, atoms, charges)
    
    return output

def evaluate_wavefunction_batch(network, params, positions_batch, spins_batch, atoms, charges):
    """compute wavefunction values for a batch of inputs"""

    batch_network = jax.vmap(network.apply, in_axes=(None, 0, 0, 0, 0), out_axes=0)

    return batch_network(params, positions_batch, spins_batch, atoms, charges)

def main_example():
    """Example usage"""
    # 1. Load checkpoint parameters
    cfg = input_params()
    checkpoint_path = cfg.checkpoint_path 
    params, data = read_npz(checkpoint_path)
    
    # 2. Create configuration from parameters
    cfg = check_params(params, cfg)

    # 3. Create network
    network, atoms, charges = create_network_from_config(cfg)
    
    # 4. Prepare input data (example)
    spins = data.spins[0, 0]
    positions = data.positions[0, 0]
    charges = data.charges[0, 0]
    atoms = data.atoms[0, 0]

    batch_spins = data.spins[0]
    batch_positions = data.positions[0]
    batch_charges = data.charges[0]
    batch_atoms = data.atoms[0]

    # 5. Compute wavefunction value
    wavefunction_value = None
    batch_value = None
    wavefunction_value = compute_wavefunction_value(
        network, params, positions, spins, atoms, charges
    )

    batch_value = evaluate_wavefunction_batch(
        network, params, batch_positions, batch_spins, batch_atoms, batch_charges
    )

    if wavefunction_value is not None:
        print(f"Wavefunction value: {wavefunction_value}")
    if batch_value is not None:
        print(f"Batch wavefunction values: {batch_value}")

if __name__ == "__main__":
    main_example()
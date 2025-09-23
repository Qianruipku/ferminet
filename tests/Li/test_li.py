"""Tests for He atom training and checkpoint comparison."""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path

import numpy as np
from absl.testing import absltest
import pytest
import sys
from pyscf import gto
from ferminet.pbc import envelopes

# Add tools directory to path for importing the base class
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from reference_test_base import ReferenceTestMixin

from ferminet.utils import system
from ferminet import base_config
from ferminet import train


class HeAtomTest(absltest.TestCase, ReferenceTestMixin):
    """Test He atom training and checkpoint validation."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.init_reference_management()
        
        # Create temporary directory for test outputs
        self.test_dir = "_tmp"
        self.ref_dir = str(Path(__file__).parent) + "/restart"

        # Set up basic He atom configuration
        self.cfg = base_config.default()
        a = 5.0  # Lattice constant in bohr
        self.cfg.system.electrons = (1, 1)

        # 1*1*1 bcc
        self.cfg.system.molecule = [system.Atom('Li', (0, 0, 0)),
                                    system.Atom('Li', (0.5*a, 0.5*a, 0.5*a))]

        lattice = a * np.eye(3)
        kpoints = envelopes.make_kpoints(lattice, self.cfg.system.electrons)

        self.cfg.system.use_pp = True
        self.cfg.system.pp.symbols = ['Li']

        mol = gto.Mole()
        mol.atom = [[atom.symbol, atom.coords] for atom in self.cfg.system.molecule]

        atoms = list(set([atom.symbol for atom in self.cfg.system.molecule]))
        pseudo_atoms = self.cfg.system.pp.symbols if self.cfg.system.use_pp else []
        mol.basis = {
            atom:
            self.cfg.system.pp.basis if atom in pseudo_atoms else 'cc-pvdz'
            for atom in atoms
        }
        mol.ecp = {
            atom: self.cfg.system.pp.type
            for atom in atoms if atom in pseudo_atoms
        }
        mol.charge = 0
        mol.spin = 0
        mol.unit = 'bohr'
        mol.build()
        self.cfg.system.pyscf_mol = mol

        pp_symbols = self.cfg.system.get('pp', {'symbols': None}).get('symbols')
        self.cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian.local_energy"
        self.cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": False, "pp_type": self.cfg.system.get('pp', {'type': 'ccecp'}).get('type'),
              "pp_symbols": pp_symbols if self.cfg.system.get('use_pp') else None}
        self.cfg.network.make_feature_layer_fn = (
              "ferminet.pbc.feature_layer.make_pbc_feature_layer")
        self.cfg.network.make_feature_layer_kwargs = {
              "lattice": lattice,
          }
        self.cfg.network.make_envelope_fn = (
              "ferminet.pbc.envelopes.make_multiwave_envelope")
        self.cfg.network.make_envelope_kwargs = {"kpoints": kpoints}

        self.cfg.system.pbc.apply_pbc = True
        self.cfg.system.pbc.lattice_vectors = lattice
        self.cfg.network.full_det = True

        # Small network for fast testing
        self.cfg.network.ferminet.hidden_dims = ((16, 4),) * 1
        self.cfg.network.determinants = 2
        self.cfg.batch_size = 16
        self.cfg.pretrain.iterations = 0
        self.cfg.mcmc.burn_in = 5
        self.cfg.optim.iterations = 5
        
        # Make training deterministic for reproducible results
        self.cfg.debug.deterministic = True
        
        # Set paths
        self.cfg.log.save_path = self.test_dir
        self.cfg.log.restore_path = self.ref_dir
        self.cfg.log.save_freq = 4

    def tearDown(self):
        """Clean up test environment and optionally update reference files."""
        self.cleanup_reference_management()
        
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        super().tearDown()

    def test_checkpoint_comparison_with_reference(self):
        """Compare generated checkpoint with reference file."""
        # Define paths
        test_dir_path = Path(__file__).parent
        ref_npz_file = test_dir_path / "ref_qmcjax_ckpt_000004.npz"
        npz_file = os.path.join(self.test_dir, "qmcjax_ckpt_000004.npz")
        csv_file = os.path.join(self.test_dir, "train_stats.csv")
        ref_csv_file = test_dir_path / "ref_train_stats.csv"
        
        # Register files for potential reference updating
        self._register_generated_file("ref_qmcjax_ckpt_000004.npz", npz_file)
        self._register_generated_file("ref_train_stats.csv", csv_file)
        
        # Run training to generate checkpoint
        train.train(self.cfg)
        
        # Use the base class comparison method
        comparisons = [
            {
                'generated': npz_file,
                'reference': str(ref_npz_file),
                'type': 'npz'
            },
            {
                'generated': csv_file,
                'reference': str(ref_csv_file),
                'type': 'csv'
            }
        ]
        
        # Compare with references (automatically handles UPDATE_REFERENCES mode)
        self.compare_with_references(
            comparisons=comparisons,
            tolerance="1e-5",
            skip_fields=["opt_state"]
        )

if __name__ == '__main__':
    absltest.main()
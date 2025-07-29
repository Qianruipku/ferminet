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

# Add tools directory to path for importing the base class
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from reference_test_base import ReferenceTestMixin

from ferminet.utils import system
from ferminet import base_config
from ferminet import train


class PositronTest(absltest.TestCase, ReferenceTestMixin):
    """Test positron training and checkpoint validation."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.init_reference_management()
        
        # Create temporary directory for test outputs
        self.test_dir = "_tmp"

        # Set up basic He atom configuration
        self.cfg = base_config.default()
        self.cfg.system.particles = (3, 3, 1)
        self.cfg.system.charges = (-1., -1., 1.)
        self.cfg.system.masses = (1., 1., 1.)
        a = 6.63
        self.cfg.system.molecule = [system.Atom('Li', (0, 0, 0)),
                                    system.Atom('Li', (0.5*a, 0.5*a, 0.5*a))]
        self.cfg.pretrain.method = None
        self.cfg.system.pbc.lattice_vectors = 6.63 * np.eye(3)
        self.cfg.system.pbc.apply_pbc = True
        self.cfg.network.full_det = False

        # Small network for fast testing
        self.cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
        self.cfg.network.determinants = 2
        self.cfg.batch_size = 32
        self.cfg.mcmc.burn_in = 0
        self.cfg.mcmc.sample_all = True
        self.cfg.optim.optimizer = "adam"
        self.cfg.optim.iterations = 5
        
        # Make training deterministic for reproducible results
        self.cfg.debug.deterministic = True
        
        # Set paths
        self.cfg.log.save_path = self.test_dir
        self.cfg.log.save_freq = 2

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
        pcf_file = os.path.join(self.test_dir, "pcf_final.txt")
        ref_pcf_file = test_dir_path / "ref_pcf.txt"
        apmd_file = os.path.join(self.test_dir, "apmd_final.txt")
        ref_apmd_file = test_dir_path / "ref_apmd.txt"

        
        # Register files for potential reference updating
        self._register_generated_file("ref_qmcjax_ckpt_000004.npz", npz_file)
        self._register_generated_file("ref_train_stats.csv", csv_file)
        self._register_generated_file("ref_pcf.txt", pcf_file)
        self._register_generated_file("ref_apmd.txt", apmd_file)

        # train the model
        train.train(self.cfg)

        # Run inference to generate PCF
        self.cfg.optim.optimizer="none"
        self.cfg.optim.iterations = 5
        self.cfg.observables.pcf.calculate = True
        self.cfg.observables.pcf.rmax = 4
        self.cfg.observables.apmd.calculate = True
        self.cfg.observables.apmd.ecut = 10.0
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
            },
            {
                'generated': pcf_file,
                'reference': str(ref_pcf_file),
                'type': 'txt'
            },
            {
                'generated': apmd_file,
                'reference': str(ref_apmd_file),
                'type': 'txt'
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
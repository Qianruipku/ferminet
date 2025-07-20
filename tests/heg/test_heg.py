import os
import sys
from pathlib import Path
from absl.testing import absltest

# Add tools directory to path for importing the base class
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
from reference_test_base import ReferenceTestMixin

from ferminet.utils import system
from ferminet import base_config
from ferminet import train
from ferminet.pbc import envelopes
import numpy as np

def _sc_lattice_vecs(rs: float, nelec: int) -> np.ndarray:
  """Returns simple cubic lattice vectors with Wigner-Seitz radius rs."""
  volume = (4 / 3) * np.pi * (rs**3) * nelec
  length = volume**(1 / 3)
  return length * np.eye(3)

class HEGTest(absltest.TestCase, ReferenceTestMixin):
    """Example test for HEG using the reference management system."""

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        self.init_reference_management()
        
        # Create temporary directory for test outputs
        self.test_dir = "_tmp"
        
        # Set up basic HEG configuration
        self.cfg = base_config.default()
        self.cfg.system.electrons = (2, 2)
        self.cfg.system.molecule = [system.Atom("X", (0., 0., 0.))]
        self.cfg.pretrain.method = None
        lattice = _sc_lattice_vecs(1.0, sum(self.cfg.system.electrons))
        kpoints = envelopes.make_kpoints(lattice, self.cfg.system.electrons)
        self.cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian.local_energy"
        self.cfg.system.make_local_energy_kwargs = {"lattice": lattice, "heg": True}
        self.cfg.network.make_feature_layer_fn = (
            "ferminet.pbc.feature_layer.make_pbc_feature_layer")
        self.cfg.network.make_feature_layer_kwargs = {
            "lattice": lattice,
            "include_r_ae": False
        }
        self.cfg.network.make_envelope_fn = (
            "ferminet.pbc.envelopes.make_multiwave_envelope")
        self.cfg.network.make_envelope_kwargs = {"kpoints": kpoints}
        self.cfg.network.full_det = True

        # Small network for fast testing
        self.cfg.network.ferminet.hidden_dims = ((16, 4),) * 2
        self.cfg.network.determinants = 2
        self.cfg.batch_size = 32
        self.cfg.pretrain.iterations = 5
        self.cfg.mcmc.burn_in = 5
        self.cfg.optim.iterations = 5
        
        # Make training deterministic
        self.cfg.debug.deterministic = True
        
        # Set paths
        self.cfg.log.save_path = self.test_dir
        # self.cfg.log.restore_path = self.test_dir
        self.cfg.log.save_freq = 4

    def tearDown(self):
        """Clean up test environment."""
        self.cleanup_reference_management() 
        
        if os.path.exists(self.test_dir):
            import shutil
            shutil.rmtree(self.test_dir)
            
        super().tearDown()

    def test_heg_training_checkpoint_comparison(self):
        """Test HEG training and compare with reference files."""
        # Define file paths
        npz_file = os.path.join(self.test_dir, "qmcjax_ckpt_000004.npz")
        csv_file = os.path.join(self.test_dir, "train_stats.csv")
        
        test_dir_path = Path(__file__).parent
        ref_npz_file = test_dir_path / "ref_qmcjax_ckpt_000004.npz"
        ref_csv_file = test_dir_path / "ref_train_stats.csv"
        
        # Register files for potential reference updating
        self._register_generated_file("ref_qmcjax_ckpt_000004.npz", npz_file)
        self._register_generated_file("ref_train_stats.csv", csv_file)
        
        # Run training
        train.train(self.cfg)

        comparisons = [
            {
                'generated': npz_file,
                'reference': str(ref_npz_file)
            },
            {
                'generated': csv_file,
                'reference': str(ref_csv_file)
            }
        ]
        
        self.compare_with_references(
            comparisons=comparisons,
            tolerance="1e-5",
            skip_fields=["opt_state"]
        )

if __name__ == '__main__':
    absltest.main()
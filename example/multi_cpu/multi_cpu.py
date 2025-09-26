import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

from ferminet import train
import jax
from ferminet.configs import ch4
from absl import logging
from absl import app
from absl import flags
import sys  

FLAGS = flags.FLAGS
flags.DEFINE_integer('ntask', 1, 'Number of tasks for distributed training')
flags.DEFINE_integer('ndevice', 2, 'Number of devices per host (xla_force_host_platform_device_count)')

def main(argv):
    ntask = FLAGS.ntask
    ndevice = FLAGS.ndevice
    
    # Set XLA device count based on ndevice parameter
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={ndevice}"
    
    jax.distributed.initialize(
        coordinator_address="127.0.0.1:1234", # coordinator address
        num_processes=ntask,  # number of tasks
        process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),  # task ID
        local_device_ids=None # use default device assignment
    )
    cfg = ch4.get_config()
    cfg.batch_size = 32  # total batch size across all devices
    cfg.pretrain.iterations = 0
    cfg.optim.iterations = 101
    cfg.log.restore_path = "train"
    cfg.log.save_path = "train"
    train.train(cfg)

if __name__ == "__main__":
    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)
    app.run(main)
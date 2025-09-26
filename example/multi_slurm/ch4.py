from ferminet import train
import jax
from ferminet.configs import ch4
from absl import logging
from absl import app
from absl import flags
import sys
import os

FLAGS = flags.FLAGS
flags.DEFINE_string('server_addr', '',
                    help=('Enables multihost calculations if given. '
                          'Server ip address of host node'))

node_id = os.environ['SLURM_NODEID']
visible_devices = [int(gpu) for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

if __name__ == "__main__":
    flags.FLAGS.mark_as_parsed()
    
    jax.distributed.initialize(
        coordinator_address=FLAGS.server_addr,
        local_device_ids=visible_devices)

    logging.get_absl_handler().python_handler.stream = sys.stdout
    logging.set_verbosity(logging.INFO)

    cfg = ch4.get_config()
    cfg.batch_size = 32  # total batch size across all devices
    cfg.pretrain.iterations = 0
    cfg.optim.iterations = 101
    cfg.log.restore_path = "train"
    cfg.log.save_path = "train"
    train.train(cfg)
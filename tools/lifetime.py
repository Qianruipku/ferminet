import pyblock

import numpy as np

 

burn_in = 100

if __name__ == "__main__":

    data = np.loadtxt("ann_rate.txt", skiprows=1)

    ann_rate = np.real(data[burn_in:, 1])

    reblock_data = pyblock.blocking.reblock(ann_rate)

    opt = pyblock.blocking.find_optimal_block(len(ann_rate), reblock_data)

    # Fallback to the last block if pyblock cannot determine an optimal one.
    block_i = int(opt[0])

    mean = reblock_data[block_i].mean
    stderr = reblock_data[block_i].std_err
    


    inv_mean = 1000.0 / mean if mean != 0 else float('inf')
    inv_stderr = 1000.0 * stderr / (mean * mean) if mean != 0 else float('inf')

    print(f"Optimal block size: {block_i}, mean annihilation rate: {mean}, stderr: {stderr}")
    print(f"Inverse mean (lifetime): {inv_mean}, inverse stderr: {inv_stderr}")
import pyblock

import numpy as np

 

burn_in = 100

if __name__ == "__main__":

    data = np.loadtxt("ann_rate.txt", skiprows=1)

    ann_rate = np.real(data[burn_in:, 1])
    reblock_data = pyblock.blocking.reblock(ann_rate)
    opt = pyblock.blocking.find_optimal_block(len(ann_rate), reblock_data)
    block_i = int(opt[0])
    mean = reblock_data[block_i].mean
    stderr = reblock_data[block_i].std_err
    print(f"Mean: {mean}, stderr: {stderr}, relative error: {stderr/mean if mean != 0 else float('inf')}")
    inv_mean = 1000.0 / mean if mean != 0 else float('inf')
    inv_stderr = inv_mean * (stderr/mean) if inv_mean != float('inf') else float('inf')

    if data.shape[1] > 2:
        prob = np.real(data[burn_in:, 2])
        reblock_prob = pyblock.blocking.reblock(prob)
        opt_prob = pyblock.blocking.find_optimal_block(len(prob), reblock_prob)
        block_i_prob = int(opt_prob[0])
        mean_prob = reblock_prob[block_i_prob].mean
        stderr_prob = reblock_prob[block_i_prob].std_err
        print(f"Mean_prob: {mean_prob}, stderr_prob: {stderr_prob}, relative error: {stderr_prob/mean_prob if mean_prob != 0 else float('inf')}")
        inv_mean = 1000.0 / mean * mean_prob if mean != 0 else float('inf')
        inv_stderr = inv_mean * np.sqrt((stderr/mean)**2 + (stderr_prob/mean_prob)**2) if inv_mean != float('inf') else float('inf')


    print(f"Lifetime: {inv_mean} ps, inverse stderr: {inv_stderr} ps")
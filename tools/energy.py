import pyblock

import numpy as np

 

burn_in = 100

if __name__ == "__main__":

    _, E, *_ = np.loadtxt(f"train_stats.csv", unpack = True, skiprows = 1, delimiter = ",", dtype = np.complex128)

    E = np.real(E[burn_in:])

    reblock_data = pyblock.blocking.reblock(E)

    opt = pyblock.blocking.find_optimal_block(len(E), reblock_data)

 

    f = open("energy.txt", 'w')

    f.write(str(reblock_data[opt[0]]))

    f.close()
import numpy as np
import jax.numpy as jnp
from ferminet import networks
def print_all_keys(d: dict, indent: int = 0):
    """Prints all keys in a dictionary."""
    for key, value in d.items():
        print(' ' * indent + str(key) + ' :')
        if isinstance(value, dict):
            print_all_keys(value, indent + 2)
        elif isinstance(value, list):
            print_list_structure(value, indent + 2)
        else:
            print(' ' * (indent + 2) + str(type(value)))

def print_list_structure(lst: list, indent: int = 0):
    """Prints the structure of a list."""
    for i, item in enumerate(lst):
        print(' ' * indent + f'[{i}] :')
        if isinstance(item, dict):
            print_all_keys(item, indent + 2)
        elif isinstance(item, list):
            print_list_structure(item, indent + 2)
        else:
            print(' ' * (indent + 2) + str(type(item)))

def compare_arrays(arr, name: str):
    len_arr = arr.shape[0]
    for i in range(1, len_arr):
        is_equal = jnp.array_equal(arr[0], arr[i])
        if not is_equal:
            print(f"{name}[0] and {name}[{i}] are not equal.")
            print(f"{name}[0]: {arr[0]}")
            print(f"{name}[{i}]: {arr[i]}")

def compare_params(params):
    """Compares parameters in different devices."""
    # envelope
    for envelope in params['envelope']:
        pi_array = envelope['pi']
        compare_arrays(pi_array, 'pi')
        sigma_array = envelope['sigma']
        compare_arrays(sigma_array, 'sigma')
    
    # layers.streams
    for stream in params['layers']['streams']:
        if 'double' in stream:
            double_dict = stream['double']
            if 'w' in double_dict:
                w_array = double_dict['w']
                compare_arrays(w_array, 'double_w')
            if 'b' in double_dict:
                b_array = double_dict['b']
                compare_arrays(b_array, 'double_b')
        if 'single' in stream:
            single_dict = stream['single']
            if 'w' in single_dict:
                w_array = single_dict['w']
                compare_arrays(w_array, 'single_w')
            if 'b' in single_dict:
                b_array = single_dict['b']
                compare_arrays(b_array, 'single_b')
    # orbitals
    for orbital in params['orbital']:
        if 'w' in orbital:
            w_array = orbital['w']
            compare_arrays(w_array, 'w')
                


            

filename = "qmcjax_ckpt_009900.npz"
with open(filename, 'rb') as f:
    ckpt_data = np.load(f, allow_pickle=True)
    # Retrieve data from npz file. Non-array variables need to be converted back
    # to natives types using .tolist().
    t = ckpt_data['t'].tolist() + 1  # Return the iterations completed.
    data = networks.FermiNetData(**ckpt_data['data'].item())
    params = ckpt_data['params'].tolist()
    opt_state = ckpt_data['opt_state'].tolist()
    mcmc_width = jnp.array(ckpt_data['mcmc_width'].tolist())

    # compare_params(params)
    # compare_params(opt_state.velocities)
    print_all_keys(opt_state.estimator_state)
    # print(opt_state)
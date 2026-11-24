from pathlib import Path

import pyblock

import numpy as np

burn_in = 100


def load_twist_data(filenames):
    twist_weights = []
    twist_data = []
    for filename in filenames:
        with open(filename) as twist_file:
            weight_header = twist_file.readline().strip()
        prefix = "# twist_weight:"
        if not weight_header.startswith(prefix):
            raise ValueError(f"{filename} is missing the twist weight header")
        twist_weights.append(float(weight_header[len(prefix):]))
        twist_data.append(np.loadtxt(filename, comments="#", ndmin=2))

    reference_data = twist_data[0]
    for filename, data in zip(filenames[1:], twist_data[1:]):
        if data.shape != reference_data.shape:
            raise ValueError(f"{filename} has a different shape than "
                             f"{filenames[0]}")
        if not np.array_equal(data[:, 0], reference_data[:, 0]):
            raise ValueError(f"{filename} has different step values than "
                             f"{filenames[0]}")

    twist_weights = np.asarray(twist_weights)
    if not np.all(np.isfinite(twist_weights)) or np.any(twist_weights < 0):
        raise ValueError("Twist weights must be finite and non-negative")
    if np.sum(twist_weights) <= 0:
        raise ValueError("The sum of twist weights must be positive")

    return twist_weights, np.stack(twist_data)


def blocked_mean(values):
    reblock_data = pyblock.blocking.reblock(values)
    optimal_block = pyblock.blocking.find_optimal_block(len(values), reblock_data)
    if np.isnan(optimal_block[0]):
        return np.mean(values), None
    block = reblock_data[int(optimal_block[0])]
    return block.mean, block.std_err


def weighted_blocked_mean(twist_values, twist_weights):
    means = np.asarray([blocked_mean(values)[0] for values in twist_values])
    weighted_mean = np.average(means, weights=twist_weights)

    weighted_series = np.average(
        twist_values, axis=0, weights=twist_weights)
    _, stderr = blocked_mean(weighted_series)
    return weighted_mean, stderr


if __name__ == "__main__":

    twist_files = sorted(
        Path(".").glob("ann_rate_twist_*.txt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]))
    if not twist_files:
        raise FileNotFoundError("No ann_rate_twist_*.txt files found")
    twist_weights, twist_data = load_twist_data(twist_files)
    if burn_in >= twist_data.shape[1]:
        raise ValueError("Burn-in removes all annihilation-rate samples")

    ann_rate = np.real(twist_data[:, burn_in:, 1])
    mean, stderr = weighted_blocked_mean(ann_rate, twist_weights)
    if stderr is not None:
        relative_error = stderr / mean if mean != 0 else float('inf')
        print(f"Weighted mean: {mean}, stderr: {stderr}, "
              f"relative error: {relative_error}")
    inv_mean = 1000.0 / mean if mean != 0 else float('inf')
    inv_stderr = None
    if stderr is not None and mean != 0:
        inv_stderr = abs(inv_mean * stderr / mean)

    if twist_data.shape[2] > 2:
        prob = np.real(twist_data[:, burn_in:, 2])
        mean_prob, stderr_prob = weighted_blocked_mean(prob, twist_weights)
        if stderr_prob is not None:
            relative_error = (
                stderr_prob / mean_prob if mean_prob != 0 else float('inf'))
            print(f"Weighted mean_prob: {mean_prob}, "
                  f"stderr_prob: {stderr_prob}, "
                  f"relative error: {relative_error}")
        inv_mean = 1000.0 * mean_prob / mean if mean != 0 else float('inf')
        if (stderr is not None and stderr_prob is not None and
                mean != 0 and mean_prob != 0):
            inv_stderr = abs(inv_mean) * np.sqrt(
                (stderr / mean)**2 + (stderr_prob / mean_prob)**2)
    if inv_stderr is not None:
        print(f"Lifetime: {inv_mean} ps, inverse stderr: {inv_stderr} ps")
    else:
        print(f"Lifetime: {inv_mean} ps")
    
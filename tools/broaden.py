#!/usr/bin/env python3
"""Gaussian broadening (convolution) tool

Reads a two-column data file (x y), convolves y with a Gaussian
kernel of specified FWHM (in the same units as x) and writes the
broadened result to an output file.

Usage example:
  python3 tools/broaden.py data.txt --fwhm 0.5 -o data_broadened.txt

The convolution is performed by numerical integration on the input
grid. The kernel is normalized for each output point so that the
integral of the kernel over x is 1 (using dx from the input grid).
This works for non-uniform x spacing. For large files and uniform
spacing an FFT-based approach would be faster; this version keeps
the implementation simple and robust.
"""
import argparse
import math
import numpy as np
import sys


def compute_cell_widths(x: np.ndarray) -> np.ndarray:
    """Compute effective dx for each sample using half-distance to neighbors.
    For interior points use (x[i+1]-x[i-1]) / 2, for endpoints use forward/backward diff.
    """
    n = x.size
    if n < 2:
        return np.array([1.0])
    widths = np.empty(n, dtype=float)
    widths[0] = x[1] - x[0]
    widths[-1] = x[-1] - x[-2]
    if n > 2:
        widths[1:-1] = 0.5 * (x[2:] - x[:-2])
    return widths


def gaussian_broaden(x: np.ndarray, y: np.ndarray, fwhm: float, nsigma: float = 8.0) -> np.ndarray:
    if fwhm <= 0:
        # no broadening
        return y.copy()

    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    n = x.size
    widths = compute_cell_widths(x)
    out = np.empty_like(y, dtype=float)
    norm_prefactor = math.sqrt(2.0 * math.pi) * sigma
    cutoff = nsigma * sigma

    # For each output x_i, convolve using local neighborhood within cutoff
    for i in range(n):
        xi = x[i]
        left = xi - cutoff
        right = xi + cutoff
        # select indices in window
        idx = np.nonzero((x >= left) & (x <= right))[0]
        if idx.size == 0:
            out[i] = 0.0
            continue
        dxs = widths[idx]
        dxs = dxs.astype(float)
        ex = (x[idx] - xi) / sigma
        g = np.exp(-0.5 * ex * ex) / norm_prefactor
        weights = g * dxs
        wsum = weights.sum()
        if wsum == 0.0:
            out[i] = 0.0
        else:
            out[i] = float((weights * y[idx]).sum() / wsum)

    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gaussian broadening of (x,y) data by FWHM")
    parser.add_argument('input', help='input data file (two columns: x y)')
    parser.add_argument('--fwhm', type=float, required=True, help='Gaussian FWHM in same units as x')
    parser.add_argument('-o', '--output', help='output filename (default: "broad_" + input)')
    parser.add_argument('--nsigma', type=float, default=8.0, help='kernel cutoff in multiples of sigma (default 8)')
    parser.add_argument('--skiprows', type=int, default=0, help='number of header lines to skip when reading input')
    args = parser.parse_args(argv)

    try:
        data = np.loadtxt(args.input, comments='#', skiprows=args.skiprows)
    except Exception as e:
        print(f"Error reading input file '{args.input}': {e}", file=sys.stderr)
        sys.exit(2)


    if data.shape[1] < 2:
        print("Input file must contain at least two columns (x y)", file=sys.stderr)
        sys.exit(2)
    x = data[:, 0].astype(float)
    y = data[:, 1].astype(float)

    # ensure x is sorted
    sort_idx = np.argsort(x)
    if not np.all(sort_idx == np.arange(x.size)):
        x = x[sort_idx]
        y = y[sort_idx]

    # If the input only contains x >= 0 assume the data represents an even function
    # and mirror it to negative x before convolution. This improves behavior near x=0.
    x_work = x
    y_work = y
    mirrored = False
    if x.size > 0 and np.all(x >= 0.0):
        # create mirrored negative side. Avoid duplicating x==0 if present.
        if np.isclose(x[0], 0.0):
            x_neg = -x[1:][::-1]
            y_neg = y[1:][::-1]
        else:
            x_neg = -x[::-1]
            y_neg = y[::-1]

        if x_neg.size > 0:
            x_work = np.concatenate((x_neg, x))
            y_work = np.concatenate((y_neg, y))
            mirrored = True

    yb_work = gaussian_broaden(x_work, y_work, args.fwhm, nsigma=args.nsigma)

    # If we mirrored, keep only the portion corresponding to original (non-negative) x
    if mirrored:
        yb = yb_work[-x.size:]
    else:
        yb = yb_work

    outname = args.output if args.output else 'broad_' + args.input
    try:
        np.savetxt(outname, np.column_stack((x, yb)), fmt='%.6f %.6e')
    except Exception as e:
        print(f"Error writing output file '{outname}': {e}", file=sys.stderr)
        sys.exit(3)

    print(f"Wrote broadened data to: {outname}")


if __name__ == '__main__':
    main()

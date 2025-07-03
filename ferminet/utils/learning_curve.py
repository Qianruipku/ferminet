import matplotlib.pyplot as plt
import numpy as np
from glob import glob

w = 1000
def moving_average(x, w):
  return np.convolve(x, np.ones(w), 'valid') / w

def learning_curve(path, true_E = None, log = False, lim = False):
  files = sorted(glob(f"{path}/train_stats_*.csv"))
  E = []
  for i in range(len(files)):
    _, Ei, *_ = np.loadtxt(files[i], skiprows = 1, unpack = True, delimiter = ",")
    E = E + list(Ei)

  plt.figure()
  plt.title("Learning curve")
  if not log:
    plt.plot(moving_average(E, w), c = "black")
    plt.ylabel("Energy")
    if true_E != None:
      plt.axhline(true_E, c = "black", linestyle = "--")
  else:
    plt.plot(np.log10(moving_average(E, w) - true_E), c = "black")
    plt.ylabel(r"log$_{10}$(E - E$_{ref}$)")
  plt.xlabel("Iteration")
  plt.grid()
  if lim and not log:
    std = np.std(E[20_000:])
    last = np.mean(E[-10_000:])
    plt.ylim([last - 2 * std, last + 5 * std])
  if not log:
    plt.savefig("linear_learning_curve.pdf")
  else:
    plt.savefig("log_learning_curve.pdf")
import random

import numpy
import torch
from visdom import Visdom

class AverageMeter(object):
    """Keeps track of most recent, average, sum, and count of a metric.

    Examples
    --------
    losses = AverageMeter()
    losses.update(1, 5)
    print(losses.avg)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # Reset all value to 0.
        self.value = 0
        self.avg   = 0
        self.sum   = 0
        self.count = 0

    def update(self, value, n=1):
        """Update value, average, sum, and count.

        Parameters
        ----------
        n : int, optional (default = 5)
        value : double

        """
        self.value = value
        self.sum   += value * n
        self.count += n
        self.avg   = self.sum / self.count

class VisdomLinePlotter(object):
    """Plots lines in Visdom.

    Parameter
    ---------
    env_name: str, optional (default = 'main')
        Environment name of Visdom, you should not change it if don't know what's going on.

    Example
    -------
    import time

    plotter = VisdomLinePlotter()
    for x, y in zip(range(10), range(10)):
        plotter.plot("var_name", "split_name", "title_name", x, y)
        time.sleep(2)
    """
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X   = np.array([x,x]),
                                                 Y   = np.array([y,y]),
                                                 env = self.env,
                                                 opts= dict(legend = [split_name],
                                                           title   = title_name,
                                                           xlabel  = 'Epochs',
                                                           ylabel  = var_name))

        else:
            self.viz.line(X      = np.array([x]),
                          Y      = np.array([y]),
                          env    = self.env,
                          win    = self.plots[var_name],
                          name   = split_name,
                          update = 'append')

def pytorch_reproducer(device="cpu", seed=2019):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed : int, optional (default = 2019)
    	Radnom seed.
    device : str, optinal (default = "cpu")
    	Device type.

    Examples
    --------
    pytorch_reproducer(seed=2019, device=DEVICE).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_device():
    """Return device type.

    Returns
    -------
    DEVICE : torch.device

    Examples
    --------
    DEVICE = get_device()
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DEVICE

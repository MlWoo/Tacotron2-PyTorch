import matplotlib.pyplot as plt
import time
import sys
import numpy as np
from collections import namedtuple
from datetime import datetime
_format = '%Y-%m-%d-%H-%M-%S-%f'


def time_string():
    return datetime.now().strftime(_format)


class ValueWindow(object):
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []
        self.val = None

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]\

    @property
    def sum(self):
        return sum(self._values)\

    @property
    def count(self):
        return len(self._values)\

    @property
    def average(self):
        return self.sum / max(1, self.count)

    @property
    def get_dinwow_size(self):
        return self._window_size

    @property
    def avg(self):
        return self.average

    def reset(self):
        self._values = []

    def update(self, val):
        self.append(val)
        self.val = val


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _json_object_hook(d):
    return namedtuple('X', d.keys())(*d.values())


def display(string, variables):
    sys.stdout.write(f'\r{string}' % variables)


def num_params(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3f million' % parameters)


def time_since(started):
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60:
        h = int(m // 60)
        m = m % 60
        return f'{h}h {m}m {s}s'
    else:
        return f'{m}m {s}s'


def plot(array):
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis='x', colors='grey', labelsize=23)
    ax.tick_params(axis='y', colors='grey', labelsize=23)
    plt.plot(array)


def plot_spec(M):
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18, 4))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.show()


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()

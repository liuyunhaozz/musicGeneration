import time
import matplotlib.pyplot as plt
from IPython import display
import IPython

"""
可视化类，用于训练时显示训练结果
"""
class PeriodicPlotter:
  def __init__(self, sec, xlabel='', ylabel='', scale=None):

    self.xlabel = xlabel
    self.ylabel = ylabel
    self.sec = sec
    self.scale = scale

    self.tic = time.time()

  def plot(self, data):
    if time.time() - self.tic > self.sec:
        plt.cla()

        if self.scale is None:
            plt.plot(data)
        elif self.scale == 'semilogx':
            plt.semilogx(data)
        elif self.scale == 'semilogy':
            plt.semilogy(data)
        elif self.scale == 'loglog':
            plt.loglog(data)
        else:
            raise ValueError("unrecognized parameter scale {}".format(self.scale))

        plt.xlabel(self.xlabel); plt.ylabel(self.ylabel)
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.savefig('training_checkpoints/train.jpg')
        
        self.tic = time.time()
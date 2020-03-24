# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
    dpi = 150 if dpi == None else dpi
    plt.savefig(fname, dpi=dpi)
    
def plot_overlap(logger, ax, axins, names=None):
    names = logger.names if names == None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        ax.plot(x, np.asarray(numbers[name], np.float32), '-.' if 'Valid' in name else '-', linewidth=.5)
        axins.plot(x, np.asarray(numbers[name], np.float32), '-.' if 'Valid' in name else '-', linewidth=.5)
    return [logger.title + '(' + name + ')' for name in names]

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):   
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        fig, ax = plt.subplots()
        axins = zoomed_inset_axes(ax, 2.5, loc='center')
        #plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, ax, axins, names)
        x1, x2, y1, y2 = 80, 90, 75, 85 # specify the limits
        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        axins.set_yticks([])
        axins.set_xticks([])
        axins.tick_params(axis='x', colors='yellow')
        axins.tick_params(axis='y', colors='yellow')
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.8")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.legend(legend_text, loc='lower right', borderaxespad=0.)
        ax.axis([-5, 95, 0, 90])
        ax.grid(axis='y', linestyle='--')
                    
if __name__ == '__main__':
    # # Example
    # logger = Logger('test.txt')
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])

    # length = 100
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()

    # Example: logger monitor
    paths = {
    'ResNet-50':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet50/log.txt',
    'ResNet-50(DSL)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet50-extraction/log.txt',
    'ResNet-50(DHM)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/exp_integration/resnet50-integration-customized-without-scale-augmentation/log.txt',
    #'ResNet-101':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet101/log.txt',
    #'ResNet-101(DSL)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet101-extraction/log.txt',
    #'ResNet-101(DHM)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/exp_integration/resnet101-integration/log.txt',
    #'ResNet-152':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet152/log.txt',
    #'ResNet-152(DSL)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/resnet152-extraction/log.txt',
    #'ResNet-152(DHM)':'/home/rll/lid/auxiliary/classification/checkpoints/imagenet/exp_integration/resnet152-integration-customized/log.txt',
    }

    field = ['Train Acc.', 'Valid Acc.']

    monitor = LoggerMonitor(paths)
    monitor.plot(names=field)
    savefig('test.eps')

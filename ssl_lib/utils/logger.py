# A simple torch style logger
# (C) Wei YANG 2017
from __future__ import absolute_import
import os
import sys
import numpy as np
import json
import scipy.signal
from matplotlib import pyplot as plt

__all__ = ['Logger', 'LoggerMonitor', 'init_trial_path']

    

def init_trial_path(args):
	"""Initialize the path for a hyperparameter setting
	"""
	args.result_dir = os.path.join(args.out_dir, args.task_name)
	os.makedirs(args.result_dir, exist_ok=True)
	trial_id = 0
	path_exists = True
	while path_exists:
		trial_id += 1
		path_to_results = args.result_dir + '/{:d}'.format(trial_id)
		path_exists = os.path.exists(path_to_results)
	args.save_path = path_to_results
	os.makedirs(args.save_path, exist_ok=True)
	with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
		json.dump(args.__dict__, f)
	return args

class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, folder, title=None, resume=False):
        fpath = os.path.join(folder, 'log.txt')
        self.folder = folder
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

    def write(self, info):
        print(info)
        self.file.write(info)
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers, verbose=True):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
        if verbose:
            self.print()
        self.loss_plot()
        self.acc_plot()

    def concat(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()
        log_info = self.print_info()
        return log_info

    def print_info(self):
        log_str = ""
        for name, num in self.numbers.items():
            log_str += f"{name}: {num[-1]}, "
        return log_str

    def print(self):
        log_str = ""
        for name, num in self.numbers.items():
            log_str += "{}: {:.6f}, ".format(name, num[-1])
        print(log_str)

    def close(self):
        if self.file is not None:
            self.file.close()

    def loss_plot(self):
        iters = range(len(self.numbers['Train Loss']))

        plt.figure()
        plt.plot(iters, self.numbers['Train Loss'], 'red', linewidth=2, label='train loss')
        try:
            if len(self.numbers['Train Loss']) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters, scipy.signal.savgol_filter(self.numbers['Train Loss'], num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.folder, "train_loss.png"))
        plt.close()

    def acc_plot(self):
        iters = range(len(self.numbers['Test Acc.']))

        plt.figure()
        plt.plot(iters, self.numbers['Labeled Acc'], 'red', linewidth=2, label='Labeled Acc')
        plt.plot(iters, self.numbers['Unlabeled Acc'], 'coral', linewidth=2, label='Unlabeled Acc')
        plt.plot(iters, self.numbers['Test Acc.'], 'blue', linewidth=2, label='Test Acc')
        try:
            if len(self.numbers['Test Acc.']) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.numbers['Labeled Acc'], num, 3), 'green', linestyle='--', linewidth=2,label='smooth Labeled Acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.numbers['Unlabeled Acc'], num, 3), '#8B4513', linestyle='--', linewidth=2,label='smooth Unlabeled Acc')
            plt.plot(iters, scipy.signal.savgol_filter(self.numbers['Test Acc.'], num, 3), 'yellow', linestyle='--',linewidth=2, label='smooth Test Acc')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Acc')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.folder, "Acc.png"))
        plt.close()

class LoggerMonitor(object):
    '''Load and visualize multiple logs.'''
    def __init__ (self, paths):
        '''paths is a distionary with {name:filepath} pair'''
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

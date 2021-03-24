import time
from os.path import join as pjoin, exists as pexists
import json


class Recorder(object):
    def __init__(self, path):
        self.path = path
        self.file_path = pjoin(self.path, 'recorder.json')

        self.loss_history, self.err_history = [], []
        self.best_err = float('inf')
        self.best_step_err = 0
        self.step = 0
        self.lr_decay_step = -1
        self.run_time = 0.

        if pexists(self.file_path):
            self.load_record()

    def save_record(self):
        with open(self.file_path, 'w') as op:
            json.dump({
                'loss_history': self.loss_history,
                'err_history': self.err_history,
                'best_err': self.best_err,
                'best_step_err': self.best_step_err,
                'step': self.step,
                'lr_decay_step': self.lr_decay_step,
                'run_time': self.run_time,
            }, op)

    def load_record(self):
        with open(self.file_path) as fp:
            record = json.load(fp)

        self.loss_history, self.err_history = \
            record['loss_history'], record['err_history']
        self.best_err = record['best_err']
        self.best_step_err = record['best_step_err']
        self.step = record['step']
        if 'lr_decay_step' in record:
            self.lr_decay_step = record['lr_decay_step']
        if 'run_time' in record:
            self.run_time = record['run_time']

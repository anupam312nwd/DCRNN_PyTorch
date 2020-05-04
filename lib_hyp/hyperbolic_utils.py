#!/usr/bin/env python

from collections import defaultdict
import os
import pickle
import json
import torch.nn as nn
import torch as th
import torch.optim as optim
import numpy as np
import random
from optimizer.ramsgrad import RiemannianAMSGrad
from optimizer.rsgd import RiemannianSGD
import math
import subprocess

class NoneScheduler:
    def step(self):
        pass

def get_optimizer(args, params):
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimmizer == 'amsgrad':
        optimizer = optim.Adam(params, lr=args.lr, amsgrad=True,
                               weight_decay=args.weight_decay)

def get_hyperbolic_optimizer(args, params):
    if args.hyper_optimizer == 'rsgd':
        optimizer = RiemannianSGD(args, params, lr=args.lr_hyperbolic)
    elif args.hyper_optimizer == 'rmsgrad':
        optimizer = RiemannianAMSGrad(args, params, lr=args.lr_hyperbolic)
    else:
        print("unsupported hyper optimizer")
        exit(1)
    return optimizer

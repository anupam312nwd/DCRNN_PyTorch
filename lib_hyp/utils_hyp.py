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
	elif args.optimizer == 'amsgrad':
		optimizer = optim.Adam(params, lr=args.lr, amsgrad=True, weight_decay=args.weight_decay)
	return optimizer

def get_hyperbolic_optimizer(args, params):
    if args.hyper_optimizer == 'rsgd':
	    optimizer = RiemannianSGD(args, params, lr=args.lr_hyperbolic)
    elif args.hyper_optimizer == 'ramsgrad':
        optimizer = RiemannianAMSGrad(args, params, lr=args.lr_hyperbolic)
    else:
        print("unsupported hyper optimizer")
        exit(1)
    return optimizer


def pad_sequence(data_list, maxlen, value=0):
	return [row + [value] * (maxlen - len(row)) for row in data_list]

def normalize_weight(adj_mat, weight):
	degree = [1 / math.sqrt(sum(np.abs(w))) for w in weight]
	for dst in range(len(adj_mat)):
		for src_idx in range(len(adj_mat[dst])):
			src = adj_mat[dst][src_idx]
			weight[dst][src_idx] = degree[dst] * weight[dst][src_idx] * degree[src]

def save_model_weights(args, model, path):
	"""
	save model weights out to file
	"""
	if args.distributed_rank == 0:
		make_dir(path)
		th.save(model.state_dict(), os.path.join(path, args.name))

def load_model_weights(model, path):
	"""
	load saved weights
	"""
	model.load_state_dict(th.load(path))

def th_atanh(x, EPS):
	values = th.min(x, th.Tensor([1.0 - EPS]).cuda())
	return 0.5 * (th.log(1 + values + EPS) - th.log(1 - values + EPS))

def th_norm(x, dim=1):
	"""
	Args
		x: [batch size, dim]
	Output:
		[batch size, 1]
	"""
	return th.norm(x, 2, dim, keepdim=True)

def th_dot(x, y, keepdim=True):
	return th.sum(x * y, dim=1, keepdim=keepdim)

def clip_by_norm(x, clip_norm):
	return th.renorm(x, 2, 0, clip_norm)

def get_params(params_list, vars_list):
	"""
	Add parameters in vars_list to param_list
	"""
	for i in vars_list:
		if issubclass(i.__class__, nn.Module):
			params_list.extend(list(i.parameters()))
		elif issubclass(i.__class__, nn.Parameter):
			params_list.append(i)
		else:
			print("Encounter unknown objects")
			exit(1)

def categorize_params(args):
	"""
	Categorize parameters into hyperbolic ones and euclidean ones
	"""
	hyperbolic_params, euclidean_params = [], []
	get_params(euclidean_params, args.eucl_vars)
	get_params(hyperbolic_params, args.hyp_vars)
	return hyperbolic_params, euclidean_params

def get_activation(args):
	if args.activation == 'leaky_relu':
		return nn.LeakyReLU(args.leaky_relu)
	elif args.activation == 'rrelu':
		return nn.RReLU()
	elif args.activation == 'relu':
		return nn.ReLU()
	elif args.activation == 'elu':
		return nn.ELU()
	elif args.activation == 'prelu':
		return nn.PReLU()
	elif args.activation == 'selu':
		return nn.SELU()

def set_up_optimizer_scheduler(hyperbolic, args, model):
	if hyperbolic:
		hyperbolic_params, euclidean_params = categorize_params(args)
		assert(len(list(model.parameters())) == len(hyperbolic_params) + len(euclidean_params))
		optimizer = get_optimizer(args, euclidean_params)
		lr_scheduler = get_lr_scheduler(args, optimizer)
		if len(hyperbolic_params) > 0:
			hyperbolic_optimizer = get_hyperbolic_optimizer(args, hyperbolic_params)
			hyperbolic_lr_scheduler = get_lr_scheduler(args, hyperbolic_optimizer)
		else:
			hyperbolic_optimizer, hyperbolic_lr_scheduler = None, None
		return optimizer, lr_scheduler, hyperbolic_optimizer, hyperbolic_lr_scheduler
	else:
		optimizer = get_optimizer(args, model.parameters())
		lr_scheduler = get_lr_scheduler(args, optimizer)
		return optimizer, lr_scheduler, None, None

# reimplement clamp functions to avoid killing gradient during backpropagation
def clamp_max(x, max_value):
	t = th.clamp(max_value - x.detach(), max=0)
	return x + t

def clamp_min(x, min_value):
	t = th.clamp(min_value - x.detach(), min=0)
	return x + t

def one_hot_vec(length, pos):
	vec = [0] * length
	vec[pos] = 1
	return vec

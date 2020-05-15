import torch as th
import torch.nn as nn

import numpy as np
from torch.autograd import Function, Variable

from .hyperbolic_utils import *


class PoincareManifold:

    def __init__(self, args=0, logger=0, EPS=1e-5, PROJ_EPS=1e-5):
        self.args = args
        self.logger = logger
        self.EPS = EPS
        self.PROJ_EPS = PROJ_EPS
        self.tanh = nn.Tanh()

    def normalize(self, x):
        return clip_by_norm(x, (1. - self.PROJ_EPS))

    def init_embed(self, embed, irange=1e-2):
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def mob_add(self, u, v):
        """
        Add two vectors in hyperbolic space
        """
        v = v + self.EPS
        th_dot_u_v = 2. * th_dot(u, v)
        th_norm_u_sq = th_dot(u, u)
        th_norm_v_sq = th_dot(v, v)
        denominator = 1. + th_dot_u_v + th_norm_v_sq * th_norm_u_sq
        result = (1. + th_dot_u_v + th_norm_v_sq) / (denominator + self.EPS) * u + \
                 (1. - th_norm_u_sq) / (denominator + self.EPS) * v
        return self.normalize(result)

    def lambda_x(self, x):
        """
        A conformal factor
        """
        return 2. / (1 - th_dot(x, x))

    def log_map_zero(self, y):
        diff = y + self.EPS
        norm_diff = th_norm(diff)
        return 1. / th_atanh(norm_diff, self.EPS) / norm_diff * diff

    def log_map_x(self, x, y):
        diff = self.mob_add(-x, y) + self.EPS
        norm_diff = th_norm(diff)
        lam = self.lambda_x(x)
        return (( 2. / lam) * th_atanh(norm_diff, self.EPS) / norm_diff) * diff

    def metric_tensor(self, x, u, v):
        """
        The metric tensor in hyperbolic space.
        In-place operations for saving memory. (do not use this function in forward calls)
        """
        u_dot_v = th_dot(u, v)
        lambda_x = self.lambda_x(x)
        lambda_x *= lambda_x
        lambda_x *= u_dot_v
        return lambda_x

    def exp_map_zero(self, v):
        """
        Exp map from tangent space of zero to hyperbolic space
        Args:
            v: [batch_size, *] in tangent space
        """
        v = v + self.EPS
        norm_v = th_norm(v) # [batch_size, 1]
        result = self.tanh(norm_v) / (norm_v) * v
        return self.normalize(result)

    def exp_map_x(self, x, v):
        """
        Exp map from tangent space of x to hyperbolic space
        """
        v = v + self.EPS # Perturbe v to avoid dealing with v = 0
        norm_v = th_norm(v)
        second_term = (self.tanh(self.lambda_x(x) * norm_v / 2) / norm_v) * v
        return self.normalize(self.mob_add(x, second_term))

    def gyr(self, u, v, w):
        u_norm = th_dot(u, u)
        v_norm = th_dot(v, v)
        u_dot_w = th_dot(u, w)
        v_dot_w = th_dot(v, w)
        u_dot_v = th_dot(u, v)
        A = - u_dot_w * v_norm + v_dot_w + 2 * u_dot_v * v_dot_w
        B = - v_dot_w * u_norm - u_dot_w
        D = 1 + 2 * u_dot_v + u_norm * v_norm
        return w + 2 * (A * u + B * v) / (D + self.EPS)

    def parallel_transport(self, src, dst, v):
        return self.lambda_x(src) / th.clamp(self.lambda_x(dst), min=self.EPS) * self.gyr(dst, -src, v)

    def rgrad(self, p, d_p):
        """
        Function to compute Riemannian gradient from the
        Euclidean gradient in the Poincare ball.
        Args:
            p (Tensor): Current point in the ball
            d_p (Tensor): Euclidean gradient at p
        """
        p_sqnorm = th.sum(p.data ** 2, dim=-1, keepdim=True)
        d_p = d_p * ((1 - p_sqnorm) ** 2 / 4.0).expand_as(d_p)
        return d_p

    def matrix_vec(self, M, v):
        """matrix operation on a vector in Poincare Ball"""
        v_column_norm = th.sqrt(th.sum(v*v, axis=0))
        Mv = th.matmul(M, v)/v_column_norm
        Mv_norm = th.sqrt(th.sum(Mv*Mv, axis=0))
        M_otimes_v = self.tanh((Mv_norm/v_column_norm)*th_atanh(v_column_norm))*(Mv/v_column_norm)
        return M_otimes_v

    def matrix_vec_stack(self, M, X):
        """matrix operation on a stack of hyperbolic vectors"""
        MX = th.matmul(M, X)
        MX_norm = th.norm(MX, dim=0, keepdim=True)
        X_norm = th.norm(X, dim=0, keepdim=True)
        abs_max = abs(max([X_norm.max(), X_norm.min()], key=abs)) + 1e-4
        MX_over_X = MX_norm/X_norm
        M_otimes_X = th.mul(self.tanh(th.mul(MX_over_X, th_atanh(X_norm, 1e-4))), MX/MX_norm)
        return M_otimes_X

    def diag_matrix_vec_stack(self, M, X):
        """diag_vec matrix operation on hyperbolic vector
        M:= stack of vec of which diag(vec) to be considered
        X:= stack of hyperbolic vectors"""
        MX = th.mul(M, X)
        MX_norm = th.norm(MX, dim=1, keepdim=True)
        X_norm = th.norm(X, dim=1, keepdim=True)
        MX_over_X = MX_norm/X_norm
        abs_max = abs(max([X_norm.max(), X_norm.min()], key=abs)) + 1e-4
        diag_M_otimes_X = th.mul(self.tanh(th.mul(MX_over_X, th_atanh(X_norm, 1e-4))), MX/MX_norm)
        return diag_M_otimes_X


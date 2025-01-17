# -*- coding:utf-8 -*-
#The MIT License (MIT)
#Copyright (c) 2021 Huawei Technologies Co., Ltd.

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import math
import torch.optim
from torch.nn import Parameter
# from hyperbolic.euclidean import Euclidean
# from hyperbolic.poincare import ManifoldParameter


class ManifoldParameter(Parameter):

    def __new__(cls, data, requires_grad, manifold, c):
        return Parameter.__new__(cls, data, requires_grad)

    def __init__(self, data, requires_grad, manifold, c):
        self.c = c
        self.manifold = manifold

    def __repr__(self):
        return '{} Parameter containing:\n'.format(self.manifold.name) + super(Parameter, self).__repr__()


class Euclidean(object):

    def __init__(self):
        super(Euclidean,self).__init__()
        self.name = 'Euclidean'

    def normalize(self, p):
        dim = p.size(-1)
        p.view(-1, dim).renorm_(2, 0, 1.)
        return p

    def sqdist(self, p1, p2, c):
        """
        compute Euclidean distance
        :param p1:
        :param p2:
        :param c: curvalture
        :return: distance value
        """
        return (p1 - p2).pow(2).sum(dim=-1)

    def egrad2rgrad(self, p, dp, c):
        return dp

    def proj(self, p, c):
        return p

    def proj_tan(self, u, p, c):
        return u

    def proj_tan0(self, u, c):
        return u

    def expmap(self, u, p, c):
        return p + u

    def logmap(self, p1, p2, c):
        return p2 - p1

    def expmap0(self, u, c):
        return u

    def logmap0(self, p, c):
        return p

    def mobius_add(self, x, y, c, dim=-1):
        """
        general add in Euclidean Space
        :param x:
        :param y:
        :param c:curvalture
        :param dim:
        :return:
        """
        return x + y

    def mobius_matvec(self, m, x, c):
        """
        general matrix multiplication
        :param m:
        :param x:
        :param c:curvalture
        :return:
        """
        mx = x @ m.transpose(-1, -2)
        return mx

    def init_weights(self, w, c, irange=1e-5):
        """
        init weight value using uniform distribution
        :param w:
        :param c:
        :param irange: default 1e-5
        :return:
        """
        w.data.uniform_(-irange, irange)
        return w

    def inner(self, p, c, u, v=None, keepdim=False):
        if v is None:
            v = u
        return (u * v).sum(dim=-1, keepdim=keepdim)

    def ptransp(self, x, y, v, c):
        return v

    def ptransp0(self, x, v, c):
        return x + v

class RiemannianAdam(torch.optim.Adam):
    """
    RiemannianAdam optimizer
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(RiemannianAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.euclidean_manifold = Euclidean()

    def reset_value(self, target, source):
        return target.copy_(source) if target.stride() != source.stride() else target.set_(source)

    def step(self, closure=None):
        loss = closure() if closure else None
        with torch.no_grad():
            for p_group in self.param_groups:
                if "step" not in p_group:
                    p_group["step"] = 0
                betas = p_group["betas"]
                weight_decay = p_group["weight_decay"]
                eps = p_group["eps"]
                learning_rate = p_group["lr"]
                amsgrad = p_group["amsgrad"]

                for point in p_group["params"]:
                    if isinstance(point, (ManifoldParameter)):
                        manifold = point.manifold
                        c = point.c
                    else:
                        manifold = self.euclidean_manifold
                        c = None
                    grad = point.grad
                    if grad is None:
                        continue

                    state = self.state[point]
                    if len(state) == 0:
                        if amsgrad:
                            state["max_exp_avg_sq"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)
                        state["exp_avg_sq"] = torch.zeros(point.size(), dtype=point.dtype, device=point.device)

                    grad.add_(weight_decay * point)
                    grad = manifold.egrad2rgrad(point, grad, c)
                    exp_avg = state["exp_avg"]
                    exp_avg.mul_(betas[0]).add_((1 - betas[0]) * grad)
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(betas[1]).add_(
                        (1 - betas[1]) * manifold.inner(point, c, grad, keepdim=True)
                    )
                    if amsgrad:
                        max_exp_avg_sq = torch.max(state["max_exp_avg_sq"], exp_avg_sq)
                        denom = torch.add(max_exp_avg_sq.sqrt(), eps)
                    else:
                        denom = torch.add(exp_avg_sq.sqrt(), eps)
                    p_group["step"] += 1
                    bias_cor1 = 1 - math.pow(betas[0], p_group["step"])
                    bias_cor2 = 1 - math.pow(betas[1], p_group["step"])
                    step_size = (
                            learning_rate * bias_cor2 ** 0.5 / bias_cor1
                    )
                    direction = exp_avg / denom
                    new_point = manifold.proj(manifold.expmap(-step_size * direction, point, c), c)
                    exp_avg_new = manifold.ptransp(point, new_point, exp_avg, c)
                    self.reset_value(point, new_point)
                    exp_avg.set_(exp_avg_new)
                    p_group["step"] += 1

        return loss
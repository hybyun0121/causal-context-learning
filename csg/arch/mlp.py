#!/usr/bin/env python3.6
'''Multi-Layer Perceptron Architecture.

For causal discriminative model and the corresponding generative model.
'''
import sys, os
import json
import torch as tc
import torch.nn.functional as F
import torch.nn as nn
from numbers import Number
sys.path.append('..')
from distr import tensorify, is_same_tensor, wrap4_multi_batchdims

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def init_linear(nnseq, wmean, wstd, bval):
    for mod in nnseq:
        if type(mod) is nn.Linear:
            mod.weight.data.normal_(wmean, wstd)
            mod.bias.data.fill_(bval)

def mlp_constructor(dims, actv = "Sigmoid", lastactv = True): # `Sequential()`, or `Sequential(*[])`, is the identity map for any shape!
    if type(actv) is str: actv = getattr(nn, actv)
    if len(dims) <= 1: return nn.Sequential()
    else: return nn.Sequential(*(
        sum([[nn.Linear(dims[i], dims[i+1]), actv()] for i in range(len(dims)-2)], []) + \
        [nn.Linear(dims[-2], dims[-1])] + ([actv()] if lastactv else [])
    ))

class MLPBase(nn.Module):
    def save(self, path): tc.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(tc.load(path))
        self.eval()
    def load_or_save(self, filename):
        dirname = "init_models_mlp/"
        os.makedirs(dirname, exist_ok=True)
        path = dirname + filename
        if os.path.exists(path): self.load(path)
        else: self.save(path)

class MLP(MLPBase):
    def __init__(self, dims, actv = "Sigmoid"):
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLP, self).__init__()
        self.f_x2y = mlp_constructor(dims, actv, lastactv = False)
    def forward(self, x): return self.f_x2y(x).squeeze(-1)

class MLPcsy1x(MLPBase):
    def __init__(self, dim_x, dims_postx2pres, dim_s, dim_paras, dims_posts2c, dims_postc2prey, dim_y, dim_t, actv = "Sigmoid",
            std_s1xte_val: float=-1., std_c1sxt_val: float=-1., after_actv: bool=True, ind_cs: bool=False,
            dims_vars2prev_xte=None, dims_vars2prev_xt=None, dims_prevs2c=None, dims_prevs2s=None): # if <= 0, then learn the std.
        """
        x, t, e == g_s()==> prevs --k()->   s   -\
                                                 |=w()=> c ==> y
           x, t == g_c()==> prevc --h()-> parav -/

        """
        super(MLPcsy1x, self).__init__()
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_s, self.dim_y = dim_x, dim_s, dim_y
        self.dim_t = dim_t
        dim_pres, dim_c = dims_postx2pres[-1], dims_posts2c[-1]
        self.dim_pres, self.dim_c = dim_pres, dim_c
        self.shape_x, self.shape_s, self.shape_c = (dim_x,), (dim_s,), (dim_c,)
        self.shape_t, self.shape_e = (dim_t,), (dim_x,)
        self.dims_postx2pres, self.dim_paras, self.dims_posts2c, self.dims_postc2prey, self.actv \
                = dims_postx2pres, dim_paras, dims_posts2c, dims_postc2prey, actv
        
        self.ind_cs = ind_cs

        # g(x,t,e)
        self.f_xte2prev = mlp_constructor([dim_x*3] + dims_postx2pres, actv)
        self.f_xe2prev = mlp_constructor([dim_x*2] + dims_postx2pres, actv)
        self.f_xt2prev = mlp_constructor([dim_x+dim_t] + dims_postx2pres, actv)
        
        # k(prev)
        self.f_prev2s = nn.Linear(dim_pres, dim_s)
        # h(prev)
        self.f_prev2parav = nn.Linear(dim_pres, dim_paras)
        # w(s, parav)
        if not ind_cs:
            self.f_sparas2c = mlp_constructor([dim_s + dim_paras] + dims_posts2c, actv, lastactv = False)
        else:
            self.f_paras2c = mlp_constructor([dim_paras] + dims_posts2c, actv, lastactv = False)

        # q(y|c)
        self.f_c2y = mlp_constructor([dim_c] + dims_postc2prey + [dim_y], actv, lastactv = False)
        
        # if after_actv:
        #     # k(prev)
        #     self.f_prev2s = nn.Linear(dim_pres, dim_s)
        #     # h(prev)
        #     self.f_prev2parav = nn.Linear(dim_pres, dim_paras)
        #     # w(s, parav)
        #     if not ind_cs:
        #         self.f_sparas2c = mlp_constructor([dim_s + dim_paras] + dims_posts2c, actv, lastactv = False)
        #     else:
        #         self.f_paras2c = mlp_constructor([dim_paras] + dims_posts2c, actv, lastactv = False)

        #     # q(y|c)
        #     self.f_c2y = mlp_constructor([dim_c] + dims_postc2prey + [dim_y], actv, lastactv = False)
        # else:
        #     self.f_prev2s = nn.Linear(dim_pres, dim_s)
        #     self.f_prev2parav = nn.Linear(dim_pres, dim_paras)
        #     self.f_sparas2c = nn.Sequential( actv(), mlp_constructor([dim_s + dim_paras] + dims_posts2c, actv, lastactv = False) )
        #     self.f_c2y = nn.Sequential( actv(), mlp_constructor([dim_c] + dims_postc2prey + [dim_y], actv, lastactv = False) )

        self.std_s1xte_val = std_s1xte_val
        self.std_s1xe_val = std_s1xte_val

        self.std_c1sxt_val = std_c1sxt_val
        self.std_c1xt_val = std_c1sxt_val
        self.learn_std_s1xte = std_s1xte_val <= 0 if type(std_s1xte_val) is float else (std_s1xte_val <= 0).any()
        self.learn_std_c1sxt = std_c1sxt_val <= 0 if type(std_c1sxt_val) is float else (std_c1sxt_val <= 0).any()

        self._prev_cache_s = None
        self._prev_cache_c = None
        self._parav_cache = None
        self._x_cache_prev_s = None
        self._x_cache_prev_c = None
        self._t_cache_prev_s = None
        self._t_cache_prev_c = None
        self._e_cache_prev_s = None
        self._e_cache_prev_c = None
        self._s_cache = None
        self._x_cache_s = None
        self._t_cache_s = None
        self._e_cache_s = None
        self._t_cache_c = None
        self._x_cache_parav = None

        ## std models
        if self.learn_std_s1xte:
            self.nn_std_s = nn.Sequential(
                    mlp_constructor(
                        [dim_pres, dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = self.nn_std_s

        if self.learn_std_c1sxt:
            if not ind_cs:
                _input_dim = dim_s + dim_paras
            else:
                _input_dim = dim_paras
                
            self.nn_std_c = nn.Sequential(
                    nn.BatchNorm1d(_input_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp_constructor(
                        [_input_dim] + dims_posts2c,
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_c, 0., 1e-2, 0.)
            self.f_std_c = wrap4_multi_batchdims(self.nn_std_c, ndim_vars=1)
        
        self.nn_std_y1c = nn.Sequential(
                                mlp_constructor(
                                [dim_c] + dims_postc2prey + [dim_y],
                                nn.ReLU, lastactv = False),
                                nn.Softplus())
        init_linear(self.nn_std_y1c, 0., 1e-2, 0.)
        self.f_std_y1c = self.nn_std_y1c

    def _get_prevs(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(t, self._t_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._t_cache_prev_s = t
            self._e_cache_prev_s = e
            _input = tc.cat([x, t, e], dim=-1)
            self._prev_cache_s = self.f_xte2prev(_input) # g(x,t,e)
        return self._prev_cache_s
    
    def _get_prevs_ind(self, x,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._e_cache_prev_s = e
            _input = tc.cat([x, e], dim=-1)
            self._prev_cache_s = self.f_xe2prev(_input) # g(x,t,e)
        return self._prev_cache_s
    
    def _get_prevc(self, x,t):
        if (not is_same_tensor(x, self._x_cache_prev_c))\
        and (not is_same_tensor(t, self._t_cache_prev_c)):
            self._x_cache_prev_c = x
            self._t_cache_prev_c = t
            _input = tc.cat([x, t], dim=-1)
            self._prev_cache_c = self.f_xt2prev(_input) # g(x,t,e)
        return self._prev_cache_c
    
    def s1xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(t, self._t_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._t_cache_s = t
            self._e_cache_s = e

            self._s_cache = self.f_prev2s(self._get_prevs(x,t,e))
        return self._s_cache

    def std_s1xte(self, x,t,e):
        if self.learn_std_s1xte:
            return self.f_std_s(self._get_prevs(x,t,e))
        else:
            return tensorify(x.device, self.std_s1xte_val)[0].expand(x.shape[:-1]+(self.dim_s,))

    def s1xe(self, x,e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._e_cache_s = e
            self._s_cache = self.f_prev2s(self._get_prevs_ind(x,e))
        return self._s_cache
    
    def std_s1xe(self, x,e):
        if self.learn_std_s1xte:
            return self.f_std_s(self._get_prevs_ind(x,e))
        else:
            return tensorify(x.device, self.std_s1xte_val)[0].expand(x.shape[:-1]+(self.dim_s,))

    def _get_parav(self, x,t):
        # h(prev)
        if (not is_same_tensor(x, self._x_cache_parav))\
        and (not is_same_tensor(t, self._t_cache_c)):
            self._x_cache_parav = x
            self._t_cache_c = t
            self._parav_cache = self.f_prev2parav(self._get_prevc(x,t))
        return self._parav_cache

    def c1sxt(self, s, x, t): # q(c|s,x,t)
        parav = self._get_parav(x,t) # parav | g(x,t,e)
        # w(s, parav)
        return self.f_sparas2c(tc.cat([s, parav], dim=-1))

    def std_c1sxt(self, s, x, t):
        if self.learn_std_c1sxt:
            parav = self._get_parav(x,t)
            return self.f_std_c(tc.cat([s, parav], dim=-1))
        else:
            return tensorify(x.device, self.std_c1sxt_val)[0].expand(x.shape[:-1]+t.shape[:-1]+(self.dim_c,))

    def c1xt(self, x,t):
        '''
        q(c|x,t,e) = q(c|s,x,t,e)q(s|x,t,e)
        '''
        parav = self._get_parav(x,t) # parav | g(x,t,e)
        return self.f_paras2c(parav)
    
    def c1x(self, x,t,e):
        '''
        q(c|x,t,e) = q(c|s,x,t,e)q(s|x,t,e)
        '''
        # if self.ind_cs:
        return self.c1xt(x,t)
        # else:
        #     return self.c1sxt(self.s1xte(x,t,e), x,t)

    def std_c1xt(self, x,t):
        if self.learn_std_c1sxt:
            parav = self._get_parav(x,t)
            return self.f_std_c(parav)
        else:
            return tensorify(x.device, self.std_c1xt_val)[0].expand(x.shape[:-1]+t.shape[:-1]+(self.dim_c,))
    
    def y1c(self, c):
        '''
        q(y|c)
        '''
        return self.f_c2y(c)
    
    def std_y1c(self, c):
        return self.f_std_y1c(c)
    
    def ys1x(self, x):
        c = self.c1x(x) # q(c|x,t,e)
        return self.y1c(c), c

    def forward(self, x,t,e):
        '''
        q(y|c) = q(y|c)q(c,s|x,t,e)
        '''
        return self.y1c(self.c1x(x,t,e))

class MLPcsy1xte(MLPBase):
    def __init__(self, dim_x, dims_postx2pres, dim_s, dim_paras, dims_posts2c, dims_postc2prey, dim_y, dim_t, actv = "Sigmoid",
            std_s_val: float=-1., std_c_val: float=-1., after_actv: bool=True, ind_cs: bool=False,
            dims_vars2prev_xte=None, dims_vars2prev_xe=None, dims_prevs2c=None, dims_prevs2s=None, dims_t2c=None): # if <= 0, then learn the std.
        """
        x,t,e = prev_xte =>  c     ===============\
                                                    ==> s
                            x,e    === prev_xe ===/
        """
        super(MLPcsy1xte, self).__init__()
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_s, self.dim_y = dim_x, dim_s, dim_y
        self.dim_t = dim_t
        dim_c = dims_posts2c[-1]
        self.dim_pres4c = dims_vars2prev_xte[-1]
        self.dim_pres4s = dims_vars2prev_xe[-1]
        self.dim_pres, self.dim_c = self.dim_pres4s, dim_c
        self.shape_x, self.shape_s, self.shape_c = (dim_x,), (dim_s,), (dim_c,)
        self.shape_t, self.shape_e = (dim_t,), (dim_x,)
        self.dims_postx2pres, self.dim_paras, self.dims_posts2c, self.dims_postc2prey, self.actv \
                = dims_postx2pres, dim_paras, dims_posts2c, dims_postc2prey, actv
        
        self.dims_vars2prev_xte = dims_vars2prev_xte
        self.dims_vars2prev_xe = dims_vars2prev_xe
        self.dims_t2c = dims_t2c
        self.ind_cs = ind_cs

        # g(x,t,e)
        self.f_xte2prev = mlp_constructor([dim_x+dim_t+dim_x] + dims_vars2prev_xte if dims_vars2prev_xte is not None else dims_postx2pres, actv, lastactv = False)
        self.f_xe2prev = mlp_constructor([dim_x+dim_x] + dims_vars2prev_xe if dims_vars2prev_xe is not None else dims_postx2pres, actv, lastactv = False)
        
        self.f_prev2c = mlp_constructor([self.dim_pres4c] + dims_prevs2s + [dim_c], actv, lastactv=False)
        self.f_prevs2s = mlp_constructor([self.dim_pres4s + dim_c] + dims_prevs2c + [dim_s], actv, lastactv=False)
  
        # q(y|c)
        self.f_c2y = mlp_constructor([dim_c] + dims_postc2prey + [dim_y], actv, lastactv = False)
        
        self.std_s_val = std_s_val
        self.std_s1xe_val = std_s_val

        self.std_c_val = std_c_val
        self.std_c1xt_val = std_c_val
        self.learn_std_s = std_s_val <= 0 if type(std_s_val) is float else (std_s_val <= 0).any()
        self.learn_std_c = std_c_val <= 0 if type(std_c_val) is float else (std_c_val <= 0).any()

        self._prev_cache_s = None
        self._prev_cache_c = None
        self._parav_cache = None
        self._x_cache_prev_s = None
        self._x_cache_prev_c = None
        self._t_cache_prev_s = None
        self._t_cache_prev_c = None
        self._e_cache_prev_s = None
        self._e_cache_prev_c = None
        self._s_cache = None
        self._x_cache_s = None
        self._t_cache_s = None
        self._e_cache_s = None
        self._t_cache_c = None
        self._x_cache_parav = None

        ## std models
        if self.learn_std_c:
            self.nn_std_c = nn.Sequential(
                    mlp_constructor(
                        [self.dim_pres4s] + dims_prevs2c + [dim_c],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_c, 0., 1e-2, 0.)
            self.f_std_c = self.nn_std_c

        if self.learn_std_s:
            _input_dim = dim_c + self.dim_pres4s

            self.nn_std_s = nn.Sequential(
                    nn.BatchNorm1d(_input_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp_constructor(
                        [_input_dim] + dims_prevs2s + [dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = wrap4_multi_batchdims(self.nn_std_s, ndim_vars=1)
        
        self.nn_std_y1c = nn.Sequential(
                                mlp_constructor(
                                [dim_c] + dims_postc2prey + [dim_y],
                                nn.ReLU, lastactv = False),
                                nn.Softplus())
        init_linear(self.nn_std_y1c, 0., 1e-2, 0.)
        self.f_std_y1c = self.nn_std_y1c

    def _get_prevs_xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_prev_c))\
        and (not is_same_tensor(t, self._t_cache_prev_c))\
        and (not is_same_tensor(e, self._e_cache_prev_c)):
            self._x_cache_prev_c = x
            self._t_cache_prev_c = t
            self._e_cache_prev_c = e
            _input = tc.cat([x, t, e], dim=-1)
            self._prev_cache_xte = self.f_xte2prev(_input) # g(x,t,e)
        return self._prev_cache_xte
    
    def _get_prevs_xe(self, x,e):
        # h(prev)
        if (not is_same_tensor(x, self._x_cache_parav))\
        and (not is_same_tensor(e, self._t_cache_c)):
            self._x_cache_parav = x
            self._e_cache_s = e
            self._parav_cache_xt = self.f_xt2prev(tc.cat([x, e], dim=-1))
        return self._parav_cache_xt

    def std_s1xte(self, x,t,e):
        if self.learn_std_c:
            return self.f_std_c(self._get_prevs_xte(x,t,e))
        else:
            return tensorify(x.device, self.std_c_val)[0].expand(x.shape[:-1]+(self.dim_c,))
    
    def std_s1cxe(self, c, x, e):
        if self.learn_std_s:
            prev_xe = self._get_prevs_xe(x,e)
            return self.f_std_s(tc.cat([c, prev_xe], dim=-1))
        else:
            return tensorify(x.device, self.std_s_val)[0].expand(x.shape[:-1]+e.shape[:-1]+(self.dim_s,))
    
    def c1xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_c))\
        and (not is_same_tensor(t, self._t_cache_c))\
        and (not is_same_tensor(e, self._e_cache_c)):
            self._x_cache_c = x
            self._t_cache_c = t
            self._e_cache_c = e
            self._c_cache = self.f_prev2c(self._get_prevs_xte(x,t,e))
        return self._c_cache
    
    def s1cxe(self, c, x, e):
        prev_xe = self._get_prevs_xe(x,e)
        return self.f_prevs2s(tc.cat([c, prev_xe], dim=-1))
    
    def y1c(self, c):
        '''
        q(y|c)
        '''
        return self.f_c2y(c)
    
    def std_y1c(self, c):
        return self.f_std_y1c(c)

    def forward(self, x,t,e):
        '''
        q(y|c) = q(y|c)q(c,s|x,t,e)
        '''
        c = self.c1xte(x,t,e)
        s = self.s1cxe(c,x,e)
        return self.y1c(c)

class MLPx1cs(MLPBase):
    def __init__(self, dim_c = None, dims_prec2paras = None, dim_s = None, dims_pres2postx = None, dim_x = None,
                 dim_t = None, dim_e = None, actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if dim_t is None: dim_t = discr.dim_x
        if dim_e is None: dim_e = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        # if dims_prec2paras is None: dims_prec2paras = discr.dims_posts2c[::-1][1:] + [discr.dim_paras]
        if dims_pres2postx is None: dims_pres2postx = discr.dims_postx2pres
        super(MLPx1cs, self).__init__()
        self.dim_c, self.dim_s, self.dim_x = dim_c, dim_s, dim_x
        # self.dims_prec2paras, self.dims_pres2postx, self.actv = dims_prec2paras, dims_pres2postx, actv
        self.dims_pres2postx, self.actv = dims_pres2postx, actv
        # self.f_c2paras = mlp_constructor([dim_c] + dims_prec2paras, actv, lastactv=False)
        self.f_vparas2x = mlp_constructor([dim_s + dim_c] + dims_pres2postx + [dim_x], actv, lastactv = False)

        # Env | S
        self.f_vparas2e = mlp_constructor([dim_s] + dims_pres2postx + [dim_x], actv, lastactv = False)
 
    def x1cs(self, c, s): return self.f_vparas2x(tc.cat([s, c], dim=-1))
    def e1s(self, s): return self.f_vparas2e(s)
    # def t1c(self, c): return self.f_vparas2t(self.f_c2paras(c))
    def forward(self, c, s): return self.x1cs(c, s), self.e1s(s)

class MLPs1x(MLPBase):
    def __init__(self, dim_s = None, std_s1x_val = -1., dims_t2c = None, dims_posts2prey = None,dims_pres2postx = None, dim_x = None, dim_y = None,
                 after_actv=None, actv = None):
        if actv is None: actv = "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        super(MLPs1x, self).__init__()
        self.dim_s, self.dim_x, self.dims_pres2postx, self.actv = dim_s, dim_x, dims_pres2postx, actv
        self.shape_s = (dim_s,)
        self.f_x2s = mlp_constructor([dim_x] + dims_pres2postx + [dim_s], actv, lastactv=False)
        self.f_s2y = mlp_constructor([dim_s] + dims_posts2prey + [dim_y], actv, lastactv = False)
        self.learn_std_s = std_s1x_val <= 0 if type(std_s1x_val) is float else (std_s1x_val <= 0).any()

        if self.learn_std_s:
            self.nn_std_s = nn.Sequential(
                    mlp_constructor(
                        [dim_x] + dims_pres2postx + [dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = self.nn_std_s

        self.nn_std_y1s = nn.Sequential(
                                mlp_constructor(
                                [dim_s] + dims_posts2prey + [dim_y],
                                nn.ReLU, lastactv = False),
                                nn.Softplus())
        init_linear(self.nn_std_y1s, 0., 1e-2, 0.)
        self.f_std_y1s = self.nn_std_y1s

    def s1x(self, x): return self.f_x2s(x)
    def std_s1x(self, x): return self.f_std_s(x)
    def y1s(self, s): return self.f_s2y(s)
    def std_y1s(self, s): return self.f_std_y1s(s)
    def forward(self, x): return self.y1s(self.s1x(x))

class MLPx1s(MLPBase):
    def __init__(self, dim_s = None, dims_pres2postx = None, dim_x = None,
            actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_x is None: dim_x = discr.dim_x
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postx is None:
            dims_pres2postx = discr.dims_pres2postx[::-1]
        super(MLPx1s, self).__init__()
        self.dim_s, self.dim_x, self.dims_pres2postx, self.actv = dim_s, dim_x, dims_pres2postx, actv
        self.f_s2x = mlp_constructor([dim_s] + dims_pres2postx + [dim_x], actv, lastactv=False)

    def x1s(self, s): return self.f_s2x(s)
    def forward(self, s): return self.x1s(s)

class MLPv1s(MLPBase):
    def __init__(self, dim_c = None, dims_pres2postv = None, dim_s = None,
            actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_s is None: dim_s = discr.dim_s
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if type(actv) is str: actv = getattr(nn, actv)
        if dims_pres2postv is None: dims_pres2postv = discr.dims_posts2c[::-1][1:]
        super(MLPv1s, self).__init__()
        self.dim_c, self.dim_s, self.dims_pres2postv, self.actv = dim_c, dim_s, dims_pres2postv, actv
        self.f_s2v = mlp_constructor([dim_c] + dims_pres2postv + [dim_s], actv)

    def v1s(self, s): return self.f_s2v(s)
    def forward(self, s): return self.v1s(s)
    
class MLPc1t(MLPBase):
    def __init__(self, dim_c = None, dims_postx2pres = None,
                 dim_t = None, actv = None, *, discr = None):
        if dim_c is None: dim_c = discr.dim_c
        if dim_t is None: dim_t = discr.dim_t
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if dims_postx2pres is None: dims_postx2pres = discr.dims_t2c
        super(MLPc1t, self).__init__()
        self.dim_c, self.dim_t = dim_c, dim_t
        self.dims_postx2pres, self.actv = dims_postx2pres, actv
        
        slope = 0.1
        self.input_dim = dim_t
        self.output_dim = dim_c
        self.n_layers = len(self.dims_postx2pres)
        
        if isinstance(dims_postx2pres, Number):
            self.hidden_dim = [dims_postx2pres] * (self.n_layers - 1)
        elif isinstance(dims_postx2pres, list):
            self.hidden_dim = dims_postx2pres
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(dims_postx2pres))

        if isinstance(actv, str):
            self.activation = [actv] * (self.n_layers - 1)
        elif isinstance(actv, list):
            self.hidden_dim = actv
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(actv))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def std_c1t(self, t):
        h = t
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h.exp()
    
    def forward(self, t):
        return self.std_c1t(t)

class MLPs1ct(MLPBase):
    def __init__(self, dim_s = None, dims_postx2pres = None,
                 dim_t = None, actv = None, *, discr = None):
        if dim_s is None: dim_s = discr.dim_s
        if dim_t is None: dim_t = discr.dim_t
        if actv is None: actv = discr.actv if hasattr(discr, "actv") else "Sigmoid"
        if dims_postx2pres is None: dims_postx2pres = discr.dims_posts2c[::-1][1:] + [discr.dim_paras, discr.dim_paras]
        super(MLPs1ct, self).__init__()
        self.dim_s, self.dim_t = dim_s, dim_t
        self.dims_postx2pres, self.actv = dims_postx2pres, actv
        
        slope = 0.1
        self.input_dim = dim_s + dim_t
        self.output_dim = dim_s
        self.n_layers = len(self.dims_postx2pres)
        
        if isinstance(dims_postx2pres, Number):
            self.hidden_dim = [dims_postx2pres] * (self.n_layers - 1)
        elif isinstance(dims_postx2pres, list):
            self.hidden_dim = dims_postx2pres
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(dims_postx2pres))

        if isinstance(actv, str):
            self.activation = [actv] * (self.n_layers - 1)
        elif isinstance(actv, list):
            self.hidden_dim = actv
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(actv))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
        self.fc = nn.ModuleList(_fc_list)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def std_s1ct(self, c, t):
        h = tc.cat([c, t], dim=-1)
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h.exp()
    
    def forward(self, c, t):
        return self.std_s1ct(c, t)
        
def create_discr_from_json(stru_name: str, dim_x: int, dim_y: int, dim_t: int, actv: str=None,
        std_s1xte_val: float=-1., std_c1sxt_val: float=-1., after_actv: bool=True, ind_cs: bool=False, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPcsy1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPcsy1x(dim_x=dim_x, dim_y=dim_y, dim_t=dim_t, std_s1xte_val=std_s1xte_val, std_c1sxt_val=std_c1sxt_val,
            after_actv=after_actv, ind_cs=ind_cs, **stru)

def create_ccl_discr_from_json(stru_name: str, dim_x: int, dim_y: int, dim_t: int, actv: str=None,
        std_s1xte_val: float=-1., std_c1sxt_val: float=-1., after_actv: bool=True, ind_cs: bool=False, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPcsy1xte'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPcsy1xte(dim_x=dim_x, dim_y=dim_y, dim_t=dim_t, std_s_val=std_s1xte_val, std_c_val=std_c1sxt_val,
            after_actv=after_actv, ind_cs=ind_cs, **stru)

def create_vae_discr_from_json(stru_name: str, dim_x: int, dim_y: int, actv: str=None,
                               std_s1x_val: float=-1., after_actv: bool=True, jsonfile: str="mlpstru.json"):
    stru = json.load(open(jsonfile))['MLPs1x'][stru_name]
    if actv is not None: stru['actv'] = actv
    return MLPs1x(dim_x=dim_x, dim_y=dim_y, std_s1x_val=std_s1x_val,
                  after_actv=after_actv, **stru)

def create_gen_from_json(model_type: str="MLPx1cs", discr: MLPcsy1x=None, stru_name: str=None, dim_x: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_x=dim_x, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_x=dim_x, discr=discr, **stru)
    
def create_prior_from_json(model_type: str="MLPc1t", discr: MLPcsy1x=None, stru_name: str=None, dim_c: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_c=dim_c, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_c=dim_c, discr=discr, **stru)

def create_s_prior_from_json(model_type: str="MLPs1ct", discr: MLPcsy1x=None, stru_name: str=None, dim_s: int=None, actv: str=None, jsonfile: str="mlpstru.json"):
    if stru_name is None:
        return eval(model_type)(dim_s=dim_s, discr=discr, actv=actv)
    else:
        stru = json.load(open(jsonfile))[model_type][stru_name]
        if actv is not None: stru['actv'] = actv
        return eval(model_type)(dim_s=dim_s, discr=discr, **stru)

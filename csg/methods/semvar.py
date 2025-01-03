#!/usr/bin/env python3.6
''' The Semantic-Variation Generative Model.

I.e., the proposed Causal Semantic Generative model (CSG).
'''
import sys
import math
import torch as tc
sys.path.append('..')
import distr as ds
from . import xdistr as xds

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

class SemVar:
    @staticmethod
    def _get_priors(mean_c, std_c, shape_c, mean_s, std_s, std_s1ct, shape_s, corr_cs, mvn_prior: bool = False, device = None):
        
        if callable(std_c):
            print("Set prior for CCL")
            p_c = ds.Normal('c', mean=mean_c, std=std_c, shape=shape_c)
            p_s = ds.Normal('s', mean=mean_s, std=std_s, shape=shape_s)

            if std_s1ct is not None:
                p_s1ct = ds.Normal('s', mean=mean_s, std=std_s1ct, shape=shape_s)
            else:
                p_s1ct = None

            return p_c, p_s1ct, p_s, []
        if not mvn_prior:
            if not callable(mean_c): mean_c_val = mean_c; mean_c = lambda: mean_c_val
            if not callable(std_c): std_c_val = std_c; std_c = lambda: std_c_val
            if not callable(mean_s): mean_s_val = mean_s; mean_s = lambda: mean_s_val
            if not callable(std_s): std_s_val = std_s; std_s = lambda: std_s_val
            if not callable(corr_cs):
                if not corr_cs**2 < 1.: raise ValueError("correlation coefficient larger than 1")
                corr_cs_val = corr_cs; corr_cs = lambda: corr_cs_val

            p_c = ds.Normal('c', mean=mean_c, std=std_c, shape=shape_c)
            dim_c, dim_s = tc.tensor(shape_c).prod(), tc.tensor(shape_s).prod()
            def mean_s1c(c):
                shape_bat = c.shape[:-len(shape_c)] if len(shape_c) else c.shape
                c_normal_flat = ((c - mean_c()) / std_c()).reshape(shape_bat+(dim_c,))
                s_normal_flat = c_normal_flat[..., :dim_s] if dim_s <= dim_c \
                    else tc.cat([c_normal_flat, tc.zeros(shape_bat+(dim_s-dim_c,), dtype=c.dtype, device=c.device)], dim=-1)
                
                return mean_s() + corr_cs() * std_s() * s_normal_flat.reshape(shape_bat+shape_s)
            def std_s1c(c):
                corr_cs_val = ds.tensorify(device, corr_cs())[0]

                return ( std_s() * (1. - corr_cs_val**2).sqrt()
                        ).expand( (c.shape[:-len(shape_c)] if len(shape_c) else c.shape) + shape_s )
            p_s1c = ds.Normal('s', mean=mean_s1c, std=std_s1c, shape=shape_s)
            p_s = ds.Normal('s', mean=mean_s, std=std_s, shape=shape_s)
            prior_params_list = []
        else:
            print("MVN PRIOR!!!")
            if len(shape_c) != 1 or len(shape_s) != 1:
                raise RuntimeError("only 1-dim vectors are supported for `c` and `s` in `mvn_prior` mode")
            dim_c = shape_c[0]; dim_s = shape_s[0]
            mean_c = tc.zeros(shape_c, device=device) if callable(mean_c) else ds.tensorify(device, mean_c)[0].expand(shape_c).clone().detach()
            mean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
            # Sigma_cs = L_cs L_cs^T, L_cs = (L_cc, 0; M_sc, L_ss)
            std_c_offdiag = tc.zeros((dim_c, dim_c), device=device) # lower triangular of L_ss (excl. diag)
            std_s_offdiag = tc.zeros((dim_s, dim_s), device=device) # lower triangular of L_ss (excl. diag)
            if callable(std_c): # for diag of L_ss
                std_c_diag_param = tc.zeros(shape_c, device=device)
            else:
                std_c = ds.tensorify(device, std_c)[0].expand(shape_c)
                std_c_diag_param = std_c.log().clone().detach()
            if callable(std_s): # for diag of L_ss
                std_s_diag_param = tc.zeros(shape_s, device=device)
            else:
                std_s = ds.tensorify(device, std_s)[0].expand(shape_s)
                std_s_diag_param = std_s.log().clone().detach()
            if any(callable(var) for var in [std_c, std_s, corr_cs]): # M_sc
                std_sc_mat = tc.zeros(dim_s, dim_s, device=device)
            else:
                std_sc_mat = tc.eye(dim_s, dim_c, device=device)
                dim_min = min(dim_s, dim_c)
                std_sc_diag = (ds.tensorify(device, corr_cs)[0].expand((dim_min,)) * std_c[:dim_min] * std_s[:dim_min]).sqrt()
                if dim_min == dim_c: std_sc_mat = (std_sc_mat @ std_sc_diag.diagflat()).clone().detach()
                else: std_sc_mat = (std_sc_diag.diagflat() @ std_sc_mat).clone().detach()
            prior_params_list = [mean_c, std_c_diag_param, std_c_offdiag, mean_s, std_s_diag_param, std_s_offdiag, std_sc_mat]

            def std_c_tril(): # L_cc
                return std_c_offdiag.tril(-1) + std_c_diag_param.exp().diagflat()
            p_c = ds.MVNormal('c', mean=mean_c, std_tril=std_c_tril, shape=shape_c)

            def mean_s1c(c):
                return mean_s + ( std_sc_mat @ tc.linalg.solve_triangular(
                                  std_c_tril(), (c - mean_c).unsqueeze(-1), upper=False)[0]).squeeze(-1)
            def std_s1c_tril(c): # L_ss
                return ( std_s_offdiag.tril(-1) + std_s_diag_param.exp().diagflat()
                        ).expand( (c.shape[:-len(shape_c)] if len(shape_c) else c.shape) + (dim_s, dim_s) )
            p_s1c = ds.MVNormal('s', mean=mean_s1c, std_tril=std_s1c_tril, shape=shape_s)

            def cov_s(): # M_sc M_sc^T + L_ss L_ss^T
                L_ss = std_s_offdiag.tril(-1) + std_s_diag_param.exp().diagflat()
                return std_sc_mat @ std_sc_mat.T + L_ss @ L_ss.T
        
            p_s = ds.MVNormal('s', mean=mean_s, cov=cov_s, shape=shape_s)
        return p_c, p_s1c, p_s, prior_params_list

    def __init__(self, shape_c, shape_s, shape_x, dim_y,
            mean_x1cs, std_x1cs, mean_e1s, std_e1s, mean_y1c, std_y1c,
            mean_s1x = None, std_s1x = None, mean_c1sx = None, std_c1sx = None,
            tmean_s1xe = None, tstd_s1xe = None, tmean_c1xt = None, tstd_c1xt = None,
            mean_c = 0., std_c = 1., mean_s = 0., std_s = 1., std_s1ct = None, corr_cs = .5,
            learn_tprior = False, src_mvn_prior = False, tgt_mvn_prior = False, device = None):
        '''
        p(x,t,e|c,s) = p(x|c,s)p(e|s)
        mean_x1cs: mean of p(x|c,s) <= gen.x1cs : Causal path
        mean_e1s: mean of p(e|s) <= gen.e1s : Causal path
        mean_t1cs: mean of p(t|c) <= gen.t1c : 
        mean_y1c: logits of p(y|s) <= discr.y1s : Causal path
        mean_s1x: mean of q(s|x,t,e) <= discr.s1x : computational path
        mean_c1sx: mean of q(c|s,x) <= discr.c1sx : computational path
        '''
        self.src_mvn_prior = src_mvn_prior
        
        if device is not None: ds.Distr.default_device = device
        self._parameter_dict = {}
        self.shape_c, self.shape_s, self.shape_x, self.dim_y = shape_c, shape_s, shape_x, dim_y
        self.learn_tprior = learn_tprior

        # P(x|c,s)
        self.p_x1cs = ds.Normal('x', mean=mean_x1cs, std=std_x1cs, shape=shape_x)
        
        # P(y|c) : causal mechanism
        self.p_y1c = ds.Normal('y', mean=mean_y1c, std=std_y1c, shape=shape_x)
        self.p_e1s = ds.Normal('e', mean=mean_e1s, std=std_e1s, shape=shape_x)

        self.p_c1t, self.p_s1c, self.p_s, prior_params_list = self._get_priors(
                mean_c, std_c, shape_c, mean_s, std_s, std_s1ct, shape_s, corr_cs, src_mvn_prior, device)
        if src_mvn_prior: self._parameter_dict.update(zip([
                'mean_c', 'std_c_diag_param', 'std_c_offdiag', 'mean_s', 'std_s_diag_param', 'std_s_offdiag', 'std_sc_mat'
            ], prior_params_list))

        if self.p_s1c is not None:
            self.p_cs = self.p_c1t * self.p_s1c     # p(c, s|t)
        else:
            self.p_cs = self.p_c1t * self.p_s
        # self.p_csx = self.p_cs * self.p_x1cs * self.p_e1s  # p(s, c, x, e)

        if mean_s1x is not None: # None
            self.q_s1x = ds.Normal('s', mean=mean_s1x, std=std_s1x, shape=shape_s)
            self.q_c1sx = ds.Normal('c', mean=mean_c1sx, std=std_c1sx, shape=shape_c)
            self.q_cs1x = self.q_s1x * self.q_c1sx
        else: self.q_s1x, self.q_c1sx, self.q_cs1x = None, None, None

        if tmean_s1xe is not None:
            self.qt_s1x = ds.Normal('s', mean=tmean_s1xe, std=tstd_s1xe, shape=shape_s)
            self.qt_c1x = ds.Normal('c', mean=tmean_c1xt, std=tstd_c1xt, shape=shape_c)
            self.qt_cs1x = self.qt_s1x * self.qt_c1x
        else: self.qt_s1x, self.qt_c1x, self.qt_cs1x = None, None, None

        if learn_tprior:
            if not tgt_mvn_prior:
                tmean_c = tc.zeros(shape_c, device=device) if callable(mean_c) else ds.tensorify(device, mean_c)[0].expand(shape_c).clone().detach()
                tmean_s = tc.zeros(shape_s, device=device) if callable(mean_s) else ds.tensorify(device, mean_s)[0].expand(shape_s).clone().detach()
                tstd_c_param = tc.zeros(shape_c, device=device) if callable(std_c) else ds.tensorify(device, std_c)[0].expand(shape_c).log().clone().detach()
                tstd_s_param = tc.zeros(shape_s, device=device) if callable(std_s) else ds.tensorify(device, std_s)[0].expand(shape_s).log().clone().detach()
                if callable(corr_cs): tcorr_cs_param = tc.zeros((), device=device)
                else:
                    val = (ds.tensorify(device, corr_cs)[0].reshape(()) + 1.) / 2.
                    tcorr_cs_param = (val / (1-val)).clone().log().detach()
                self._parameter_dict.update({'tmean_c': tmean_c, 'tmean_s': tmean_s,
                    'tstd_c_param': tstd_c_param, 'tstd_s_param': tstd_s_param, 'tcorr_cs_param': tcorr_cs_param})

                def tstd_c(): return tc.exp(tstd_c_param)
                def tstd_v(): return tc.exp(tstd_s_param)
                def tcorr_cs(): return 2. * tc.sigmoid(tcorr_cs_param) - 1.
                self.pt_c, self.pt_s1c, self.pt_s, tprior_params_list = self._get_priors(
                        tmean_c, tstd_c, shape_c, tmean_s, tstd_v, shape_s, tcorr_cs, False, device)
            else:
                self.pt_c, self.pt_s1c, self.pt_s, tprior_params_list = self._get_priors(
                        mean_c, std_c, shape_c, mean_s, std_s, shape_s, corr_cs, True, device)
                self._parameter_dict.update(zip([
                        'tmean_c', 'tstd_c_diag_param', 'tstd_c_offdiag', 'tmean_s', 'tstd_s_diag_param', 'tstd_s_offdiag', 'tstd_sc_mat'
                    ], tprior_params_list))
        else: self.pt_c, self.pt_s1c, self.pt_s = self.p_c1t, self.p_s1c, self.p_s # independent prior
        self.pt_cs = self.pt_c * self.pt_s
        self.pt_csx = self.pt_cs * self.p_x1cs * self.p_e1s
        for param in self._parameter_dict.values(): param.requires_grad_()

    def parameters(self):
        for param in self._parameter_dict.values(): yield param

    def state_dict(self):
        return self._parameter_dict

    def load_state_dict(self, state_dict: dict):
        for name in list(self._parameter_dict.keys()):
            with tc.no_grad(): self._parameter_dict[name].copy_(state_dict[name])

    def get_lossfn(self, n_mc_q: int=0, reduction: str="mean", mode: str="defl", weight_da: float=None, wlogpi: float=None):
        if reduction == "mean": reducefn = tc.mean
        elif reduction == "sum": reducefn = tc.sum
        elif reduction is None or reduction == "none": reducefn = lambda x: x
        else: raise ValueError(f"unknown `reduction` '{reduction}'")

        if mode == 'ccl':
            if self.p_s1c is None:
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    return xds.ccl_elbo(qt_z1x=self.qt_cs1x, p_y1c=self.p_y1c, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_c1t=self.pt_c, p_s=self.pt_s,
                                        q_c1x=self.qt_c1x, q_s1x=self.qt_s1x, 
                                        obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q, wlogpi=wlogpi)
            else:
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    return xds.ccl_elbo(qt_z1x=self.qt_cs1x, p_y1c=self.p_y1c, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_c1t=self.pt_c, p_s=self.pt_s,
                                        q_c1x=self.qt_c1x, q_s1x=self.qt_s1x, p_s1ct=self.p_s1c,
                                        obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q, wlogpi=wlogpi)
            return lossfn_src
        elif mode == 'debug':
            if self.p_s1c is None:
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    return xds.ccl_elbo_debug(qt_z1x=self.qt_cs1x, p_y1c=self.p_y1c, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_c1t=self.pt_c, p_s=self.pt_s,
                                        q_c1x=self.qt_c1x, q_s1x=self.qt_s1x, 
                                        obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q, wlogpi=wlogpi)
            else:
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    return xds.ccl_elbo_debug(qt_z1x=self.qt_cs1x, p_y1c=self.p_y1c, p_x1cs=self.p_x1cs, p_e1s=self.p_e1s, p_c1t=self.pt_c, p_s=self.pt_s,
                                        q_c1x=self.qt_c1x, q_s1x=self.qt_s1x, p_s1ct=self.p_s1c,
                                        obs_xyte={'x':x, 't':t, 'e':e, 'y':y}, n_mc=n_mc_q, wlogpi=wlogpi)
            return lossfn_src
        else:
            if self.learn_tprior: # svgm-da
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    return -reducefn( xds.elbo_z2xy_twist(self.pt_csx, self.p_y1c, self.p_cs, self.pt_cs, self.qt_cs1x, {'x':x, 't':t, 'e':e, 'y':y}, n_mc_q, wlogpi) )
                    # return -reducefn( xds.elbo_z2xy_twist_fixpt(self.p_x1cs, self.p_y1c, self.p_cs, self.pt_cs, self.qt_cs1x, {'x':x, 'y':y}, n_mc_q, wlogpi) )
            else: # svgm-ind
                def lossfn_src(x, t, e, y) -> tc.Tensor:
                    '''
                    self.pt_csx => pt_zx: p'(c,s,x,t,e)
                    self.p_y1c => p_y1z: p(y|c)
                    self.p_s1c => p_z: p(s|c)
                    self.pt_s1c => pt_z: p'(s)
                    self.qt_cs1x => qt_z1x: q'(c,s|x,t,e)
                    obs_xyte: mini-batch of (x,y,t,e)
                    '''
                    return -reducefn( xds.elbo_z2xy_twist(self.pt_csx, self.p_y1c, self.p_s1c, self.pt_s1c, self.qt_cs1x, {'x':x, 't':t, 'e':e, 'y':y}, n_mc_q, wlogpi) )
            
            def lossfn_tgt(xt, tt, et, yt) -> tc.Tensor:
                return -reducefn( ds.elbo(self.pt_csx, self.qt_cs1x, {'x': xt, 't': tt, 'e': et, 'y': yt}, n_mc_q) )
                # return -reducefn( xds.elbo_fixllh(self.pt_sv, self.p_x1sv, self.qt_sv1x, {'x': xt}, n_mc_q) )

            if not mode or mode == "defl":
                if self.learn_tprior:
                    def lossfn(x, t, e, y, xt, tt, et, yt) -> tc.Tensor:
                        return lossfn_src(x,t,e,y) + weight_da * lossfn_tgt(xt, tt, et, yt)
                    return lossfn
                else: return lossfn_src
            else: raise ValueError(f"unknown `mode` '{mode}'")

    # Utilities
    def llh(self, x: tc.Tensor, y: tc.LongTensor=None, n_mc_marg: int=64, use_q: bool=True, mode: str="src") -> float:
        if mode == "src":
            p_joint = self.p_csx
            q_cond = self.q_cs1x if self.q_cs1x else self.qt_cs1x
        elif mode == "tgt":
            p_joint = self.pt_csx
            q_cond = self.qt_cs1x if self.qt_cs1x else self.q_cs1x
        else: raise ValueError(f"unknown `mode` '{mode}'")
        if not use_q:
            if y is None: llh_vals = p_joint.marg({'x'}, n_mc_marg).logp({'x': x})
            else: llh_vals = (p_joint * self.p_y1c).marg({'x', 'y'}, n_mc_marg).logp({'x': x, 'y': y})
        else:
            if y is None: llh_vals = q_cond.expect(lambda dc: p_joint.logp(dc) - q_cond.logp(dc,dc),
                    {'x': x}, n_mc_marg, reducefn=tc.logsumexp) - math.log(n_mc_marg)
            else: llh_vals = q_cond.expect(lambda dc: (p_joint * self.p_y1c).logp(dc) - q_cond.logp(dc,dc),
                    {'x': x, 'y': y}, n_mc_marg, reducefn=tc.logsumexp) - math.log(n_mc_marg)
        return llh_vals.mean().item()

    def logit_y1x_src(self, x, t, e, n_mc_q: int=0, repar: bool=True):
        dim_y = self.dim_y
        y_eval = ds.expand_front(tc.arange(dim_y, device=x.device), ds.tcsize_div(x.shape, self.shape_x))
        x_eval = ds.expand_middle(x, (dim_y,), -len(self.shape_x))
        t_eval = ds.expand_middle(t, (dim_y,), -len(self.shape_x))
        e_eval = ds.expand_middle(e, (dim_y,), -len(self.shape_x))
        obs_xte = ds.edic({'x': x_eval, 't': t_eval, 'e': e_eval, 'y': y_eval})

        vwei_p_y1c_logp = lambda dc: self.p_cs.logp(dc,dc) - self.pt_cs.logp(dc,dc) + self.p_y1c.logp(dc,dc)
        infer_y1x = (self.qt_cs1x.expect(vwei_p_y1c_logp, obs_xte, 0, repar) #, reducefn=tc.logsumexp)
            ) if n_mc_q == 0 else (
                self.qt_cs1x.expect(vwei_p_y1c_logp, obs_xte, n_mc_q, repar, reducefn=tc.logsumexp) - math.log(n_mc_q)
            )
        return infer_y1x

    def generate(self, shape_mc: tc.Size=tc.Size(), mode: str="src") -> tuple:
        if mode == "src": smp_cs = self.p_cs.draw(shape_mc)
        elif mode == "tgt": smp_cs = self.pt_cs.draw(shape_mc)
        else: raise ValueError(f"unknown 'mode' '{mode}'")
        return self.p_x1cs.mode(smp_cs, False)['x'], self.p_y1c.mode(smp_cs, False)['y']
    
    def inference(self, x, t, e, shape_mc:tc.Size=tc.Size()):
        obs_xte = ds.edic({'x': x, 't': t, 'e': e})

        # c, s, x, y, e
        return self.qt_cs1x.draw(shape_mc=shape_mc, conds=obs_xte), self.p_x1cs.draw(shape_mc=shape_mc, conds=obs_xte), \
            self.p_y1c.draw(shape_mc=shape_mc, conds=obs_xte), self.p_e1s.draw(shape_mc=shape_mc, conds=obs_xte)

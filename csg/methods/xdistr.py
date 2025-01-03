#!/usr/bin/env python3.6
""" Modification to the `distr` package for the structure of the
    Causal Semantic Generative model.
"""
import sys
import math
import torch as tc
sys.path.append('..')
from distr import Distr, edic

import wandb

__author__ = "Chang Liu"
__email__ = "changliu@microsoft.com"

def elbo_z2xy(p_zx: Distr, p_y1z: Distr, q_z1x: Distr, obs_xy: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    """ For supervised VAE with structure x <- z -> y.
    Observations are supervised (x,y) pairs.
    For unsupervised observations of x data, use `elbo(p_zx, q_z1x, obs_x)` as VAE z -> x. """
    if n_mc == 0:
        q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(q_z1x, "entropy"): # No difference for Gaussian
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc), obs_xy, 0, repar) + q_z1x.entropy(obs_xy)
        else:
            expc_val = q_z1x.expect(lambda dc: p_zx.logp(dc,dc) - q_z1x.logp(dc,dc), obs_xy, 0, repar)
        return wlogpi * q_y1x_logpval + expc_val
    else:
        q_y1x_pval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp(), obs_xy, n_mc, repar)
        expc_val = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc).exp() * (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)),
                obs_xy, n_mc, repar)
        return wlogpi * q_y1x_pval.log() + expc_val / q_y1x_pval
        # q_y1x_logpval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc), obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc)
        # expc_logval = q_z1x.expect(lambda dc: p_y1z.logp(dc,dc) + (p_zx.logp(dc,dc) - q_z1x.logp(dc,dc)).log(),
        #         obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * q_y1x_logpval + (expc_logval - q_y1x_logpval).exp()

def elbo_z2xy_twist(pt_zx: Distr, p_y1z: Distr, p_z: Distr, pt_z: Distr, qt_z1x: Distr, obs_xyte: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    '''
    pt_zx: p'(c,s,x,t,e)
    p_y1z: p(y|c)
    p_z: p(s|c)
    pt_z: p'(s)
    qt_z1x: q'(c,s|x,t,e)
    obs_xyte: mini-batch of (x,y,t,e)
    '''

    # log([p(c,s|t)/p'(c,s|t)]*p(y|c))
    vwei_p_y1z_logp = lambda dc: p_z.logp(dc,dc) - pt_z.logp(dc,dc) + p_y1z.logp(dc,dc)
    if n_mc == 0:
        r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xyte, 1, repar, reducefn=tc.logsumexp) - math.log(1) # qt_z1x.expect(vwei_p_y1z_logp, obs_xyte, 0, repar)
        expc_val = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc), obs_xyte, 0, repar)
        
        wandb.log(
            {
                "Avg. log(Phi(y|x,t,e))": r_y1x_logpval.mean(),
                "Std. log(Phi(y|x,t,e))": r_y1x_logpval.std(),
                "Avg. expc_val": expc_val.mean(),
                "Std. expc_val": expc_val.std(),
            }
        )

        return wlogpi * r_y1x_logpval + expc_val
    else:
        log_phi = qt_z1x.expect(vwei_p_y1z_logp, obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)

        def debug_vals():
            expc1 = qt_z1x.expect(lambda dc: vwei_p_y1z_logp(dc).exp(), obs_xyte, n_mc, repar)
            expc2 = qt_z1x.expect(lambda dc: pt_zx.logp(dc,dc), obs_xyte, n_mc, repar)
            expc3 = qt_z1x.expect(lambda dc: qt_z1x.logp(dc,dc), obs_xyte, n_mc, repar)
            return expc1, expc2, expc3

        expc1, expc2, expc3 = debug_vals()

        expc_val = qt_z1x.expect( lambda dc:
                vwei_p_y1z_logp(dc).exp() * (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)),
                obs_xyte, n_mc, repar)
        
        wandb.log(
            {
            "log(Phi(y|x,t,e))": log_phi.sum(),
            "expc_val": expc_val.sum(),
            "expc_1": expc1.sum(),
            "expc_2": expc2.sum(),
            "expc_3": expc3.sum(),
            })
        
        return wlogpi * log_phi + (1/(log_phi.exp()+1e-6)) * expc_val
    
        # r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xyte, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc) # z, y:
        # expc_logval = qt_z1x.expect(lambda dc: # z, x, y:
        #         vwei_p_y1z_logp(dc) + (pt_zx.logp(dc,dc) - qt_z1x.logp(dc,dc)).log(),
        #     obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * r_y1x_logpval + (expc_logval - r_y1x_logpval).exp()

def ccl_elbo(qt_z1x: Distr, p_y1c: Distr, p_x1cs: Distr, p_e1s: Distr, p_c1t: Distr, p_s: Distr, q_c1x: Distr, q_s1x: Distr,
             obs_xyte: edic, p_s1ct=None, n_mc: int=1, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    
    if n_mc == 0:
        cs_samples = qt_z1x.draw((1,), obs_xyte)
        obs_xyte['c'] = cs_samples['c'].squeeze(0)
        obs_xyte['s'] = cs_samples['s'].squeeze(0)
        
        if p_s1ct is None:
            log_phi_y1xte = p_y1c.logp(obs_xyte, obs_xyte)
        else:
            log_phi_y1xte = (p_s1ct.logp(obs_xyte, obs_xyte) + p_y1c.logp(obs_xyte, obs_xyte) - p_s.logp(obs_xyte, obs_xyte))

        # p_x1cs
        expc_val_x = p_x1cs.logp(obs_xyte, obs_xyte)
        # p_e1s
        expc_val_e = p_e1s.logp(obs_xyte, obs_xyte)
        # p_c1t - q_c1x
        expc_val_c = (p_c1t.logp(obs_xyte, obs_xyte) - q_c1x.logp(obs_xyte, obs_xyte))
        # p_s - q_s1x
        expc_val_s = (p_s.logp(obs_xyte, obs_xyte) - q_s1x.logp(obs_xyte, obs_xyte))

        return wlogpi * log_phi_y1xte, expc_val_x, expc_val_e, expc_val_c, expc_val_s

    else:
        log_phi_y1xte = qt_z1x.expect(lambda dc: p_y1c.logp(dc, dc), obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        phi_y1xte = log_phi_y1xte.exp()

        expc_val = qt_z1x.expect(lambda dc: p_y1c.logp(dc, dc) + (p_x1cs.logp(dc, dc) + p_e1s.logp(dc, dc) + p_c1t.logp(dc, dc) + p_s.logp(dc, dc) - q_c1x.logp(dc, dc) - q_s1x.logp(dc, dc)).log(), 
                                obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        return wlogpi * log_phi_y1xte + (1/phi_y1xte) * expc_val

def ccl_elbo_debug(qt_z1x: Distr, p_y1c: Distr, p_x1cs: Distr, p_e1s: Distr, p_c1t: Distr, p_s: Distr, q_c1x: Distr, q_s1x: Distr,
             obs_xyte: edic, p_s1ct=None, n_mc: int=1, wlogpi: float=1., repar: bool=True) -> tc.Tensor:

    if n_mc == 0:
        cs_samples = qt_z1x.draw((1,), obs_xyte)
        obs_xyte['c'] = cs_samples['c'].squeeze(0)
        obs_xyte['s'] = cs_samples['s'].squeeze(0)
        
        if p_s1ct is None:
            log_phi_y1xte = p_y1c.logp(obs_xyte, obs_xyte)
        else:
            log_phi_y1xte = (p_s1ct.logp(obs_xyte, obs_xyte) + p_y1c.logp(obs_xyte, obs_xyte) - p_s.logp(obs_xyte, obs_xyte))

        # p_x1cs
        expc_val_x = p_x1cs.logp(obs_xyte, obs_xyte)
        # p_e1s
        expc_val_e = p_e1s.logp(obs_xyte, obs_xyte)
        # p_c1t - q_c1x
        expc_val_c = (p_c1t.logp(obs_xyte, obs_xyte) - q_c1x.logp(obs_xyte, obs_xyte))
        # p_s - q_s1x
        expc_val_s = (p_s.logp(obs_xyte, obs_xyte) - q_s1x.logp(obs_xyte, obs_xyte))

        return log_phi_y1xte, expc_val_x, expc_val_e, expc_val_c, expc_val_s

    else:
        log_phi_y1xte = qt_z1x.expect(lambda dc: p_y1c.logp(dc, dc), obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        phi_y1xte = log_phi_y1xte.exp()

        expc_val = qt_z1x.expect(lambda dc: p_y1c.logp(dc, dc) + (p_x1cs.logp(dc, dc) + p_e1s.logp(dc, dc) + p_c1t.logp(dc, dc) + p_s.logp(dc, dc) - q_c1x.logp(dc, dc) - q_s1x.logp(dc, dc)).log(), 
                                obs_xyte, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        return wlogpi * log_phi_y1xte + (1/phi_y1xte) * expc_val

def elbo_fixllh(p_prior: Distr, p_llh: Distr, q_cond: Distr, obs: edic, n_mc: int=10, repar: bool=True) -> tc.Tensor: # [shape_bat] -> [shape_bat]
    def logp_llh_nograd(dc):
        with tc.no_grad(): return p_llh.logp(dc,dc)
    if hasattr(q_cond, "entropy"):
        return q_cond.expect(lambda dc: p_prior.logp(dc,dc) + logp_llh_nograd(dc),
                obs, n_mc, repar) + q_cond.entropy(obs)
    else:
        return q_cond.expect(lambda dc: p_prior.logp(dc,dc) + logp_llh_nograd(dc) - q_cond.logp(dc,dc),
                obs, n_mc, repar)

def elbo_z2xy_twist_fixpt(p_x1z: Distr, p_y1z: Distr, p_z: Distr, pt_z: Distr, qt_z1x: Distr, obs_xy: edic, n_mc: int=0, wlogpi: float=1., repar: bool=True) -> tc.Tensor:
    def logpt_z_nograd(dc):
        with tc.no_grad(): return pt_z.logp(dc,dc)
    vwei_p_y1z_logp = lambda dc: p_z.logp(dc,dc) - logpt_z_nograd(dc) + p_y1z.logp(dc,dc) # z, y:
    if n_mc == 0:
        r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, 0, repar) #, reducefn=tc.logsumexp)
        if hasattr(qt_z1x, "entropy"):
            expc_val = qt_z1x.expect(lambda dc: logpt_z_nograd(dc) + p_x1z.logp(dc,dc),
                    obs_xy, 0, repar) + qt_z1x.entropy(obs_xy)
        else:
            expc_val = qt_z1x.expect(lambda dc: logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc), obs_xy, 0, repar)
        return wlogpi * r_y1x_logpval + expc_val
    else:
        r_y1x_pval = qt_z1x.expect(lambda dc: vwei_p_y1z_logp(dc).exp(), obs_xy, n_mc, repar)
        expc_val = qt_z1x.expect( lambda dc: # z, x, y:
                vwei_p_y1z_logp(dc).exp() * (logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc)),
            obs_xy, n_mc, repar)
        return wlogpi * r_y1x_pval.log() + expc_val / r_y1x_pval
        # r_y1x_logpval = qt_z1x.expect(vwei_p_y1z_logp, obs_xy, n_mc, repar,
        #         reducefn=tc.logsumexp) - math.log(n_mc) # z, y:
        # expc_logval = qt_z1x.expect(lambda dc: # z, x, y:
        #         vwei_p_y1z_logp(dc) + (logpt_z_nograd(dc) + p_x1z.logp(dc,dc) - qt_z1x.logp(dc,dc)).log(),
        #     obs_xy, n_mc, repar, reducefn=tc.logsumexp) - math.log(n_mc)
        # return wlogpi * r_y1x_logpval + (expc_logval - r_y1x_logpval).exp()


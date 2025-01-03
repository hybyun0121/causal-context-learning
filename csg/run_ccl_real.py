import os
import argparse
import tqdm
import tempfile
import pandas as pd
import numpy as np
import torch as tc
import torchvision as tv
import torch.nn.functional as F

import distr as ds

from torch.utils.data import DataLoader, random_split

from distr import edic
from arch import mlp
from methods import SemVar

from copy import deepcopy
from functools import partial
from utils import Averager, unique_filename, boolstr, zip_longer, print_infrstru_info, EarlyStopping, compute_cossim

from src.utils import set_seed_all

# Synthetic Data
from utils_data import HighDimSCMRealWorldDataset

from dalib.modules.domain_discriminator import DomainDiscriminator
from dalib.modules.kernels import GaussianKernel
from dalib.adaptation.dann import DomainAdversarialLoss
from dalib.adaptation.cdan import ConditionalDomainAdversarialLoss
from dalib.adaptation.dan import MultipleKernelMaximumMeanDiscrepancy
from dalib.adaptation.mdd import MarginDisparityDiscrepancy

import wandb

class ParamGroupsCollector:
    def __init__(self, lr):
        self.reset(lr)

    def reset(self, lr):
        self.lr = lr
        self.param_groups = []

    def collect_params(self, *models):
        for model in models:
            if hasattr(model, 'parameter_groups'):
                groups_inc = list(model.parameter_groups())
                for grp in groups_inc:
                    if 'lr_ratio' in grp:
                        grp['lr'] = self.lr * grp['lr_ratio']
                    elif 'lr' not in grp: # Do not overwrite existing lr assignments
                        grp['lr'] = self.lr
                self.param_groups += groups_inc
            else:
                self.param_groups += [
                        {'params': model.parameters(), 'lr': self.lr} ]

class ShrinkRatio:
    def __init__(self, w_iter, decay_rate):
        self.w_iter = w_iter
        self.decay_rate = decay_rate

    def __call__(self, n_iter):
        return (1 + self.w_iter * n_iter) ** (-self.decay_rate)

class ResultsContainer:
    def __init__(self, len_ts, frame, ag, is_binary, device, ckpt = None):
        for k,v in locals().items():
            if k not in {"self", "ckpt"}: setattr(self, k, v)
        self.dc = dict( epochs = [], losses = [],
                accs_tr = [], llhs_tr = [], accs_val = [], llhs_val = [] )
        if len_ts:
            ls_empty = [[] for _ in range(len_ts)]
            self.dc.update( ls_accs_ts = ls_empty, ls_llhs_ts = deepcopy(ls_empty) )
        else:
            self.dc.update( accs_ts = [], llhs_ts = [] )
        # if ckpt is not None:
        #     for k in self.dc.keys():
        #         if not k.startswith('ls_'): self.dc[k] = ckpt[k]
        #         else: self.dc[k] = [ckpt[k[3:]]]

    def update(self, *, epk, loss):
        self.dc['epochs'].append(epk)
        self.dc['losses'].append(loss)

    def evaluate(self, discr, dname, dpart, loader, llh_mode, i = None):
        acc = evaluate_acc(discr, loader, self.device)
        if i is None: self.dc['accs_'+dpart].append(acc)
        else: self.dc['ls_accs_'+dpart][i].append(acc)
        print(f"On {dname}, acc: {acc:.3f}", end="", flush=True)
        if self.frame is not None and self.ag.eval_llh:
            llh = evaluate_llhx(
                    self.frame, loader, self.ag.n_marg_llh, self.ag.use_q_llh, llh_mode, self.device)
            if i is None: self.dc['llhs_'+dpart].append(llh)
            else: self.dc['ls_llhs_'+dpart][i].append(llh)
            print(f", llhs: {llh:.3e}. ", end="", flush=True)
        else: print(". ", end="", flush=True)

    def summary(self, dname, dpart, i = None):
        if i is None:
            accs = self.dc['accs_'+dpart]
            llhs = self.dc['llhs_'+dpart]
        else:
            accs = self.dc['ls_accs_'+dpart][i]
            llhs = self.dc['ls_llhs_'+dpart][i]
        acc_fin = tc.tensor(accs[-self.ag.avglast:]).mean().item()
        llh_fin = tc.tensor(llhs[-self.ag.avglast:]).mean().item() if llhs[-self.ag.avglast:] else None
        acc_max = tc.tensor(accs).max().item()
        llh_max = tc.tensor(llhs).max().item() if llhs else None
        print(f"On {dname}, final acc: {acc_fin:.3f}" + (
                f", llh: {llh_fin:.3f}" if llh_fin else ""
            ) + f", max acc: {acc_max:.3f}" + (
                f", llh: {llh_max:.3f}" if llh_max else "") + ".")

def eval_with_pred(model: tc.nn.Module, x,t,e,y) -> float:
    with tc.no_grad():
        logits = model(x,t,e)
    # return mse_loss

# Init models
def auto_load(dc_vars, names, ckpt):
    if ckpt:
        if type(names) is str: names = [names]
        for name in names:
            model = dc_vars[name]
            model.load_state_dict(ckpt[name+'_state_dict'])
            if hasattr(model, 'eval'): model.eval()

def get_frame(discr, gen, dc_vars, device = None, discr_src = None):
    if type(dc_vars) is not edic:
        dc_vars = edic(dc_vars)

    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_c = discr.shape_c if hasattr(discr, "shape_c") else (dc_vars['dim_c'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)
    if dc_vars['ind_cs']:
        if dc_vars['is_given_c']:
            tmean_s=discr.s1cxe
            tmean_c=discr.c1xte
            qstd_s = discr.std_s1cxe if hasattr(discr, "std_s1cxe") else dc_vars['qstd_s']
            qstd_c = discr.std_c1xte if hasattr(discr, "std_c1xte") else dc_vars['qstd_c']
        else:
            tmean_s=discr.s1xe
            tmean_c=discr.c1xt
            qstd_s = discr.std_s1xe if hasattr(discr, "std_s1xe") else dc_vars['qstd_s']
            qstd_c = discr.std_c1xt if hasattr(discr, "std_c1xt") else dc_vars['qstd_c']
    else:
        qstd_s = discr.std_s1xte if hasattr(discr, "std_s1xte") else dc_vars['qstd_s']
        qstd_c = discr.std_c1sxt if hasattr(discr, "std_c1sxt") else dc_vars['qstd_c']

    std_c1x = discr.std_c1x if hasattr(discr, "std_c1x") else dc_vars['qstd_c']
    mode = dc_vars['mode']

    if dc_vars['cond_prior']:
        prior_std = mlp.create_prior_from_json("MLPc1t", discr, actv=dc_vars['actv_prior'],jsonfile=dc_vars['mlpstrufile']).to(device)
        std_c = prior_std.std_c1t
    else:
        std_c = dc_vars['sig_c']

    if dc_vars['is_corr_cs']:
        s_prior_std = mlp.create_s_prior_from_json("MLPs1ct", discr, actv=dc_vars['actv_prior'],jsonfile=dc_vars['mlpstrufile']).to(device)
        std_s1ct = s_prior_std.std_s1ct
    else:
        std_s1ct = None

    frame = SemVar(shape_c=shape_c, shape_s=shape_s, shape_x=shape_x, dim_y=dc_vars['dim_y'],
                   mean_x1cs=gen.x1cs, std_x1cs=dc_vars['pstd_x'],
                   mean_e1s=gen.e1s, std_e1s=dc_vars['pstd_e'], mean_y1c=discr.y1c, std_y1c=discr.std_y1c,
                   mean_s1x=None, std_s1x=None, mean_c1sx=None, std_c1sx=None,
                   tmean_s1xe=tmean_s, tstd_s1xe=qstd_s, tmean_c1xt=tmean_c, tstd_c1xt=qstd_c,
                   mean_c=dc_vars['mu_c'], std_c=std_c, mean_s=dc_vars['mu_s'], std_s = dc_vars['sig_s'], std_s1ct = std_s1ct, corr_cs=dc_vars['corr_cs'],
                   learn_tprior=mode == 'da', src_mvn_prior=dc_vars['src_mvn_prior'], tgt_mvn_prior=dc_vars['tgt_mvn_prior'], device=device
                   )
    return frame

def get_discr(archtype, dc_vars):
    if archtype == "mlp":
        if dc_vars['is_given_c']:
            discr = mlp.create_ccl_discr_from_json(
                    *dc_vars.sublist([
                        'discrstru', 'dim_x', 'dim_y', 'dim_t', 'actv',
                        'qstd_s', 'qstd_c', 'after_actv']),
                    ind_cs=dc_vars['ind_cs'],
                    jsonfile=dc_vars['mlpstrufile']
                )
        else:
            discr = mlp.create_discr_from_json(
                    *dc_vars.sublist([
                        'discrstru', 'dim_x', 'dim_y', 'dim_t', 'actv',
                        'qstd_s', 'qstd_c', 'after_actv']),
                    ind_cs=dc_vars['ind_cs'],
                    jsonfile=dc_vars['mlpstrufile']
                )
    else: raise ValueError(f"unknown `archtype` '{archtype}'")
    return discr

def get_gen(archtype, dc_vars, discr):
    if archtype == "mlp":
        gen = mlp.create_gen_from_json(
            "MLPx1cs", discr, dc_vars['genstru'], jsonfile=dc_vars['mlpstrufile'] )
    return gen

def get_models(archtype, dc_vars, ckpt = None, device = None):
    if type(dc_vars) is not edic: dc_vars = edic(dc_vars)
    discr = get_discr(archtype, dc_vars)
    if ckpt is not None:
        print("Loading discr from ckpt")
        auto_load(locals(), 'discr', ckpt)
    discr.to(device)

    gen = get_gen(archtype, dc_vars, discr)
    if ckpt is not None:
        print("Loading Gen from ckpt")
        auto_load(locals(), 'gen', ckpt)
    gen.to(device)

    frame = get_frame(discr, gen, dc_vars, device)
    if ckpt is not None:
        print("Loading Frame from ckpt")
        auto_load(locals(), 'frame', ckpt)
    
    return discr, gen, frame

def dc_state_dict(dc_vars, *name_list):
    return {name+"_state_dict" : dc_vars[name].state_dict()
            for name in name_list if hasattr(dc_vars[name], 'state_dict')}

def save_ckpt(ag, res, ckpt, dc_vars, discr, gen, frame, opt, IS_OOD=True, filename=None):
    dirname = "ckpt_" + ag.mode + "/" + ag.traindom + "/"
    os.makedirs(dirname, exist_ok=True)
    i = 0
    testdoms = ag.testdoms
    dim_x = dc_vars['dim_x']
    dim_y = dc_vars['dim_x']
    shape_x = dc_vars['shape_x'] if 'shape_x' in dc_vars else (dc_vars['dim_x'],)
    shape_c = discr.shape_c if hasattr(discr, "shape_c") else (dc_vars['dim_c'],)
    shape_s = discr.shape_s if hasattr(discr, "shape_s") else (dc_vars['dim_s'],)

    if filename is None:
        filename = unique_filename(
                dirname + (f"ood_{ag.emb_model}" if IS_OOD else f"da_{ag.emb_model}"), ".pt", n_digits = 3
            ) if ckpt is None else ckpt['filename']
        
    dc_vars = edic(locals()).sub([
            'dirname', 'filename', 'testdoms',
            'shape_x', 'shape_c', 'shape_s', 'dim_x', 'dim_y']) | ( edic(vars(ag)) - {'testdoms'}
        ) | dc_state_dict(locals(), "discr", "gen", "frame", "opt")

    tc.save(dc_vars, filename)
    print(f"checkpoint saved to '{filename}'.")
    return filename
    
def load_ckpt(filename: str, loadmodel: bool=False, device: tc.device=None, archtype: str="mlp", map_location: tc.device=None):
    ckpt = tc.load(filename, map_location)
    if loadmodel:
        return (ckpt,) + get_models(archtype, ckpt, ckpt, device)
    else: return ckpt

# Built methods
def get_ce_or_bce_loss(discr, dim_y: int, reduction: str="mean", mode='ind'):
    if dim_y == 1:
        lossobj = tc.nn.BCEWithLogitsLoss(reduction=reduction)
        lossfn = lambda x, y: lossobj(discr(x), y.float())
    else:
        lossobj = tc.nn.MSELoss(reduction=reduction)
        lossfn = lambda x: lossobj(discr(*x[:-1]), x[-1])

    return lossobj, lossfn

def add_ce_loss(lossobj, celossfn, ag):
    shrink_sup = ShrinkRatio(w_iter=ag.wsup_wdatum*ag.n_bat, decay_rate=ag.wsup_expo)

    def lossfn(*x_y_maybext_niter):
        '''
        x_y_maybext_niter: x, t, e, y, (xt, tt, et, yt,) niter
        '''
        if ag.mode == 'ind':
            log_phi_loss, x_recon_loss, e_recon_loss, c_recon_loss, s_recon_loss = lossobj(*x_y_maybext_niter[:-1]) # [batch_size]
            
            elbo = ag.wlogpi * log_phi_loss
            elbo += ag.wrecon_x * x_recon_loss
            elbo += ag.wrecon_e * e_recon_loss
            elbo += ag.wrecon_c * c_recon_loss
            elbo += ag.wrecon_s * s_recon_loss

            elbo = -1*elbo.mean()

            celoss = celossfn(x_y_maybext_niter[:-1])

        wandb.log(
        {
            "Elbo": elbo.item(),
            "Loss_sup": celoss.item(),
            "log_phi_loss": -1*log_phi_loss.mean().item(),
            "Loss_x": -1*x_recon_loss.mean().item(),
            "Loss_e": -1*e_recon_loss.mean().item(),
            "Loss_c": -1*c_recon_loss.mean().item(),
            "Loss_s": -1*s_recon_loss.mean().item()
        }
        )
        return ag.wgen * elbo + ag.wsup * shrink_sup(x_y_maybext_niter[-1]) * celoss

    return lossfn

def ood_methods(discr, frame, ag, dim_y):
    if ag.true_sup:
        celossfn = get_ce_or_bce_loss(partial(frame.logit_y1x_src, n_mc_q=ag.n_mc_q), dim_y, ag.reduction)[1]
    else:
        celossfn = get_ce_or_bce_loss(discr, dim_y, ag.reduction)[1]
    
    if ag.mode == 'ind' and not ag.debug:
        loss_mode = 'ccl'
    elif ag.mode == 'ind' and ag.debug:
        loss_mode = 'debug'
    
    lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, loss_mode, wlogpi=ag.wlogpi)
    lossfn = add_ce_loss(lossobj, celossfn, ag)
    return lossfn

def da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt, discr_src = None):
    celossfn = get_ce_or_bce_loss(discr, dim_y, ag.reduction, mode='da')[1]
    lossobj = frame.get_lossfn(ag.n_mc_q, ag.reduction, "defl", weight_da=ag.wda/ag.wgen, wlogpi=ag.wlogpi/ag.wgen)
    lossfn = add_ce_loss(lossobj, celossfn, ag)
    domdisc, dalossobj = None, None

    return lossfn, domdisc, dalossobj

def evaluate_llhx(frame, input_loader, n_marg_llh, use_q_llh, mode, device):
    avgr = Averager()
    for x, y in input_loader:
        x = x.to(device)
        avgr.update(frame.llh(x, None, n_marg_llh, use_q_llh, mode), nrep = len(x))
    return avgr.avg

def process_continue_run(ag):
    # Process if continue running
    if ag.init_model not in {"rand", "fix"}: # continue running
        ckpt = load_ckpt(ag.init_model, loadmodel=False)
        if ag.mode != ckpt['mode']: raise RuntimeError("mode not match")
        for k in vars(ag):
            if k not in {"testdoms", "n_epk", "gpu", "deploy", "init_model"}: # use the new final number of epochs
                setattr(ag, k, ckpt[k])
        ag.testdoms = [ckpt['testdoms']] # overwrite the input `testdoms`
    else: ckpt = None
    return ag, ckpt

class InferenceCollector:
    def __init__(self):
        # Initialize empty lists to store tensors
        self.xs = []
        self.ys = []
        self.ts = []
        self.envs = []
        self.label_t = []
        self.label_e = []
        self.sample_idx = []
        self.c_hat = []
        self.s_hat = []
        self.x_hat = []
        self.y_hat = []
        self.e_hat = []
        self.label_subT = []

    def collect_batch(self, data_bat, c_hat, s_hat, x_hat, y_hat, e_hat):
        """Collect tensors from a batch"""
        # Move tensors to CPU and convert to numpy
        self.xs.append(data_bat['X'].cpu().numpy())
        self.ys.append(data_bat['Y'].cpu().numpy())
        self.ts.append(data_bat['T'].cpu().numpy())
        self.envs.append(data_bat['E'].cpu().numpy())
        self.label_t.append(data_bat['label_T'].cpu().numpy())
        self.label_e.append(data_bat['label_E'].cpu().numpy())
        self.sample_idx.append(data_bat['index'].cpu().numpy())

        if 'label_subT' in data_bat.keys():
            self.label_subT.append(data_bat['label_subT'].cpu().numpy())

        self.c_hat.append(c_hat.cpu().numpy())
        self.s_hat.append(s_hat.cpu().numpy())
        self.x_hat.append(x_hat.cpu().numpy())
        self.y_hat.append(y_hat.cpu().numpy())
        self.e_hat.append(e_hat.cpu().numpy())

    def to_dataframe(self):
        """Convert collected data to pandas DataFrame"""
        # Concatenate all batches
        xs = np.concatenate(self.xs, axis=0)
        ys = np.concatenate(self.ys, axis=0)
        ts = np.concatenate(self.ts, axis=0)
        envs = np.concatenate(self.envs, axis=0)
        label_t = np.concatenate(self.label_t, axis=0)
        label_e = np.concatenate(self.label_e, axis=0)
        sample_idx = np.concatenate(self.sample_idx, axis=0)
        c_hat = np.concatenate(self.c_hat, axis=0)
        s_hat = np.concatenate(self.s_hat, axis=0)
        x_hat = np.concatenate(self.x_hat, axis=0)
        y_hat = np.concatenate(self.y_hat, axis=0)
        e_hat = np.concatenate(self.e_hat, axis=0)

        if len(self.label_subT) != 0:
            label_subT = np.concatenate(self.label_subT, axis=0)
            df = pd.DataFrame({
                'sample_idx': sample_idx,
                'X': xs.tolist(),
                'Y': ys.tolist(),
                'T': ts.tolist(),
                'E': envs.tolist(),
                'Index_T': label_t,
                'SubTask': label_subT.tolist(),
                'Index_E': label_e,
                'C_hat': c_hat.tolist(),
                'S_hat': s_hat.tolist(),
                'X_hat': x_hat.tolist(),
                'Y_hat': y_hat.tolist(),
                'E_hat': e_hat.tolist(),
            })
            return df
        
        df = pd.DataFrame({
            'X': xs.tolist(),
            'Y': ys.tolist(),
            'T': ts.tolist(),
            'E': envs.tolist(),
            'Index_T': label_t,
            'Index_E': label_e,
            'sample_idx': sample_idx,
            'C_hat': c_hat.tolist(),
            'S_hat': s_hat.tolist(),
            'X_hat': x_hat.tolist(),
            'Y_hat': y_hat.tolist(),
            'E_hat': e_hat.tolist()
        })

        return df
    
def inference_variables(lossfn_eval, frame, discr, gen, data_loader, device, phase='val', ta_data_loader=None, IS_OOD=True, n_mc=0,
                        sample_cs_draw=False, gen_probs=True):
    total_loss, total_x_recon_loss, total_y_recon_loss, total_e_recon_loss = 0, 0, 0, 0
    total_cossim_x, total_cossim_y, total_cossim_e = 0, 0, 0
    collector = InferenceCollector()

    for i_bat, data_bat in enumerate(data_loader if IS_OOD else zip_longer(data_loader, ta_data_loader), start=1):
        if IS_OOD:
            xs = data_bat['X'].to(device, dtype=tc.float32)
            ys = data_bat['Y'].to(device, dtype=tc.float32)
            ts = data_bat['T'].to(device, dtype=tc.float32)
            envs = data_bat['E'].to(device, dtype=tc.float32)                
            data_args = (xs, ts, envs, ys)
        else:
            xs = data_bat[0]['X'].to(device, dtype=tc.float32)
            ys = data_bat[0]['Y'].to(device, dtype=tc.float32)
            ts = data_bat[0]['T'].to(device, dtype=tc.float32)
            envs = data_bat[0]['E'].to(device, dtype=tc.float32)

            xts = data_bat[1]['X'].to(device, dtype=tc.float32)
            yts = data_bat[1]['Y'].to(device, dtype=tc.float32)
            tts = data_bat[1]['T'].to(device, dtype=tc.float32)
            envts = data_bat[1]['E'].to(device, dtype=tc.float32)
            data_args = (xs, ts, envs, ys, xts, tts, envts, yts)

        if phase != 'tr':
            total_loss += lossfn_eval(*data_args, 0)
        else:
            total_loss = 0

        if phase == 'test':
            if n_mc == 0:
                with tc.no_grad():
                    if sample_cs_draw:
                        cs_samples = frame.qt_cs1x.draw((1,), {'x':xs, 't':ts, 'e':envs})
                        c_hat = cs_samples['c'].squeeze(0)
                        s_hat = cs_samples['s'].squeeze(0)
                    else:
                        c_hat = frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                        s_hat = frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']

                    if gen_probs:
                        x_hat = frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                        y_hat = frame.p_y1c.mean({'c':c_hat})['y']
                        e_hat = frame.p_e1s.mean({'s':s_hat})['e']
                    else:
                        x_hat, e_hat = gen(c_hat, s_hat)
                        y_hat = discr(xs, ts, envs)

            else:
                cs_hat, x_hat, y_hat, e_hat = frame.inference(xs, ts, envs, shape_mc=(n_mc,))

                c_hat = cs_hat['c'].mean(0)
                s_hat = cs_hat['s'].mean(0)
                x_hat = x_hat.mean(0)
                y_hat = y_hat.mean(0)
                e_hat = e_hat.mean(0)

            collector.collect_batch(data_bat, c_hat, s_hat, x_hat, y_hat, e_hat)
        else:
            if n_mc == 0:
                with tc.no_grad():
                    if sample_cs_draw:
                        cs_samples = frame.qt_cs1x.draw((1,), {'x':xs, 't':ts, 'e':envs})
                        c_hat = cs_samples['c'].squeeze(0)
                        s_hat = cs_samples['s'].squeeze(0)
                    else:
                        c_hat = frame.qt_c1x.mean({'x':xs, 't':ts, 'e':envs})['c']
                        s_hat = frame.qt_s1x.mean({'x':xs, 't':ts, 'e':envs, 'c':c_hat})['s']
                    
                    if gen_probs:
                        x_hat = frame.p_x1cs.mean({'c':c_hat, 's':s_hat})['x']
                        y_hat = frame.p_y1c.mean({'c':c_hat})['y']
                        e_hat = frame.p_e1s.mean({'s':s_hat})['e']
                    else:
                        x_hat, e_hat = gen(c_hat, s_hat)
                        y_hat = discr(xs, ts, envs)
            else:
                _, x_hat, y_hat, e_hat = frame.inference(xs, ts, envs, shape_mc=(n_mc,))
                x_hat = x_hat.mean(0)
                y_hat = y_hat.mean(0)
                e_hat = e_hat.mean(0)

        total_x_recon_loss += F.mse_loss(x_hat, xs, reduction='none').mean(dim=1).sum()
        total_y_recon_loss += F.mse_loss(y_hat, ys, reduction='none').mean(dim=1).sum()
        total_e_recon_loss += F.mse_loss(e_hat, envs, reduction='none').mean(dim=1).sum()
        total_cossim_x += compute_cossim(x_hat.detach().cpu().numpy(), xs.detach().cpu().numpy())
        total_cossim_y += compute_cossim(y_hat.detach().cpu().numpy(), ys.detach().cpu().numpy())
        total_cossim_e += compute_cossim(e_hat.detach().cpu().numpy(), envs.detach().cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    avg_x_recon_loss = total_x_recon_loss / len(data_loader.dataset)
    avg_y_recon_loss = total_y_recon_loss / len(data_loader.dataset)
    avg_e_recon_loss = total_e_recon_loss / len(data_loader.dataset)
    
    avg_cossim_x = total_cossim_x / len(data_loader)
    avg_cossim_y = total_cossim_y / len(data_loader)
    avg_cossim_e = total_cossim_e / len(data_loader)

    if phase == 'test':
        results_df = collector.to_dataframe()
        return avg_loss, avg_x_recon_loss, avg_y_recon_loss, avg_e_recon_loss, avg_cossim_x, avg_cossim_y, avg_cossim_e, results_df
    return avg_loss, avg_x_recon_loss, avg_y_recon_loss, avg_e_recon_loss, avg_cossim_x, avg_cossim_y, avg_cossim_e

def main(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader, val_src_loader,
        ls_ts_tgt_loader = None, # for ood
        tr_tgt_loader = None,
        ts_tgt_loader = None,
        testdoms = None, # for da
        IS_OOD = True # for ood
    ):
    print(ag)
    print_infrstru_info()
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Datasets
    dim_x = tc.tensor(shape_x).prod().item()
    if IS_OOD: n_per_epk = len(tr_src_loader)
    else: n_per_epk = max(len(tr_src_loader), len(tr_tgt_loader))
    
    # Models
    res = get_models(archtype, edic(locals()) | vars(ag), ckpt, device)
    discr, gen, frame = res
    discr.train()
    gen.train()

    if IS_OOD:
        lossfn = ood_methods(discr, frame, ag, dim_y)
        domdisc = None
    else:
        lossfn, domdisc, dalossobj = da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt, None)

    # Optimizer
    pgc = ParamGroupsCollector(ag.lr)
    pgc.collect_params(discr)
    if gen is not None: pgc.collect_params(gen, frame)
    if domdisc is not None: pgc.collect_params(domdisc)

    opt = getattr(tc.optim, ag.optim)(pgc.param_groups, weight_decay=ag.wl2)
    auto_load(locals(), 'opt', ckpt)

    epk0 = 1; i_bat0 = 1
    if ckpt is not None:
        epk0 = ckpt['epochs'][-1] + 1 if ckpt['epochs'] else 1
        i_bat0 = ckpt['i_bat']

    res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)
    
    print(f"Run in mode '{ag.mode}' for {ag.n_epk:3d} epochs:")

    early_stopping = EarlyStopping(
        patience=ag.patience if hasattr(ag, 'patience') else 10,
        verbose=True
    )

    _filename = None
    for epk in range(epk0, ag.n_epk+1):
        epk_loss = 0
        pbar = tqdm.tqdm(total=n_per_epk, desc=f"Train epoch = {epk:3d}", ncols=80, leave=False)
        for i_bat, data_bat in enumerate(tr_src_loader if IS_OOD else zip_longer(tr_src_loader, tr_tgt_loader), start=1):
            if i_bat < i_bat0: continue
            if IS_OOD:
                xs = data_bat['X'].to(device, dtype=tc.float32)
                ys = data_bat['Y'].to(device, dtype=tc.float32)
                ts = data_bat['T'].to(device, dtype=tc.float32)
                envs = data_bat['E'].to(device, dtype=tc.float32)                
                data_args = (xs, ts, envs, ys)
            else:
                xs = data_bat[0]['X'].to(device, dtype=tc.float32)
                ys = data_bat[0]['Y'].to(device, dtype=tc.float32)
                ts = data_bat[0]['T'].to(device, dtype=tc.float32)
                envs = data_bat[0]['E'].to(device, dtype=tc.float32)

                xts = data_bat[1]['X'].to(device, dtype=tc.float32)
                yts = data_bat[1]['Y'].to(device, dtype=tc.float32)
                tts = data_bat[1]['T'].to(device, dtype=tc.float32)
                envts = data_bat[1]['E'].to(device, dtype=tc.float32)
                data_args = (xs, ts, envs, ys, xts, tts, envts, yts)

            opt.zero_grad()

            n_iter_tot = (epk-1)*n_per_epk + i_bat-1
            loss = lossfn(*data_args, n_iter_tot)

            if tc.isnan(loss):
                print("\n Loss is NaN, stopping training")
                return
            
            loss.backward()
            opt.step()
            pbar.update(1)

            epk_loss += loss.item()

        pbar.close()
        i_bat = 1; i_bat0 = 1

        if epk % ag.eval_interval == 0:
            res.update(epk=epk, loss=loss.item())
            epk_loss /= len(tr_src_loader)
            print(f"Mode '{ag.mode}': Epoch {epk}, Tr Loss = {epk_loss:.3f},")

            with tc.no_grad():
                discr.eval(); gen.eval()
                
                if IS_OOD:
                    lossfn_eval = ood_methods(discr, frame, ag, dim_y)
                # else:
                #     lossfn_eval, _, _ = da_methods(discr, frame, ag, dim_x, dim_y, device, ckpt, None)

                if IS_OOD:
                    loss_eval, avg_x_recon_loss, avg_y_recon_loss, avg_e_recon_loss, val_avg_cossim_x, val_avg_cossim_y, val_avg_cossim_e = inference_variables(lossfn_eval, frame, discr, gen, val_src_loader, device, sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs)
                    
                    loss_eval_ts, avg_x_recon_loss_ts, avg_y_recon_loss_ts, avg_e_recon_loss_ts, test_avg_cossim_x_ts, test_avg_cossim_y_ts, test_avg_cossim_e_ts, _ = inference_variables(lossfn_eval, frame, discr, gen, ls_ts_tgt_loader, device, phase='test', sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs)
                # else:
                #     loss_eval = inference_variables(lossfn_eval, frame, val_src_loader, device, ta_data_loader=ls_ts_tgt_loader, IS_OOD=IS_OOD)

                print(f"Mode '{ag.mode}': Epoch {epk}, Val Loss = {loss_eval:.3f}")

                wandb.log({
                    'epoch': epk,
                    'Tr_Loss': epk_loss,
                    'Val_Loss': loss_eval.item(),
                    'X_Recon_Loss': avg_x_recon_loss.item(),
                    'Y_Recon_Loss': avg_y_recon_loss.item(),
                    'E_Recon_Loss': avg_e_recon_loss.item(),
                    'Val_Cossim_X': val_avg_cossim_x,
                    'Val_Cossim_Y': val_avg_cossim_y,
                    'Val_Cossim_E': val_avg_cossim_e,
                    'Test_Loss': loss_eval_ts.item(),
                    'X_Recon_Loss_ts': avg_x_recon_loss_ts.item(),
                    'Y_Recon_Loss_ts': avg_y_recon_loss_ts.item(),
                    'E_Recon_Loss_ts': avg_e_recon_loss_ts.item(),
                    'Test_Cossim_X': test_avg_cossim_x_ts,
                    'Test_Cossim_Y': test_avg_cossim_y_ts,
                    'Test_Cossim_E': test_avg_cossim_e_ts,
                })

                if early_stopping(loss_eval):
                    print("Early stopping triggered")
                    if not ag.no_save:
                        _filename = save_ckpt(ag, res, ckpt, edic(locals()) | vars(ag), 
                                               discr, gen, frame, opt, IS_OOD=IS_OOD, filename=_filename)
                    break

            discr.train(); gen.train()
    
    print("Training finished")
    print("Validation loss did not saturated till the end of training")
    if not ag.no_save:
        _filename = save_ckpt(ag, res, ckpt, edic(locals()) | vars(ag), 
                              discr, gen, frame, opt, IS_OOD=IS_OOD, filename=_filename)

def main_depoly(ag, ckpt, archtype, shape_x, dim_y,
        tr_src_loader,
        ls_ts_tgt_loader = None, # for ood
        tr_tgt_loader = None,
        ts_tgt_loader = None,
        testdoms = None, # for da
        IS_OOD = True # for ood
    ):
    print(ag)
    print_infrstru_info()
    device = tc.device("cuda:"+str(ag.gpu) if tc.cuda.is_available() else "cpu")

    # Models
    dc_vars = edic(locals()) | vars(ag) | \
        {'dim_x':ckpt['dim_x'], 'dim_y':ckpt['dim_y'],}

    res = get_models(archtype, dc_vars, ckpt, device)
    discr, gen, frame = res
    discr.eval()
    gen.eval()

    if IS_OOD:
        res = ResultsContainer(len([ag.testdoms]), frame, ag, dim_y==1, device, ckpt)
    else:
        res = ResultsContainer(None, frame, ag, dim_y==1, device, ckpt)

    with tc.no_grad():
        discr.eval(); gen.eval()

        lossfn_eval = ood_methods(discr, frame, ag, dim_y)

        loss_eval_ID, avg_x_recon_loss_id, avg_y_recon_loss_id, avg_e_recon_loss_id, avg_cossim_x_id, avg_cossim_y_id, avg_cossim_e_id, results_ID = inference_variables(lossfn_eval, frame, discr, gen, tr_src_loader, device, phase='test', sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs)
        loss_eval_OOD, avg_x_recon_loss_ood, avg_y_recon_loss_ood, avg_e_recon_loss_ood, avg_cossim_x_ood, avg_cossim_y_ood, avg_cossim_e_ood, results_OOD = inference_variables(lossfn_eval, frame, discr, gen, ls_ts_tgt_loader, device, phase='test', sample_cs_draw=ag.sample_cs_draw, gen_probs=ag.gen_probs)
        
        print(f"Mode '{ag.mode}': ID Loss = {loss_eval_ID:.3f}")
        print(f"Mode '{ag.mode}': ID X Recon Loss = {avg_x_recon_loss_id:.3f}")
        print(f"Mode '{ag.mode}': ID Y Recon Loss = {avg_y_recon_loss_id:.3f}")
        print(f"Mode '{ag.mode}': ID E Recon Loss = {avg_e_recon_loss_id:.3f}")

        print(f"Mode '{ag.mode}': OOD Loss = {loss_eval_OOD:.3f}")
        print(f"Mode '{ag.mode}': OOD X Recon Loss = {avg_x_recon_loss_ood:.3f}")
        print(f"Mode '{ag.mode}': OOD Y Recon Loss = {avg_y_recon_loss_ood:.3f}")
        print(f"Mode '{ag.mode}': OOD E Recon Loss = {avg_e_recon_loss_ood:.3f}")

        wandb.log({
            'ID_Loss': loss_eval_ID.item(),
            'OOD_Loss': loss_eval_OOD.item(),
            'ID_X_Recon_Loss': avg_x_recon_loss_id.item(),
            'OOD_X_Recon_Loss': avg_x_recon_loss_ood.item(),
            'ID_Y_Recon_Loss': avg_y_recon_loss_id.item(),
            'OOD_Y_Recon_Loss': avg_y_recon_loss_ood.item(),
            'ID_E_Recon_Loss': avg_e_recon_loss_id.item(),
            'OOD_E_Recon_Loss': avg_e_recon_loss_ood.item()
        })
    
    results_ID.to_pickle(f"./results/{ag.traindom}/results_{ckpt['filename'].split('/')[0]}_{ckpt['filename'].split('/')[-1].split(".")[0]}_ID.pkl")
    results_OOD.to_pickle(f"./results/{ag.traindom}/results_{ckpt['filename'].split('/')[0]}_{ckpt['filename'].split('/')[-1].split(".")[0]}_OOD.pkl")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", type = str, choices = ["ind", "da"])
    parser.add_argument("--deploy", type=boolstr, default=False)
    parser.add_argument("--emb_model", type=str, default='gpt')
    parser.add_argument("--elbo_only", type=boolstr, default=False)
    parser.add_argument("--debug", type=boolstr, default=False)
    parser.add_argument("--sub_task", type=boolstr, default=False)
    parser.add_argument("--is_given_c", type=boolstr, default=False)
    parser.add_argument("--sample_cs_draw", type=boolstr, default=False)
    parser.add_argument("--gen_probs", type=boolstr, default=True)

    # Data
    parser.add_argument("--tr_val_split", type = float, default = .8)
    parser.add_argument("--n_bat", type = int, default = 128)

    # Model
    parser.add_argument("--init_model", type = str, default = "rand") # or a model file name to continue running
    parser.add_argument("--discrstru", type = str, default="mgsm_gpt")
    parser.add_argument("--genstru", type = str)

    # Process
    parser.add_argument("--n_epk", type = int, default = 800)
    parser.add_argument("--eval_interval", type = int, default = 5)
    parser.add_argument("--avglast", type = int, default = 4)
    parser.add_argument("-ns", "--no_save", action = "store_true")
    parser.add_argument("--patience", type = int, default = 5)

    # Optimization
    parser.add_argument("--optim", type = str, default="Adam")
    parser.add_argument("--lr", type = float, default=1e-3)
    parser.add_argument("--wl2", type = float, default=1e-2)
    parser.add_argument("--reduction", type = str, default = "mean")
    parser.add_argument("--momentum", type = float, default = 0.) # only when "lr" is "SGD"
    parser.add_argument("--nesterov", type = boolstr, default = False) # only when "lr" is "SGD"
    parser.add_argument("--lr_expo", type = float, default = .75) # only when "lr" is "SGD"
    parser.add_argument("--lr_wdatum", type = float, default = 6.25e-6) # only when "lr" is "SGD"

    # For generative models only
    parser.add_argument("--sig_c", type = float, default = 1.)
    parser.add_argument("--sig_s", type = float, default = 1.)
    parser.add_argument("--corr_cs", type = float, default = 0.5)
    parser.add_argument("--pstd_y", type = float, default = 3e-2)
    parser.add_argument("--qstd_c", type = float, default = -1.)
    parser.add_argument("--qstd_s", type = float, default = -1.) # for svgm only
    parser.add_argument("--wgen", type = float, default = 1.)
    parser.add_argument("--wsup", type = float, default = 1.)
    parser.add_argument("--wsup_expo", type = float, default = 0.75) # only when "wsup" is not 0
    parser.add_argument("--wsup_wdatum", type = float, default = 6.25e-6) # only when "wsup" and "wsup_expo" are not 0
    
    parser.add_argument("--wlogpi", type = float, default = None)
    parser.add_argument("--wrecon_x", type = float, default = 1.)
    parser.add_argument("--wrecon_e", type = float, default = 1.)
    parser.add_argument("--wrecon_c", type = float, default = 1.)
    parser.add_argument("--wrecon_s", type = float, default = 1.)

    parser.add_argument("--n_mc_q", type = int, default = 0)
    parser.add_argument("--eval_llh", action = "store_true")
    parser.add_argument("--use_q_llh", type = boolstr, default = True)
    parser.add_argument("--n_marg_llh", type = int, default = 16)
    parser.add_argument("--true_sup", type = boolstr, default = False, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")
    parser.add_argument("--true_sup_val", type = boolstr, default = True, help = "for 'svgm-ind', 'svgm-da', 'svae-da' only")

    parser.add_argument("--ind_cs", type = boolstr, default = False)
    ## For OOD
    parser.add_argument("--mvn_prior", type = boolstr, default = False)
    ## For DA
    parser.add_argument("--src_mvn_prior", type = boolstr, default = False)
    parser.add_argument("--tgt_mvn_prior", type = boolstr, default = False)

    # For OOD
    ## For cnbb only
    parser.add_argument("--reg_w", type = float, default = 1e-4)
    parser.add_argument("--reg_s", type = float, default = 3e-6)
    parser.add_argument("--lr_w", type = float, default = 1e-3)
    parser.add_argument("--n_iter_w", type = int, default = 4)

    # For DA
    parser.add_argument("--wda", type = float, default = .25)
    
    parser.add_argument("--gpu", type=int, default = 0)
    parser.add_argument("--data_root", type = str, default = "../data/synthetic_dataset/")
    parser.add_argument("--traindom", type = str) # 12665 = 5923 (46.77%) + 6742
    parser.add_argument("--testdoms", type = str) # 2115 = 980 (46.34%) + 1135
    parser.add_argument("--shuffle", type = boolstr, default = True)

    parser.add_argument("--mlpstrufile", type = str, default = "./arch/mlpstru.json")
    parser.add_argument("--actv", type = str, default = "ReLU")
    parser.add_argument("--after_actv", type = boolstr, default = True)

    parser.add_argument("--cond_prior", type = boolstr, default = False)
    parser.add_argument("--is_corr_cs", type = boolstr, default = False)
    parser.add_argument("--actv_prior", type = str, default = "lrelu")
    parser.add_argument("--mu_c", type = float, default = 0.1)
    parser.add_argument("--mu_s", type = float, default = 0.1)
    parser.add_argument("--pstd_x", type = float, default = 3e-1)
    parser.add_argument("--pstd_t", type = float, default = 3e-1)
    parser.add_argument("--pstd_e", type = float, default = 3e-1)

    # parser.set_defaults(genstru = None,
    #         optim = "Adam",
    #         # momentum = 0., nesterov = False, lr_expo = .5, lr_wdatum = 6.25e-6, # only when "lr" is "SGD"
    #         # wda = 1.,
    #     )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    ag = parser.parse_args()
    if ag.wlogpi is None: ag.wlogpi = ag.wgen
    ag, ckpt = process_continue_run(ag)

    if ckpt is not None: testdoms = ag.testdoms[0]
    else: testdoms = ag.testdoms
    
    IS_OOD = ag.mode=='ind'
    archtype = "mlp"

    set_seed_all(ag.seed)

    # Dataset
    df_ID = pd.read_pickle(f"../data/{ag.traindom}/embs/dataset_{ag.emb_model}_emb_ID.pkl")
    
    if not IS_OOD:
        df_ta = pd.read_pickle(f"../data/{ag.traindom}/embs/dataset_{ag.emb_model}_emb_OOD_tr_ta.pkl")
        df_OOD = pd.read_pickle(f"../data/{ag.traindom}/embs/dataset_{ag.emb_model}_emb_OOD_te_ta.pkl")
    else:
        df_OOD = pd.read_pickle(f"../data/{ag.traindom}/embs/dataset_{ag.emb_model}_emb_OOD.pkl")

    if ag.traindom == 'mgsm':
        if ag.emb_model.split('_')[-1] == 'v2':
            dataset = HighDimSCMRealWorldDataset(df_ID, subtask=True, use_subtask=ag.sub_task, use_y_long=True)
        else:
            dataset = HighDimSCMRealWorldDataset(df_ID, subtask=True, use_subtask=ag.sub_task)
        dataset_OOD = HighDimSCMRealWorldDataset(df_OOD, subtask=True, use_subtask=ag.sub_task)
        if not IS_OOD:
            dataset_ta = HighDimSCMRealWorldDataset(df_ta, subtask=True, use_subtask=ag.sub_task)
    else:
        dataset = HighDimSCMRealWorldDataset(df_ID)
        dataset_OOD = HighDimSCMRealWorldDataset(df_OOD)

        if not IS_OOD:
            dataset_ta = HighDimSCMRealWorldDataset(df_ta)

    sample = dataset[0]
    shape_x = sample['X'].shape
    dim_y = sample['Y'].shape[-1]
    dim_t = sample['T'].shape[-1]

    print(f"Input X shape: {shape_x}")
    setattr(ag, 'dim_var', shape_x[-1])
    setattr(ag, 'dim_t', dim_t)
    setattr(ag, 'discrstru', f'{ag.traindom}_{ag.emb_model.split("_")[0]}')
    print(ag.discrstru)

    if ag.deploy:
        tr_src_loader = DataLoader(dataset, batch_size=ag.n_bat, shuffle=True, drop_last=False)
        ls_ts_tgt_loader = DataLoader(dataset_OOD, batch_size=ag.n_bat, shuffle=False)
        
        IS_OOD = True
        tr_tgt_loader = None
        setattr(ag, 'mode', 'ind')
    else:
        total_len = len(dataset)
        train_len = int(total_len * 0.7)
        val_len = total_len - train_len

        train_dataset, val_dataset = random_split(dataset, [train_len, val_len])

        tr_src_loader = DataLoader(train_dataset, batch_size=ag.n_bat, shuffle=True, drop_last=False)
        val_src_loader = DataLoader(val_dataset, batch_size=ag.n_bat, shuffle=True)
        ls_ts_tgt_loader = DataLoader(dataset_OOD, batch_size=ag.n_bat, shuffle=False)

        tr_tgt_loader = None
        if not IS_OOD:
            tr_tgt_loader = DataLoader(dataset_ta, batch_size=ag.n_bat, shuffle=True, drop_last=False)

    with tempfile.TemporaryDirectory() as tmp_dir:
        wandb.init(
                project=f"CCL",
                config=ag,
                save_code=True,
                dir=tmp_dir
            )
        
        if ag.deploy:
            main_depoly(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, ls_ts_tgt_loader, tr_tgt_loader=tr_tgt_loader, IS_OOD=IS_OOD)
        else:
            main(ag, ckpt, archtype, shape_x, dim_y, tr_src_loader, val_src_loader, ls_ts_tgt_loader, tr_tgt_loader=tr_tgt_loader, IS_OOD=IS_OOD)      
        wandb.finish()
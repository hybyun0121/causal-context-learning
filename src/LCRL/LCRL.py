import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

import numpy as np
from .modules import q_z_vall, q_t_xy, q_e_x, q_y_xt, p_x_ze, p_y_z

class LatentCausalRepresentationLearner(nn.Module):
    def __init__(self, model_config: dict,
                 dim_var: int, dim_projection: int,
                 projection_grad: bool=False,
                 is_deployment_mode=False):
        super().__init__()
        self.model_config = model_config
        self.is_deployment_mode = is_deployment_mode

        # Inference / Encoder
        self.q_z_vall = q_z_vall(dim_in=model_config['q_z']['dim_in'],
                                 dim_out=model_config['q_z']['dim_out'],
                                 dim_hidden=model_config['q_z']['dim_hidden'])
        
        # Auxiliary distribution
        self.q_t_xy = q_t_xy(dim_in=model_config['q_t']['dim_in'],
                             dim_out=model_config['q_t']['dim_out'],
                             dim_hidden=model_config['q_t']['dim_hidden'])
        
        self.q_e_x = q_e_x(dim_in=model_config['q_e']['dim_in'],
                           dim_out=model_config['q_e']['dim_out'],
                           dim_hidden=model_config['q_e']['dim_hidden'])

        self.q_y_xt = q_y_xt(dim_in=model_config['q_y']['dim_in'],
                               dim_out=model_config['q_y']['dim_out'],
                               dim_hidden=model_config['q_y']['dim_hidden'])
        
        # Decoder
        self.p_x_ze = p_x_ze(dim_in=model_config['p_x']['dim_in'],
                             dim_out=model_config['p_x']['dim_out'],
                             dim_hidden=model_config['p_x']['dim_hidden'])
        
        self.p_y_z = p_y_z(dim_in=model_config['p_y']['dim_in'],
                            dim_out=model_config['p_y']['dim_out'],
                            dim_hidden=model_config['p_y']['dim_hidden'])
        
        # Linear projection
        if projection_grad:
            self.lin_pro_x = nn.Linear(dim_var, dim_projection)
            self.lin_pro_e = nn.Linear(dim_var, dim_projection)
            self.lin_pro_t = nn.Linear(dim_var, dim_projection)
            self.lin_pro_y = nn.Linear(dim_var, dim_projection)
        else:
            self.lin_pro_x = nn.Linear(dim_var, dim_projection).requires_grad_(False)
            self.lin_pro_e = nn.Linear(dim_var, dim_projection).requires_grad_(False)
            self.lin_pro_t = nn.Linear(dim_var, dim_projection).requires_grad_(False)
            self.lin_pro_y = nn.Linear(dim_var, dim_projection).requires_grad_(False)
        
    def kl_divergence(self, q):
        p = dist.normal.Normal(torch.zeros_like(q.loc), torch.ones_like(q.scale))
        KLD = dist.kl_divergence(q, p).sum(dim=-1)
        return KLD
    
    def step(self, x, y, t, e, aux_dist=True, use_aux=True):
        x_pro = self.lin_pro_x(x)
        t_pro = self.lin_pro_t(t)
        e_pro = self.lin_pro_e(e)
        xt = torch.concat([x_pro, t_pro], dim=-1)
        y_sample = self.q_y_xt(xt, distributional=aux_dist)
        y_pro = self.lin_pro_y(y_sample)

        xy = torch.concat([x_pro, y_pro], dim=-1)
        t_sample = self.q_t_xy(xy, distributional=aux_dist)
        e_sample = self.q_e_x(x_pro, distributional=aux_dist)
        
        if use_aux:
            xyte = torch.concat([x_pro, y_pro, self.lin_pro_t(t_sample), self.lin_pro_e(e_sample)], dim=-1)
        else:
            xyte = torch.concat([x_pro, y_pro, t_pro, e_pro], dim=-1)
        z_sample, q_z = self.q_z_vall(xyte)

        # KL-divergence
        kl_loss = self.kl_divergence(q_z)

        # Decoder
        if use_aux:
            ze = torch.concat([z_sample, self.lin_pro_e(e_sample)], dim=-1)
        else:
            ze = torch.concat([z_sample, e_pro], dim=-1)
        x_gen = self.p_x_ze(ze)
        y_gen = self.p_y_z(z_sample)

        # Reconstruction loss
        recon_x_loss = F.mse_loss(x_gen, x, reduction='none').mean(1)
        recon_y_loss = F.mse_loss(y_gen, y, reduction='none').mean(1)

        # Auxiliary loss
        aux_t_loss = F.mse_loss(t_sample, t, reduction='none').mean(1)
        aux_e_loss = F.mse_loss(e_sample, e, reduction='none').mean(1)
        aux_y_loss = F.mse_loss(y_sample, y, reduction='none').mean(1)

        loss = {
            "recon_x_loss":recon_x_loss,
            "recon_y_loss":recon_y_loss,
            "aux_y_loss":aux_y_loss,
            "aux_t_loss":aux_t_loss,
            "aux_e_loss":aux_e_loss,
            "kl_loss":kl_loss
        }
        return loss, z_sample, q_z

    def inference(self, x, t, e, aux_dist=True, use_aux=True, deployment_mode=True):
        x_pro = self.lin_pro_x(x)
        t_pro = self.lin_pro_t(t)
        e_pro = self.lin_pro_e(e)
        xt = torch.concat([x_pro, t_pro], dim=-1)
        y_sample = self.q_y_xt(xt, distributional=aux_dist)
        y_pro = self.lin_pro_y(y_sample)

        xy = torch.concat([x_pro, y_pro], dim=-1)
        t_sample = self.q_t_xy(xy, distributional=aux_dist)
        e_sample = self.q_e_x(x_pro, distributional=aux_dist)
        
        if use_aux:
            xyte = torch.concat([x_pro, y_pro, self.lin_pro_t(t_sample), self.lin_pro_e(e_sample)], dim=-1)
        else:
            xyte = torch.concat([x_pro, y_pro, t_pro, e_pro], dim=-1)
        z_sample, q_z = self.q_z_vall(xyte)

        # Decoder
        if use_aux:
            ze = torch.concat([z_sample, self.lin_pro_e(e_sample)], dim=-1)
        else:
            ze = torch.concat([z_sample, e_pro], dim=-1)
        x_gen = self.p_x_ze(ze)
        y_gen = self.p_y_z(z_sample)

        if deployment_mode:
            return z_sample, q_z, x_gen, t_sample, e_sample
        else:
            return z_sample, q_z, x_gen, t_sample, e_sample, y_sample, y_gen

def get_model_config(config):
    dim_vall = config.dim_projection*4
    dim_xt = config.dim_projection*2
    dim_xy = config.dim_projection*2
    dim_ze = config.dim_projection+config.dim_z

    return {
        'q_z':{
            'dim_in': dim_vall,
            'dim_hidden':config.hidden_q_z,
            'dim_out': config.dim_z
        },
        'q_y':{
            'dim_in': dim_xt,
            'dim_hidden': config.hidden_q_y,
            'dim_out': config.dim_var
        },
        'q_t':{
            'dim_in': dim_xy,
            'dim_hidden': config.hidden_q_t,
            'dim_out': config.dim_var
        },
        'q_e':{
            'dim_in': config.dim_projection,
            'dim_hidden': config.hidden_q_e,
            'dim_out': config.dim_var
        },
        'p_x':{
            'dim_in': dim_ze,
            'dim_hidden': config.hidden_p_x,
            'dim_out': config.dim_var
        },
        'p_y':{
            'dim_in': config.dim_z,
            'dim_hidden': config.hidden_p_y,
            'dim_out': config.dim_var
        }
    }

'''
def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        K.add_(torch.exp(D.mul(-gamma)))

        return K

    def mmd_compute(self, x, y, kernel_type='gaussian', gamma=1.0):
        if kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x, gamma).mean(1)
            Kyy = self.gaussian_kernel(y, y, gamma).mean(1)
            Kxy = self.gaussian_kernel(x, y, gamma).mean(1)
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean(1)
            cova_diff = (cova_x - cova_y).pow(2).mean(1)

            return mean_diff + cova_diff
        
    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1 / (1 + np.exp(-k * (step - x0))))
        elif anneal_function == 'linear':
            return min(1, step / x0)

    def compute_hsic(self, x, y, kernel='rbf'): 
'''
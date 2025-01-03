import os
import math
import torch
import random
import argparse
import numpy as np
from torch import optim
from sklearn.metrics import pairwise

def set_seed_all(seed):
    from transformers import set_seed
    set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    try:
        torch.set_deterministic_debug_mode(True)
    except AttributeError:
        pass

def get_optimizer(para, opt, lr, wd):
    if opt=='adam':
        return optim.Adam(para, lr=lr, weight_decay=wd)
    if opt=='adamw':
        return optim.AdamW(para, lr=lr, weight_decay=wd)
    if opt=='adamax':
        return optim.Adamax(para, lr=lr, weight_decay=wd)
    if opt=='adagrad':
        return optim.Adagrad(para, lr=lr, weight_decay=wd)

def format_scientific_fixed_exp(number, exp=8):
    if number == 0:
        return f"0.{'0' * (exp-1)}e+{exp}"
    
    log = math.log10(abs(number))
    diff = exp - int(log)
    adjusted_num = number * (10 ** diff)
    
    return f"{adjusted_num:.{exp}f}e+{exp}"

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str_to_list(arg):
    return [int(x) for x in arg.strip("[]").split(',')]

def str_to_list_float(arg):
    return [float(x) for x in arg.strip("[]").split(',')]

def lambda_lr(epoch, warmup_epoch=10, off_warmup=False):
    if not off_warmup:
        if epoch < warmup_epoch:
            return float(epoch) / float(warmup_epoch)
        else:
            lr = max(0.98 ** (epoch - warmup_epoch), 1e-4)
            return lr
    else:
        return max(0.98 ** epoch, 1e-6)
    
def get_model_meta_data(args):
#     meta_data = {
#         "seed" : args.seed,
#         "dim_h_q" : args.dim_h_q,
#         "dim_h" : args.dim_h,
#         "dim_h_t" : args.dim_h_t,
#         "dim_h_aux" : args.dim_h_aux,
#         "dim_emb" : args.dim_emb,
#         "dim_z" : args.dim_latent,
#         "input_mode" : args.input_mode,
#         "nh" : args.num_hidden,
#         "cond_vars" : args.cond_vars,
#         "use_mmd_reg" : args.use_reg,
#         "device" : args.device,
#         "gamma_ze" :args.gamma_ze,
#         "do_recon_t":args.do_recon_t
#     }
#     return meta_data
    return vars(args)
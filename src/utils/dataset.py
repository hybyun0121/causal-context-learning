import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TrainDataset(Dataset):
    def __init__(self, inputs, outputs, env, task):
        self.inputs = inputs
        self.outputs = outputs
        self.env = env
        self.task = task

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        xs = self.inputs[idx]
        ys = self.outputs[idx]
        es = self.env[idx]
        ts = self.task[idx]

        return xs, ys, es, ts

class TestDataset(Dataset):
    def __init__(self, inputs, env, task):
        self.inputs = inputs
        self.env = env
        self.task = task

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        xs = self.inputs[idx]
        es = self.env[idx]
        ts = self.task[idx]

        return xs, es, ts

class HighDimSCMSyntheticDataset(Dataset):
    def __init__(self, df, is_train=True):
        super(HighDimSCMSyntheticDataset, self)

        if is_train:
            # self.df = df[(df['Index_T'].isin([0,1,2])&(df['Index_E'].isin([0,1,2])))]
            self.df = df
            self.X = self.combine_arrays(self.df, 'X')
            self.Y = self.combine_arrays(self.df, 'Y')
            self.E = self.combine_arrays(self.df, 'E')
            self.T = self.combine_arrays(self.df, 'T')
            self.Z = self.combine_arrays(self.df, 'Xc')
            self.label_T = self.df['Index_T'].values
            self.label_E = self.df['Index_E'].values
            self.index = self.df.index.values
        else:
            self.df = df[(df['Index_T'].isin([3,4])&(df['Index_E'].isin([3,4])))]
            self.X = self.combine_arrays(self.df, 'X')
            self.Y = self.combine_arrays(self.df, 'Y')
            self.E = self.combine_arrays(self.df, 'E')
            self.T = self.combine_arrays(self.df, 'T')
            self.Z = self.combine_arrays(self.df, 'Xc')
            self.label_T = self.df['Index_T'].values
            self.label_E = self.df['Index_E'].values
            self.index = self.df.index.values

    def combine_arrays(self, df, column_name):
        arrays = df[column_name].tolist()
        first_array = arrays[0]
        result_shape = (len(arrays), *first_array.shape)
        result = np.array(arrays).reshape(result_shape)
        return result

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        e = self.E[idx]
        t = self.T[idx]
        z = self.Z[idx]
        label_t = self.label_T[idx]
        label_e = self.label_E[idx]
        index = self.index[idx]
        return {'X': x, 'Y': y, 'Z':z, 'T':t, 'E':e, 'label_T':label_t, 'label_E':label_e, 'index':index}
    
class RealWorldDataset(Dataset):
    def __init__(self, df, is_oop=False):
        super(RealWorldDataset, self)

        self.df = df
        self.X = self.combine_arrays(self.df, 'X')

        if not is_oop:
            self.Y = self.combine_arrays(self.df, 'Y')
        self.E = self.combine_arrays(self.df, 'E')
        self.T = self.combine_arrays(self.df, 'T')
        self.index = self.df.index.values

        self.is_oop = is_oop

    def combine_arrays(self, df, column_name):
        arrays = df[column_name].tolist()
        arrays = np.stack(df[column_name].tolist())
        return arrays

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if not self.is_oop:
            y = self.Y[idx]
        e = self.E[idx]
        t = self.T[idx]
        index = self.index[idx]
        if not self.is_oop:
            return {'X': x, 'Y': y, 'T':t, 'E':e, 'index':index}
        else:
            return {'X': x, 'T':t, 'E':e, 'index':index}

class RealWorldLargeDataset(Dataset):
    def __init__(self, df, X_emb=None, Y_emb=None, task2emb=None, env2emb=None, T_emb=None, E_emb=None, is_oop=False):
        super(RealWorldLargeDataset, self)

        self.df = df
        self.X = X_emb

        if Y_emb is not None:
            self.Y = Y_emb

        if not task2emb is None:
            self.task2emb = task2emb
            self.env2emb = env2emb
            self.T = df.Task.values
            self.E = df.Environment.values
        self.index = self.df.index.values

        self.T_emb = T_emb
        self.E_emb = E_emb
        self.is_oop = is_oop

    def __len__(self):
        return len(self.df)
    
    def get_task_embs(self, t):
        tmp = [self.task2emb[k] for k in t]
        return torch.concat(tmp)
    
    def get_env_embs(self, e):
        tmp = [self.env2emb[k] for k in e]
        return torch.concat(tmp)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if not self.is_oop:
            y = self.Y[idx]

        if self.T_emb is not None:
            t_emb = self.T_emb
            e_emb = self.E_emb
        else:
            t = self.T[idx]
            e = self.E[idx]
            t_emb = self.task2emb[t]
            e_emb = self.env2emb[e]
        
        index = self.index[idx]
        if not self.is_oop:
            return {'X': x, 'Y': y, 'T':t_emb, 'E':e_emb, 'index':index}
        else:
            return {'X': x, 'T':t_emb, 'E':e_emb, 'index':index}

class ICLDataset(Dataset):
    def __init__(self, df, icl_prompt, prompt_format, is_zeroshot=False):
        super(ICLDataset, self)
        self.examples = icl_prompt
        self.query = df[prompt_format['columns']['X']].values
        self.answer = df[prompt_format['columns']['Y']].values
        self.prompt_format = prompt_format

        if not is_zeroshot:
            self.input_text = [[{"role":"user",
                                "content":f"{self.examples[i]}{self.query[i]}\n{self.prompt_format['templates']['A'][0]}"}]
                                for i in range(len(self.query))]
        else:
            self.input_text = [[{"role":"user",
                                "content":f"{self.query[i]}\n{self.prompt_format['templates']['A'][0]}"}]
                                for i in range(len(self.query))]
        
    def __len__(self):
        return len(self.query)
    
    def __getitem__(self, idx):
        return self.input_text[idx], self.answer[idx]
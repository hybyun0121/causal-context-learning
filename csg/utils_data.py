import numpy as np
from torch.utils.data import Dataset

class HighDimSCMRealWorldDataset(Dataset):
    def __init__(self, df, subtask=False, use_subtask=False, use_y_long=False):
        super(HighDimSCMRealWorldDataset, self)

        self.df = df
        self.X = self.combine_arrays(self.df, 'X')

        if use_y_long:
            self.Y = self.combine_arrays(self.df, 'Y_long')
        else:
            self.Y = self.combine_arrays(self.df, 'Y')
        self.E = self.combine_arrays(self.df, 'E')

        if not use_subtask:
            self.T = self.combine_arrays(self.df, 'T')
        else:
            self.T = self.combine_arrays(self.df, 'ST')
        
        if subtask:
            self.label_subT = self.df['SubTask'].values
        self.subtask = subtask

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

        label_t = self.label_T[idx]
        label_e = self.label_E[idx]
        index = self.index[idx]

        if self.subtask:
            label_subT = self.label_subT[idx]
            return {'X': x, 'Y': y, 'T':t, 'E':e, 'label_T':label_t, 'label_E':label_e, 'index':index, 'label_subT':label_subT}
        return {'X': x, 'Y': y, 'T':t, 'E':e, 'label_T':label_t, 'label_E':label_e, 'index':index}

class HighDimSCMSyntheticDataset(Dataset):
    def __init__(self, df):
        super(HighDimSCMSyntheticDataset, self)

        self.df = df
        # self.X = self.combine_arrays(self.df, 'X')
        # self.Y = self.combine_arrays(self.df, 'Y')
        # self.E = self.combine_arrays(self.df, 'E')
        # self.T = self.combine_arrays(self.df, 'T')
        # self.C = self.combine_arrays(self.df, 'C')
        # self.S = self.combine_arrays(self.df, 'S')
        self.X = np.stack(self.df['X'].values)
        self.Y = np.stack(self.df['Y'].values)
        self.E = np.stack(self.df['E'].values)
        self.T = np.stack(self.df['T'].values)
        self.C = np.stack(self.df['C'].values)
        self.S = np.stack(self.df['S'].values)
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
        c = self.C[idx]
        s = self.S[idx]
        label_t = self.label_T[idx]
        label_e = self.label_E[idx]
        index = self.index[idx]
        return {'X': x, 'Y': y, 'C':c, 'S':s, 'T':t, 'E':e, 'label_T':label_t, 'label_E':label_e, 'index':index}
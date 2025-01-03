import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

# class naming convention: p_A_BC -> p(A|B,C)
####### Inference model / Encoder #######
# Encoder: q(z|x, y, t, e)
class q_z_vall(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        
        self.mu = nn.Linear(dim_hidden[-1], dim_out)
        self.logvar = nn.Linear(dim_hidden[-1], dim_out)
    
    def forward(self, input):
        out = F.elu(self.input_layer(input))

        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = logvar.exp().pow(0.5)
        q_z = dist.normal.Normal(mu, std)
        z = q_z.rsample()
        return z, q_z

# Auxiliary distribution
# q(t|x,y)
class q_t_xy(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        
        self.mu = nn.Linear(dim_hidden[-1], dim_out)
        self.logvar = nn.Linear(dim_hidden[-1], dim_out)
    
    def forward(self, input, distributional=True):
        # Common forward pass for both paths
        out = F.elu(self.input_layer(input))
        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        
        # Get distribution parameters
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = logvar.exp().pow(0.5)
        
        if distributional:
            # Return mean value directly for deterministic output
            return mu
        else:
            # Return sampled value from distribution
            q_t = dist.normal.Normal(mu, std)
            t = q_t.rsample()
            return t
    
# q(e|x)
class q_e_x(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        
        self.mu = nn.Linear(dim_hidden[-1], dim_out)
        self.logvar = nn.Linear(dim_hidden[-1], dim_out)
   
    def forward(self, input, distributional=True):
        out = F.elu(self.input_layer(input))
        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = logvar.exp().pow(0.5)
        if distributional:
            return mu
        else:
            q_e = dist.normal.Normal(mu, std)
            e = q_e.rsample()
            return e

# q(y|x,t,e)
class q_y_xt(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        
        self.mu = nn.Linear(dim_hidden[-1], dim_out)
        self.logvar = nn.Linear(dim_hidden[-1], dim_out)
    
    def forward(self, input, distributional=True):
        out = F.elu(self.input_layer(input))
        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        mu = self.mu(out)
        logvar = self.logvar(out)
        std = logvar.exp().pow(0.5)
        if distributional:
            return mu
        else:
            q_y = dist.normal.Normal(mu, std)
            y = q_y.rsample()
            return y

####### Generative model / Decoder / Model network #######
class p_x_ze(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        self.last_layer = nn.Linear(dim_hidden[-1], dim_out)
    
    def forward(self, input):
        out = F.elu(self.input_layer(input))

        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        out = self.last_layer(out)
        return out

class p_y_z(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: list):
        super().__init__()
        self.input_layer = nn.Linear(dim_in, dim_hidden[0])

        self.hidden = nn.ModuleList()
        if len(dim_hidden) > 1:
            for i in range(len(dim_hidden)-1):
                self.hidden.append(
                    nn.Sequential(
                        nn.Linear(dim_hidden[i], dim_hidden[i+1]),
                        nn.ELU(),
                    )
                )
        self.last_layer = nn.Linear(dim_hidden[-1], dim_out)
    
    def forward(self, input):
        out = F.elu(self.input_layer(input))

        if len(self.hidden) != 0:
            for hidden_layer in self.hidden:
                out = hidden_layer(out)
        
        out = self.last_layer(out)
        return out
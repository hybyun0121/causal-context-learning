class MLPcsy1x(MLPBase):
    def __init__(self, dim_x, dims_postx2pres, dim_s, dim_paras, dims_posts2c, dims_postc2prey, dim_y, dim_t, actv = "Sigmoid",
            std_s_val: float=-1., std_c_val: float=-1., after_actv: bool=True, ind_cs: bool=False,
            dims_vars2prev_xte=None, dims_vars2prev_xt=None, dims_prevs2c=None, dims_prevs2s=None): # if <= 0, then learn the std.
        """
        x,t,e = prev_xte =>  s     ============== \
                                                    ==> c
                            x,t    === prev_xt ===/ 
        """
        super(MLPcsy1x, self).__init__()
        if type(actv) is str: actv = getattr(nn, actv)
        self.dim_x, self.dim_s, self.dim_y = dim_x, dim_s, dim_y
        self.dim_t = dim_t
        dim_pres, dim_c = dims_vars2prev_xte[-1], dims_posts2c[-1]
        self.dim_pres, self.dim_c = dim_pres, dim_c
        self.shape_x, self.shape_s, self.shape_c = (dim_x,), (dim_s,), (dim_c,)
        self.shape_t, self.shape_e = (dim_t,), (dim_x,)
        self.dims_postx2pres, self.dim_paras, self.dims_posts2c, self.dims_postc2prey, self.actv \
                = dims_postx2pres, dim_paras, dims_posts2c, dims_postc2prey, actv
        
        self.dims_vars2prev_xte = dims_vars2prev_xte
        self.dims_vars2prev_xt = dims_vars2prev_xt
        self.ind_cs = ind_cs

        # g(x,t,e)
        self.f_xte2prev = mlp_constructor([dim_x+dim_t+dim_x] + dims_vars2prev_xte if dims_vars2prev_xte is not None else dims_postx2pres, actv, lastactv = False)
        self.f_xt2prev = mlp_constructor([dim_x+dim_t] + dims_vars2prev_xt if dims_vars2prev_xt is not None else dims_postx2pres, actv, lastactv = False)
        
        self.f_prev2s = mlp_constructor([dim_pres] + dims_prevs2s + [dim_s], actv, lastactv=True)
        self.f_prevs2c = mlp_constructor([dim_pres + dim_s] + dims_prevs2c + [dim_c], actv, lastactv=True)
  
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
        if self.learn_std_s:
            self.nn_std_s = nn.Sequential(
                    mlp_constructor(
                        [dim_pres] + dims_prevs2s + [dim_s],
                        nn.ReLU, lastactv = False),
                    nn.Softplus()
                )
            init_linear(self.nn_std_s, 0., 1e-2, 0.)
            self.f_std_s = self.nn_std_s

        if self.learn_std_c:
            _input_dim = dim_s + dim_pres

            self.nn_std_c = nn.Sequential(
                    nn.BatchNorm1d(_input_dim),
                    nn.ReLU(),
                    # nn.Dropout(0.5),
                    mlp_constructor(
                        [_input_dim] + dims_prevs2c + [dim_c],
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

    def _get_prevs_xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_prev_s))\
        and (not is_same_tensor(t, self._t_cache_prev_s))\
        and (not is_same_tensor(e, self._e_cache_prev_s)):
            self._x_cache_prev_s = x
            self._t_cache_prev_s = t
            self._e_cache_prev_s = e
            _input = tc.cat([x, t, e], dim=-1)
            prev_xte = self.f_xte2prev(_input) # g(x,t,e)
            self._prev_cache_s = self.f_prev2s(prev_xte)
        return self._prev_cache_s
    
    def _get_prevs_xt(self, x,t):
        # h(prev)
        if (not is_same_tensor(x, self._x_cache_parav))\
        and (not is_same_tensor(t, self._t_cache_c)):
            self._x_cache_parav = x
            self._t_cache_c = t
            self._parav_cache = self.f_prev2parav(self._get_prevc(x,t))
        return self._parav_cache

    def std_s1xte(self, x,t,e):
        if self.learn_std_s:
            return self.f_std_s(self._get_prevs_xte(x,t,e))
        else:
            return tensorify(x.device, self.std_s_val)[0].expand(x.shape[:-1]+(self.dim_s,))
    
    def std_c1sxt(self, x, t, e):
        if self.learn_std_c:
            s = self.s1xte(x,t,e)
            prev_xt = self._get_prevs_xt(x,t)
            return self.f_std_c(tc.cat([s, prev_xt], dim=-1))
        else:
            return tensorify(x.device, self.std_c_val)[0].expand(x.shape[:-1]+t.shape[:-1]+(self.dim_c,))
    
    def s1xte(self, x,t,e):
        if (not is_same_tensor(x, self._x_cache_s))\
        and (not is_same_tensor(t, self._t_cache_s))\
        and (not is_same_tensor(e, self._e_cache_s)):
            self._x_cache_s = x
            self._t_cache_s = t
            self._e_cache_s = e

            self._s_cache = self.f_prev2s(self._get_prevs_xte(x,t,e))
        return self._s_cache
    
    def c1sxt(self, x, t, e):
        s = self.s1xte(x,t,e)
        prev_xt = self._get_prevs_xt(x,t)
        return self.f_prevs2c(tc.cat([s, prev_xt], dim=-1))
    
    def y1c(self, c):
        '''
        q(y|c)
        '''
        return self.f_c2y(c)

    def forward(self, x,t,e):
        '''
        q(y|c) = q(y|c)q(c,s|x,t,e)
        '''
        return self.y1c(self.c1sxt(x,t,e))
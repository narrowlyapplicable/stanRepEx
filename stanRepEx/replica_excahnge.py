import numpy as np 

from concurrent.futures import ProcessPoolExecutor, as_completed

def idx_ex(n_rep, ee):
    idx = np.arange(0, np.floor(n_rep/2), dtype=np.int64) * 2
    if (ee % 2 == 1):
        return idx[:-1] + 1
    else:
        return idx

def cast_staninit(init):
    try:
        return np.asscalar(init)
    except ValueError:
        return init

def get_par_length(par_list):
    par_length = []
    for item in par_list.values():
        try:
            par_length += [len(item)]
        except TypeError:
            par_length += [1]
    return np.array(par_length)

def replica_exMCMC(inv_T, n_ex, *, stanmodel, data, par_list, init, n_iter, warmup):
    par_length = get_par_length(par_list)

    n_rep = len(inv_T)
    len_mcmc = n_iter - warmup
    n_param = np.sum(par_length) + 1 # n_parameters includes lp__.
    ms_T1 = np.zeros((len_mcmc*n_ex, n_param)) # <- MCMC samples at inv_T=1.

    idx_tbl = np.zeros((n_ex, n_rep), dtype=np.int64) # index table of (exchange time, replica)
    E_tbl = np.zeros((n_ex, n_rep)) # E table
    init_list = [init] * n_rep

    for ee in range(n_ex):
        fit_list = []
        for r in range(n_rep):
            data["beta"] = inv_T[r]
            fit_list.append(stanmodel.sampling(data=data, init=init_list[r], iter=n_iter, warmup=warmup, chains=1, seed=r, check_hmc_diagnostics=False))
        ms_T1[ee * len_mcmc : (ee+1) * len_mcmc, :] = fit_list[0].extract(permuted=False, inc_warmup=False)[:,0,:]

        ## exchange replicas
        E = [-fit_list[r].extract(permuted=False, pars="lp__")["lp__"][-1] for r in range(n_rep)]
        idx = np.arange(n_rep, dtype=np.int64)

        for rr in idx_ex(n_rep, ee):
            w = np.exp(-(inv_T[rr+1] - inv_T[rr]) * (E[rr] - E[rr+1]))
            if (np.random.uniform(0,1,1) < w):
                idx[rr] = rr+1
                idx[rr+1] = rr
        E_tbl[ee, :] = E
        idx_tbl[ee, :] = idx

        for rr in range(n_rep):
            ms = fit_list[idx_tbl[ee, rr]].extract(permuted=False)[-1, 0, :np.sum(par_length)]
            init_dict = [dict(zip(par_list, np.split(ms, np.cumsum(par_length))))]
            init_list[rr] = [{k:v for k,v in zip(init_dict[0].keys(), [cast_staninit(init[1]) for init in init_dict[0].items()])}]

    return ms_T1, idx_tbl, E_tbl

import numpy as np
from pystan import StanModel


class ReplicaExchange:
    """Replica Exchange Monte Carlo.
    """    
    def __init__(self, n_ex, *, inv_T, stanmodel=None, file=None):
        """initialaize ReplicaExchange

        Args:
            n_ex (int): Total number of replica exchange.
            inv_T (ndarray): ndarray of inverse temperatures.
            stanmodel (StanModel): StanModel instance used for MCMC.
            file (str) : filename of `.stan` file.
        """        
        self.n_ex = n_ex
        self.inv_T = inv_T
        self.n_rep = len(self.inv_T)

        if(stanmodel != None):
            self.stanmodel = stanmodel
        else:
            if(stanmodel==None and file != None):
                self.stanmodel = StanModel(file=file)
            else:
                raise TypeError("missing 1 required keyword-only argument: 'stanmodel'")
    
    def sampling(self, data, n_iter, warmup, par_init):
        """Sampling using Replica-Exchange MC.

        Args:
            data (dict): Standata for the stanmodel.
            n_iter (int): Positive integer specifying how many iterations for each stanmodel before swapping replicas.
            warmup (int): Positive integer specifying number of warmup (aka burin) iterations. 
            par_init (dict): dict that store the initial values of the parameters to be sampled.

        Returns:
            ms_T1: Samples from the replica that inv_T=1.0 .
            idx_tbl: Table containing the indexes of replicas during sampling.
            E_tbl: Table containing -lp__ for each replica during replica exchange.
        """
        par_length = self.get_par_length(par_init)

        len_mcmc = n_iter - warmup
        n_param = np.sum(par_length) + 1 # n_parameters includes lp__.
        ms_T1 = np.zeros((len_mcmc*self.n_ex, n_param)) # <- MCMC samples at inv_T=1.

        idx_tbl = np.zeros((self.n_ex, self.n_rep), dtype=np.int64) # index table of (exchange time, replica)
        E_tbl = np.zeros((self.n_ex, self.n_rep)) # E table
        init_list = [[par_init]] * self.n_rep

        for ee in range(self.n_ex):
            fit_list = []
            for r in range(self.n_rep):
                data_tmp = data.copy()
                data_tmp["beta"] = self.inv_T[r]
                fit_list.append(self.stanmodel.sampling(data=data_tmp, init=init_list[r], iter=n_iter, warmup=warmup, chains=1, seed=r, check_hmc_diagnostics=False))
            ms_T1[ee * len_mcmc : (ee+1) * len_mcmc, :] = fit_list[0].extract(permuted=False, inc_warmup=False)[:,0,:]

            ## exchange replicas
            E = [-fit_list[r].extract(permuted=False, pars="lp__")["lp__"][-1] for r in range(self.n_rep)]
            idx = np.arange(self.n_rep, dtype=np.int64)

            for rr in self.idx_ex(ee):
                w = np.exp(-(self.inv_T[rr+1] - self.inv_T[rr]) * (E[rr] - E[rr+1]))
                if (np.random.uniform(0,1,1) < w):
                    idx[rr] = rr+1
                    idx[rr+1] = rr
            E_tbl[ee, :] = E
            idx_tbl[ee, :] = idx

            for rr in range(self.n_rep):
                ms = fit_list[idx_tbl[ee, rr]].extract(permuted=False)[-1, 0, :np.sum(par_length)]
                init_dict = dict(zip(par_init, np.split(ms, np.cumsum(par_length))))
                init_list[rr] = [{k:v for k,v in zip(init_dict.keys(), [self.cast_staninit(init[1]) for init in init_dict.items()])}]

        return ms_T1, idx_tbl, E_tbl

    def idx_ex(self, ee):
        """generating idx of replicas to be exchanged.

        Args:
            ee (int): Number of exchanges so far.

        Returns:
            idx_half: Index of replicas to exchange.
        """   
        idx_half = np.arange(0, np.floor(self.n_rep/2), dtype=np.int64) * 2
        if (ee % 2 == 1):
            return idx_half[:-1] + 1
        else:
            return idx_half
    
    @staticmethod
    def cast_staninit(init):
        """Convert a parameter to scalar.

        Args:
            init (ndarray): Initial value of the parameter to be sampled by MCMC

        Returns:
            init: the parameter converted to scalar.
        """    
        if(init.size==1):
            try:
                return np.asscalar(init)
            except ValueError:
                return init
        else:
            return init

    @staticmethod
    def get_par_length(par_dict):
        """get length of each parameter.

        Args:
            par_dict (dict): list of parameters.

        Returns:
            ndarray of all parameters' length.
        """    
        par_length = []
        for item in par_dict.values():
            try:
                par_length += [len(item)]
            except TypeError:
                par_length += [1]
        return np.array(par_length)


if __name__ == "__main__":
    import argparse
    import pickle

    import matplotlib.pyplot as plt 

    from mpl_toolkits.mplot3d import Axes3D 
    from scipy.stats import multivariate_normal

    parser = argparse.ArgumentParser()
    parser.add_argument("stanmodel", help="path stanmodel")
    args = parser.parse_args()

    filename = args.stanmodel #"./model/gmm-posterior-PT.pickle"
    with open(filename, mode="rb") as f:
        stanmodel = pickle.load(f)
    
    A1 = 0.9 ## ratio of GMM
    A2 = 0.1 

    grid = np.meshgrid(np.arange(-3, 7, 0.005), np.arange(-3, 7, 0.005))
    z = A1*multivariate_normal.pdf(mean=[0, 0], cov=[[0.2, 0], [0, 0.2]], x=np.c_[grid[0].reshape(-1), grid[1].reshape(-1)])
    z += A2*multivariate_normal.pdf(mean=[4, 4], cov=[[0.2, 0], [0, 0.2]], x=np.c_[grid[0].reshape(-1), grid[1].reshape(-1)])

    data = {}
    init = dict(p=np.array([0, 0]))

    N_rep = 30 ## number of replicas
    N_ex = 200 ## number of exchanges
    Inv_T = 0.5 ** np.linspace(0, -np.log(0.2)/np.log(2), num=N_rep)

    replica_exchange = ReplicaExchange(n_ex=N_ex, inv_T=Inv_T, stanmodel = stanmodel)
    result = replica_exchange.sampling(data=data, par_init=init, n_iter=70, warmup=50)

    fig = plt.figure()
    fig.tight_layout()
    ax = Axes3D(fig)

    ax.plot_wireframe(grid[0], grid[1], z.reshape(grid[0].shape), color="tab:green", alpha=0.3)
    ax.scatter(result[0][100:,0], result[0][100:,1], alpha=0.3)
    plt.show()
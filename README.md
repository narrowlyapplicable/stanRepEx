# Replica Exchange MC using PyStan

is based on the implementation example (using R & RStan) on [this site](https://statmodeling.hatenablog.com/entry/stan-parallel-tempering) 

## Requirement
- NumPy
    - URL: https://github.com/numpy/numpy

- PyStan 
  - URL: https://github.com/stan-dev/pystan/ 

## Installation
`pip install git+https://github.com/narrowlyapplicable/stanRepEx`

## Usage
```python
from stanRepEx import ReplicaExchange, ReplicaExchangeParallel

data = {} # standata (empty in this example)
init = dict(p=np.array([0, 0])) # initial value of the parameter to be estimated

N_rep = 30 ## number of replicas
N_ex = 200 ## number of exchanges
Inv_T = 0.5 ** np.linspace(0, -np.log(0.2)/np.log(2), num=N_rep)

# w/o Parallelization
replica_exchange = ReplicaExchange(n_ex=N_ex, inv_T=Inv_T, stanmodel=stanmodel)
result = replica_exchange.sampling(data=data, par_init=init, n_iter=70, warmup=50)

# with Parallelization
replica_exchange = ReplicaExchangeParallel(n_ex=N_ex, inv_T=Inv_T, stanmodel=stanmodel)
result = replica_exchange.sampling(data=data, par_init=init, n_iter=70, warmup=50)
```

## DEMONSTRATION
- [test.ipynb](https://github.com/narrowlyapplicable/stanRepEx/blob/master/test.ipynb)


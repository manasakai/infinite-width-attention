import numpy as np
import multiprocess as mp
import pandas as pd
import time
from numpy.random import SeedSequence

# Simulation parameters will be set via params dict
# n: Network width
# s: Spatial dimension
# H: Number of heads
# C: Clipping threshold for psi
# num_runs: Number of simulation runs
# num_processes: Number of parallel processes
# seed: Global seed for main process, base for spawning

# Globals (initialized in init_params)
n = None
s = None
H = None
C = None
num_runs = None
num_processes = None
seed = None

# Weight sampling W ~ N(0,1/n)
def scaled_weights(shape, scale_dim):
    return np.random.randn(*shape) / np.sqrt(scale_dim)

# Initialize globals in each process or main (for simulate_theoretical and potentially main thread)
def init_globals(params):
    global n, s, H, num_runs, num_processes, seed
    n = params.get('n', n)
    s = params.get('s', s)
    H = params.get('H', H)
    C = params.get('C', C)
    num_runs = params.get('num_runs', num_runs)
    num_processes = params.get('num_processes', num_processes)
    seed = params.get('seed', seed)
    np.random.seed(seed) # Seeds the current process

# Single empirical run: returns tuple (G_sqrt_n, G_n)
def single_run(args):
    params, task_specific_seed = args
    n = params['n']
    s = params['s']
    H = params['H']
    C = params['C']
    seed_int = task_specific_seed.generate_state(1)[0]
    np.random.seed(seed_int)

    # Sample initial vector h ~ N(0, I_n)
    h = np.random.randn(n)
    #  Compute h^i = W^i h for i=1…s
    Wstack = scaled_weights((s, n, n), n).reshape(s, n, n)
    H_mat  = Wstack @ h # Each row is h^i
    # Apply clipping activation elementwise to each h^i
    X = np.clip(H_mat, -C, C) # shape (s, n)
    Wq = scaled_weights((n, n), n)
    Wk = scaled_weights((n, n), n)
    Q  = X @ Wq
    K  = X @ Wk
    Inner = Q.dot(K.T)
    G_sqrtn = Inner / np.sqrt(n)
    G_n = Inner / n
    # print(f'[{mp.current_process().name}] finished, seed used: {seed_int}')
    return (G_sqrtn[0, 0], G_n[0, 0])

# Simulate empirical: returns array (num_runs, 2)
def simulate_empirical(params):
    # Main process seed (params['seed']) is used to spawn child seeds for tasks.
    ss = SeedSequence(params['seed'])
    # Spawn num_runs independent SeedSequence objects for each task.
    child_seeds_for_tasks = ss.spawn(params['num_runs'])

    tasks = [(params, ss) for ss in child_seeds_for_tasks]
    with mp.Pool(processes=params['num_processes']) as pool:
        out = pool.map(single_run, tasks)
    return np.stack(out)

if __name__ == '__main__':
    start = time.time()
    # Settings for experiments
    base = {
        'n': 256,
        's': 4,
        'H': 2,
        'C': 100,
        'num_runs': 50000,
        'num_processes': 18,
        'seed': 0
    }

    # Generate seeds
    ss_global = SeedSequence(base['seed'])

    # empirical
    records = simulate_empirical(base)

    # Save to CSV
    pd.DataFrame(records).to_csv('data/data_score.csv', index=False)

    # print(f'Sampling done in {time.time()-start:.2f} seconds')

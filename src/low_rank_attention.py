# This considers low‐rank attention, where inner product dimension is n_H instead of n

import numpy as np
import multiprocess as mp
import pandas as pd
import time
from numpy.random import SeedSequence

# Simulation parameters will be set via params dict
# n: Full network width
# s: Spatial dimension
# H: Number of heads
# n_H: Per‐head width (so that n = H * n_H)
# C: Clipping threshold for psi
# num_runs: Number of simulation runs
# num_processes: Number of parallel processes
# seed: Global seed for main process, base for spawning

# Globals (initialized in init_globals)
n = None
s = None
H = None
n_H = None
C = None
num_runs = None
num_processes = None
seed = None

# Weight sampling W ~ N(0,1/scale_dim)
def scaled_weights(shape, scale_dim):
    return np.random.randn(*shape) / np.sqrt(scale_dim)

# Softmax function
def softmax(x, axis=-1):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Initialize globals in each process or main (for simulate_theoretical and potentially main thread)
def init_globals(params):
    global n, n_H, s, H, num_runs, num_processes, seed, C
    n = params.get('n', n)
    s = params.get('s', s)
    H = params.get('H', H)
    n_H = params.get('n_H', n // H) # per‐head dimension
    C = params.get('C', C)
    num_runs = params.get('num_runs', num_runs)
    num_processes = params.get('num_processes', num_processes)
    seed = params.get('seed', seed)
    np.random.seed(seed) # Seeds the current process

# Single empirical run: output matrix Y shape (s,n)
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
    accum = np.zeros((s, n))
    for _ in range(H):
        Wq = scaled_weights((n, n_H), n) # (n, n_H)
        Wk = scaled_weights((n, n_H), n) # (n, n_H)
        Wv = scaled_weights((n, n_H), n) # (n, n_H)
        Wo = scaled_weights((n_H, n), n_H) # (n_H, n)
        Q = X @ Wq # (s, n_H)
        K = X @ Wk # (s, n_H)
        V = X @ Wv # (s, n_H)
        tV = V @ Wo # (s, n)
        G = Q.dot(K.T) / np.sqrt(n_H)
        A = softmax(G, axis=1)
        accum += A @ tV
    # print(f'[{mp.current_process().name}] finished, seed used: {seed_int}')
    return accum

# Simulate empirical: returns array (num_runs, s, n)
def simulate_empirical(params):
    # Main process seed (params['seed']) is used to spawn child seeds for tasks.
    ss = SeedSequence(params['seed'])
    # Spawn num_runs independent SeedSequence objects for each task.
    child_seeds_for_tasks = ss.spawn(params['num_runs'])

    tasks = [(params, ss) for ss in child_seeds_for_tasks]
    with mp.Pool(processes=params['num_processes']) as pool:
        out = pool.map(single_run, tasks)
    return np.stack(out)

# Theoretical Monte Carlo: returns array (num_runs, s)
def simulate_theoretical(params):
    init_globals(params)
    p = np.random.randn(params['num_runs'], H, s, s)
    Z = np.random.randn(params['num_runs'], H, s)
    y = np.zeros((params['num_runs'], s))
    for i in range(params['num_runs']):
        for a in range(H):
            logits = p[i, a]
            probs = softmax(logits, axis=1)
            y[i] += probs.dot(Z[i, a])
    return y

if __name__ == '__main__':
    start = time.time()
    # Settings for experiments
    n_H = 64
    base = {
        'n_H': n_H,
        's': 4,
        'C': 100,
        'num_runs': 50000,
        'num_processes': 18,
        'seed': 0
    }

    # Prepare list
    H_vals = [4 ** i for i in range(3)]

    records = []

    # Generate seeds
    ss_global = SeedSequence(base['seed'])
    child_param_ss = ss_global.spawn(len(H_vals))
    param_seeds = [ ss.generate_state(1)[0] for ss in child_param_ss]

    for idx, H_val in enumerate(H_vals):
        params = base.copy()
        params['H'] = H_val
        params['n'] = n_H * H_val
        params['seed'] = param_seeds[idx]

        # theoretical
        theo = simulate_theoretical(params)
        y_theo = theo[:, 0]

        # empirical
        emp = simulate_empirical(params)
        y_emp = emp[:, 0, 0]

        for ye, yt in zip(y_emp, y_theo):
            records.append({'param': H_val, 'y_emp': ye, 'y_theo': yt})

    # Save to CSV
    pd.DataFrame(records).to_csv('data/data_lowrank.csv', index=False)

    # print(f'Sampling done in {time.time()-start:.2f} seconds')

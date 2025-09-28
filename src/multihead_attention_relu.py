import numpy as np
import multiprocess as mp
import pandas as pd
import time
from numpy.random import SeedSequence

# Simulation parameters will be set via params dict
# n: Network width
# s: Spatial dimension
# H: Number of heads
# num_runs: Number of simulation runs
# num_processes: Number of parallel processes
# seed: Global seed for main process, base for spawning

# Globals (initialized in init_globals)
n = None
s = None
H = None
num_runs = None
num_processes = None
seed = None

# Weight sampling W ~ N(0,1/n)
def scaled_weights(shape, scale_dim):
    return np.random.randn(*shape) / np.sqrt(scale_dim)

# Softmax function
def softmax(x, axis=-1):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

# Initialize globals in each process or main (for simulate_theoretical and potentially main thread)
def init_globals(params):
    global n, s, H, num_runs, num_processes, seed
    n = params.get('n', n)
    s = params.get('s', s)
    H = params.get('H', H)
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
    seed_int = task_specific_seed.generate_state(1)[0]
    np.random.seed(seed_int)

    # Sample initial vector h ~ N(0, I_n)
    h = np.random.randn(n)
    #  Compute h^i = W^i h for i=1…s
    Wstack = scaled_weights((s, n, n), n).reshape(s, n, n)
    H_mat  = Wstack @ h # Each row is h^i
    # Apply ReLU activation elementwise to each h^i
    X = np.maximum(H_mat, 0) # shape (s, n)
    accum = np.zeros((s, n))
    for _ in range(H):
        Wq = scaled_weights((n, n), n)
        Wk = scaled_weights((n, n), n)
        Wv = scaled_weights((n, n), n)
        Wo = scaled_weights((n, n), n)
        Q = X @ Wq
        K = X @ Wk
        V = X @ Wv
        tV = V @ Wo
        G = Q.dot(K.T) / np.sqrt(n)
        A = softmax(G, axis=1)
        scale = 1 / np.sqrt(H)
        accum += scale * A @ tV
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
    var_diag = 1 / 2
    cov_off_diag = 1 / (2 * np.pi)
    diff_diag = var_diag - cov_off_diag
    var_Z = cov_off_diag * np.ones((s,s)) + diff_diag * np.eye(s)
    var_p = np.kron(var_Z, var_Z)
    y = np.zeros((params['num_runs'], s))
    scale = 1 / np.sqrt(H)
    for i in range(params['num_runs']):
        for a in range(H):
            Z = np.random.multivariate_normal(np.zeros(s), var_Z)
            p = np.random.multivariate_normal(np.zeros(s**2), var_p).reshape(s,s)
            probs = softmax(p, axis=1)
            y[i] += scale * probs.dot(Z)
    return y

# Set the number of experiments
num_experiments = 10

if __name__ == '__main__':
    start = time.time()
    for i in range(num_experiments):
        # Set seed for this entire experiment iteration
        current_experiment_seed = i
        # Settings for experiments
        exp_min, exp_max = 2, 5
        default_n = 256
        default_H = 2
        base = {
            'n': default_n,
            's': 4,
            'H': default_H,
            'num_runs': 100000,
            'num_processes': 100,
            'seed': current_experiment_seed
        }

        # Prepare lists
        n_vals = [4 ** e for e in range(exp_min, exp_max + 1)]
        H_vals = [1, 256]

        records_n = []
        records_H = []

        # Generate seeds
        ss_global = SeedSequence(base['seed'])
        child_param_ss = ss_global.spawn(len(n_vals) + len(H_vals))
        param_seeds_i = [ ss.generate_state(1)[0] for ss in child_param_ss]

        # Vary n
        # theoretical
        theo_n = simulate_theoretical(base) # simulate_theoretical uses the 'seed' from 'base' via init_globals
        y_theo_n = theo_n[:, 0]

        for idx, n_val in enumerate(n_vals):
            params = base.copy()
            params['n'] = n_val
            params['seed'] = param_seeds_i[idx]

            # empirical
            emp = simulate_empirical(params)
            y_emp = emp[:, 0, 0]

            for ye, yt in zip(y_emp, y_theo_n):
                records_n.append({'param': n_val, 'y_emp': ye, 'y_theo': yt})

        # Save to CSV
        out_n = f'data/data_vary_n_relu_{current_experiment_seed}.csv'
        pd.DataFrame(records_n).to_csv(out_n, index=False)

        print(f'Experiment {current_experiment_seed} done')

    print(f'Sampling done in {time.time()-start:.2f} seconds')

# infinite-width-attention

This repository provides the source code for the numerical simulations in the paper **"Infinite-Width Limit of a Single Attention Layer: Analysis via Tensor Programs."** Please refer to the paper for a detailed explanation of the experiments and results.

**Paper reference:** Mana Sakai, Ryo Karakida, and Masaaki Imaizumi. (2025). Infinite-Width Limit of a Single Attention Layer: Analysis via Tensor Programs. [arXiv:2506.00846](https://arxiv.org/abs/2506.00846).

## Notes on arXiv Versions

The current main branch corresponds to the code used for the latest version of our paper. For reproducibility, the code corresponding to version 1 of arXiv is preserved in the `arxiv-v1` branch. Please refer to the arxiv-v1 branch and the corresponding README if you are interested in the experiments from the first version of our paper.

## Implementation

All experiments can be executed via the `run.py` script. The generated data is saved in `./data/`, and plots are saved in `./figures/`.

- **To run all experiments and generate all figures:**
  ```bash
  python run.py all
  ```
  or simply:
  ```bash
  python run.py
  ```
- **To reproduce specific figures:**
  1. **Figures 1 and 2(B) of the paper:**　This corresponds to `Experiment 1` in the script.
      ```bash
      python run.py 1
      ```
      This command will generate: `vary_n.pdf` (Figure 1(a)), `vary_H.pdf` (Figure 2(b)), `kl_vary_n.pdf` (Figure 1(b)), `kl_vary_H.pdf` (not used in the paper), and `kl_vary_H_inf_head.pdf` (not used in the paper) in the `figures/` directory.
  2.  **Figure 2(A) of the paper:**
      This corresponds to `Experiment 2`.
      ```bash
      python run.py 2
      ```
      This will generate `score_dist.pdf` (Figure 2(a)).
  3.  **Figure 3 of the paper:**
      This corresponds to `Experiment 3`.
      ```bash
      python run.py 3
      ```
      This will generate `lowrank.pdf` (Figure 3(a)) and `kl_lowrank.pdf` (Figure 3(b)).
  4.  **Figure 5 of the paper:**
      This corresponds to `Experiment 4`.
      ```bash
      python run.py 4
      ```
      This will generate `vary_n_relu.pdf` (Figure 5(a)) and `kl_vary_n_relu.pdf` (Figure 5(b)).

## Requirements

- **Python versions:** Tested on Python 3.13.0 (macOS 15 Sequoia) and Python 3.13.3 (Windows 11).
- **Required libraries:** Check the `requirements.txt` file for the list of dependencies.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

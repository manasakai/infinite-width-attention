# infinite-width-attention

This repository provides the source code for the numerical simulations in the paper **"Infinite-Width Limit of a Single Attention Layer: Analysis via Tensor Programs"**. Please refer to the paper for a detailed explanation of the experiments and results.

**Paper reference:** Mana Sakai, Ryo Karakida, and Masaaki Imaizumi. (2025). Infinite-Width Limit of a Single Attention Layer: Analysis via Tensor Programs. [arXiv:2506.00846](https://arxiv.org/abs/2506.00846).

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
      This command will generate: `vary_n.pdf` (Figure 1(A)), `vary_H.pdf` (Figure 2(B)), `kl_vary_n.pdf` (Figure 1(B)), and `kl_vary_H.pdf` (not used in the paper) in the `figures/` directory.
  2.  **Figure 2(A) of the paper:**
      This corresponds to `Experiment 2`.
      ```bash
      python run.py 2
      ```
      This will generate `score_dist.pdf` (Figure 2(A)).
  3.  **Figure 3 of the paper:**
      This corresponds to `Experiment 3`.
      ```bash
      python run.py 3
      ```
      This will generate `lowrank.pdf` (Figure 3).

## Requirements

- **Python versions:** Tested on Python 3.13.0 (macOS 15 Sequoia).
- **Required libraries:** Check the `requirements.txt` file for the list of dependencies.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

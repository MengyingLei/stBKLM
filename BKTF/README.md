# BKMF/BKTF: Bayesian Kernelized Matrix/Tensor Factorization
[[paper](https://ieeexplore.ieee.org/document/9745749)]
<p align="center">
<img src="./image/hyper_SeData_imputation.png" style="width: 50%"><br>
<em>Trace plots and probability distributions of learned kernel hyperparameters for SeData imputation.</em>
</p>

## Key
- Place GP priors on columns of the latent factor matrices to encode spatial/temporal correlations.
- Develop a fully Bayesian (MCMC) model, where the kernel hyperparameters are sampled from the marginal posterior via slice sampling.

## Code
**Script:**
- [`BKMF.py`](./BKMF.py): run BKMF on Seattle traffic speed data for one month (323 locations, 720 time points = 24 per day × 30 days).

**Notebook:**
- [`BKMF.ipynb`](./BKMF.ipynb): demonstrate BKMF on the Seattle traffic speed data.

## Citation
**bibtex**
```bibtex
@article{lei2022bkmf,
  title={Bayesian Kernelized Matrix Factorization for Spatiotemporal Traffic Data Imputation and Kriging},
  author={Lei, Mengying and Labbe, Aurelie and Wu, Yuankai and Sun, Lijun},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={10},
  pages={18962--18974},
  year={2022},
  publisher={IEEE}
}
```
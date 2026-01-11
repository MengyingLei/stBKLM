# stBKLM: Bayesian Kernelized Low-rank Modeling (BKLM) for <ins>s</ins>patio<ins>t</ins>emporal data
This repository contains implementations of Bayesian kernelized (Gaussian process regularized) low-rank models for multidimensional spatiotemporal data.

- [BKTF: Bayesian Kernelized Tensor Factorization](#bkmfbktf-bayesian-kernelized-matrixtensor-factorization)
- [BKTR: Bayesian Kernelized Tensor Regression](#bktr-bayesian-kernelized-tensor-regression)
- [BCKL: Bayesian Complementary Kernelized Learning](#bckl-bayesian-complementary-kernelized-learning)

## [BKMF/BKTF: Bayesian Kernelized Matrix/Tensor Factorization](./BKTF)
[[paper](https://ieeexplore.ieee.org/document/9745749)] [[codebase](./BKTF)]
<p align="center">
<img src="./BKTF/image/hyper_SeData_imputation.png" style="width: 50%"><br>
<em>Trace plots and probability distributions of learned kernel hyperparameters for SeData imputation.</em>
</p>

#### Abstract
Missingness and corruption are common problems for real-world traffic data. How to accurately perform imputation and prediction based on incomplete or even sparse traffic data becomes a critical research question in intelligent transportation systems. Low-rank matrix factorization (MF) is a common solution for the general missing value imputation problem. To better characterize and encode the strong spatial and temporal consistency in traffic data, existing work has introduced flexible spatial/temporal Gaussian process (GP) priors to model the latent factors in MF framework, which also allows us to perform kriging for unseen locations and virtual sensors. However, learning the hyperparameters in GP kernels remains a challenging task. In this paper, we present a Bayesian kernelized matrix factorization (BKMF) model with an efficient Markov chain Monte Carlo (MCMC) sampling algorithm for model inference. By learning the kernel hyperparameters from their marginal posteriors through a slice sampling treatment and updating the latent factors alternatively with Gibbs sampling, we achieve a fully Bayesian model for the spatiotemporally kernelized (i.e., GP prior regularized) MF framework. We apply BKMF on both imputation and kriging tasks, and our results demonstrate the superiority of BKMF compared with state-of-the-art spatiotemporal models. In addition, we also explore the effects of different GP kernels in characterizing networked spatiotemporal traffic state data.

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

## [BKTR: Bayesian Kernelized Tensor Regression](./BKTR)
[[arXiv](https://arxiv.org/abs/2109.00046)] [[paper](https://doi.org/10.1214/24-BA1428)] [[package](https://cran.r-project.org/package=BKTR)]

#### Abstract
As a regression technique in spatial statistics, the spatiotemporally varying coefficient model (STVC) is an important tool for discovering nonstationary and interpretable response-covariate associations over both space and time. However, it is difficult to apply STVC for large-scale spatiotemporal analyses due to its high computational cost. To address this challenge, we summarize the spatiotemporally varying coefficients using a third-order tensor structure and propose to reformulate the spatiotemporally varying coefficient model as a special low-rank tensor regression problem. The low-rank decomposition can effectively model the global patterns of large data sets with a substantially reduced number of parameters. To further incorporate the local spatiotemporal dependencies, we use Gaussian process (GP) priors on the spatial and temporal factor matrices. We refer to the overall framework as Bayesian Kernelized Tensor Regression (BKTR), and kernelized tensor factorization can be considered a new and scalable approach to modeling multivariate spatiotemporal processes with a low-rank covariance structure. For model inference, we develop an efficient Markov chain Monte Carlo (MCMC) algorithm, which uses Gibbs sampling to update factor matrices and slice sampling to update kernel hyperparameters. We conduct extensive experiments on both synthetic and real-world data sets, and our results confirm the superior performance and efficiency of BKTR for model estimation and parameter inference.

```bibtex
@article{lei2024bktr,
  title={Scalable Spatiotemporally Varying Coefficient Modeling with Bayesian Kernelized Tensor Regression},
  author={Lei, Mengying and Labbe, Aur{\'e}lie and Sun, Lijun},
  journal={Bayesian Analysis},
  volume={1},
  number={1},
  pages={1--29},
  year={2024},
  publisher={International Society for Bayesian Analysis}
}
```

## BCKL: Bayesian Complementary Kernelized Learning
[[arXiv](https://arxiv.org/abs/2208.09978)]

#### Abstract
Probabilistic modeling of multidimensional spatiotemporal data is critical to many real-world applications. As real-world spatiotemporal data often exhibits complex dependencies that are nonstationary and nonseparable, developing effective and computationally efficient statistical models to accommodate nonstationary/nonseparable processes containing both long-range and short-scale variations becomes a challenging task, in particular for large-scale datasets with various corruption/missing structures. In this paper, we propose a new statistical framework—Bayesian Complementary Kernelized Learning (BCKL)—to achieve scalable probabilistic modeling for multidimensional spatiotemporal data. To effectively characterize complex dependencies, BCKL integrates two complementary approaches—kernelized low-rank tensor factorization and short-range spatiotemporal Gaussian Processes. Specifically, we use a multi-linear low-rank factorization component to capture the global/long-range correlations in the data and introduce an additive short-scale GP based on compactly supported kernel functions to characterize the remaining local variabilities. We develop an efficient Markov chain Monte Carlo (MCMC) algorithm for model inference and evaluate the proposed BCKL framework on both synthetic and real-world spatiotemporal datasets. Our experiment results show that BCKL offers superior performance in providing accurate posterior mean and high-quality uncertainty estimates, confirming the importance of both global and local components in modeling spatiotemporal data.

```bibtex
@article{lei2022bckl,
  title={Bayesian Complementary Kernelized Learning for Multidimensional Spatiotemporal Data},
  author={Lei, Mengying and Labbe, Aurelie and Sun, Lijun},
  journal={arXiv preprint arXiv:2208.09978},
  year={2022}
}
```
# stBKLM: Bayesian Kernelized Low-rank Modeling (BKLM) for spatiotemporal data
This repository contains implementations of Bayesian kernelized (Gaussian process regularized) low-rank models for multidimensional spatiotemporal data.

## [BKMF/BKTF: Bayesian Kernelized Matrix/Tensor Factorization](./BKTF)
[paper](https://ieeexplore.ieee.org/document/9745749) [pdf](./BKTF/Bayesian_Kernelized_Matrix_Factorization_for_Spatiotemporal_Traffic_Data_Imputation_and_Kriging_IEEE.pdf)

#### Abstract

Missingness and corruption are common problems for real-world traffic data. How to accurately perform imputation and prediction based on incomplete or even sparse traffic data becomes a critical research question in intelligent transportation systems. Low-rank matrix factorization (MF) is a common solution for the general missing value imputation problem. To better characterize and encode the strong spatial and temporal consistency in traffic data, existing work has introduced flexible spatial/temporal Gaussian process (GP) priors to model the latent factors in MF framework, which also allows us to perform kriging for unseen locations and virtual sensors. However, learning the hyperparameters in GP kernels remains a challenging task. In this paper, we present a Bayesian kernelized matrix factorization (BKMF) model with an efficient Markov chain Monte Carlo (MCMC) sampling algorithm for model inference. By learning the kernel hyperparameters from their marginal posteriors through a slice sampling treatment and updating the latent factors alternatively with Gibbs sampling, we achieve a fully Bayesian model for the spatiotemporally kernelized (i.e., GP prior regularized) MF framework. We apply BKMF on both imputation and kriging tasks, and our results demonstrate the superiority of BKMF compared with state-of-the-art spatiotemporal models. In addition, we also explore the effects of different GP kernels in characterizing networked spatiotemporal traffic state data.

```
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
[arXiv](https://arxiv.org/abs/2109.00046) [paper](https://doi.org/10.1214/24-BA1428) [pdf](./BKTR/BA1428_Scalable_Spatiotemporally_Varying_Coefficient_Modeling_with_Bayesian_Kernelized_Tensor_Regression.pdf) [package](https://cran.r-project.org/package=BKTR)

#### Abstract

As a regression technique in spatial statistics, the spatiotemporally varying coefficient model (STVC) is an important tool for discovering nonstationary and interpretable response-covariate associations over both space and time. However, it is difficult to apply STVC for large-scale spatiotemporal analyses due to its high computational cost. To address this challenge, we summarize the spatiotemporally varying coefficients using a third-order tensor structure and propose to reformulate the spatiotemporally varying coefficient model as a special low-rank tensor regression problem. The low-rank decomposition can effectively model the global patterns of large data sets with a substantially reduced number of parameters. To further incorporate the local spatiotemporal dependencies, we use Gaussian process (GP) priors on the spatial and temporal factor matrices. We refer to the overall framework as Bayesian Kernelized Tensor Regression (BKTR), and kernelized tensor factorization can be considered a new and scalable approach to modeling multivariate spatiotemporal processes with a low-rank covariance structure. For model inference, we develop an efficient Markov chain Monte Carlo (MCMC) algorithm, which uses Gibbs sampling to update factor matrices and slice sampling to update kernel hyperparameters. We conduct extensive experiments on both synthetic and real-world data sets, and our results confirm the superior performance and efficiency of BKTR for model estimation and parameter inference.

```
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
[arXiv](https://arxiv.org/abs/2208.09978)

```
@article{lei2022bckl,
  title={Bayesian Complementary Kernelized Learning for Multidimensional Spatiotemporal Data},
  author={Lei, Mengying and Labbe, Aurelie and Sun, Lijun},
  journal={arXiv preprint arXiv:2208.09978},
  year={2022}
}
```
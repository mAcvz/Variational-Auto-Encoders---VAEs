## Introduction and theoretical background

**Variational Auto-Encoders (VAEs)** are an inference paradigm invented by D. P. Kingma and M. Welling with the goal of performing efficient approximate inference and learning with models characterized by continuous latent variables and/or parameters that have intractable posterior distributions. In order to address VAEs, it is worth starting from the basic idea of Variational Bayes, i.e. approximating the target posterior distribution with a distribution

$$
q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi}), \tag{1}
$$

belonging to a certain family of functions \(Q\), where \(\boldsymbol{z}\) and \(\boldsymbol{\phi}\) are the the latent variables and the function parameters respectively. If 

$$
p(\boldsymbol{x}, \boldsymbol{z}|\boldsymbol{\theta}) \tag{2}
$$

represents the joint probability distribution of our visible \(\boldsymbol{x}\) and hidden \(\boldsymbol{z}\) units, given the model parameters \(\boldsymbol{\theta}\), the marginal probability distribution of \(\boldsymbol{x}\) satisfies

$$
\log p(\boldsymbol{x}|\boldsymbol{\theta}) = D_{KL}(q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi})||p(\boldsymbol{z}|\boldsymbol{x}, \boldsymbol{\theta})) + \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}) \tag{3}
$$

with 

$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}) = \mathbb{E}_{q_{\phi}} \left[\log p(\boldsymbol{x}, \boldsymbol{z}|\boldsymbol{\theta}) - \log q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi})\right] \tag{4}
$$

being the **variational lower bound** of the marginal likelihood. Our goal is finding \(q\) such that the Kullback-Leibler divergence term is minimum and this minimization problem is equivalent to the maximization of the variational lower bound \(\mathcal{L}\). The standard approach to this problem is the Mean Field (MF) ansatz, which consists in assuming a factorized form of \(q\) over a subset of latent variables:

$$
q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi}) = \prod_i q(\boldsymbol{z_i}|\boldsymbol{x},\boldsymbol{\phi}). \tag{5}
$$

The problem is that MF approach involves the calculations of integrals depending on \(q_i\) that in many cases are **intractable**, thus limiting the range if its applicability. Starting from these observations, the article focuses on finding a solution to the problems that arise consequently:
- efficient approximate estimation for the parameters \(\boldsymbol{\theta}\)
- efficient approximate estimation of \(p(\boldsymbol{z}|\boldsymbol{x}, \boldsymbol{\theta})\)
- efficient approximate esitmation of \(p(\boldsymbol{x}|\boldsymbol{\theta})\).

Then, assuming that the MF approach is not doable, in our new scenario:
- \(q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi})\) is not assumed to be factorized anymore and acts as an **encoder** since starting from a datapoint \(\boldsymbol{x}\) it provides a distribution over the the latent space variables
- \(p(\boldsymbol{x}|\boldsymbol{z},\boldsymbol{\theta})\) acts as a **decoder** since starting from an encoded datapoint \(\boldsymbol{z}\) living in the latent space it provides a distribution over the possible values of \(\boldsymbol{x}\) that may correspond to it.

Of course, we re left with the problem of learning the parameters that define our probabilistic encoder and decoder. To deal with it, it is useful to rewrite the variational lower bound (4) as 

$$
\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}^{(i)}) = -D_{KL}(q(\boldsymbol{z}|\boldsymbol{x}^{(i)},\boldsymbol{\phi})||p(\boldsymbol{z}|\boldsymbol{\theta})) + \mathbb{E}_{q_{\phi}} \left[\log p(\boldsymbol{x}^{(i)}|\boldsymbol{z}, \boldsymbol{\theta})\right] \tag{6}
$$

and thus our problem becomes the following **optimization**:

$$
\hat{\boldsymbol{\phi}}, \hat{\boldsymbol{\theta}} = \argmax_{\boldsymbol{\phi}, \boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}). \tag{7}
$$

In order to compute the expectation value, the article introduces the **reparametrization trick** to avoid estimates having a large variance that would undermine the optimization. This trick consists in setting

$$
\tilde{\mathcal{L}}^B(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}^{(i)}) = -D_{KL}(q(\boldsymbol{z}|\boldsymbol{x}^{(i)},\boldsymbol{\phi})||p(\boldsymbol{z}|\boldsymbol{\theta})) + \frac{1}{L} \sum_{l=1}^L \log p(\boldsymbol{x}^{(i)}|\boldsymbol{z_l}^{(i,l)}, \boldsymbol{\theta}) \tag{8}
$$

where \(\boldsymbol{z}_l = g_{\boldsymbol{\phi}}(\boldsymbol{\epsilon_l}, \boldsymbol{x})\) is a deterministic differentiable transformation and \(\boldsymbol{\epsilon}_l \sim p(\boldsymbol{\epsilon})\) is a random noise vector chosen such that it ensures \(\boldsymbol{z}_l \sim q(\boldsymbol{z}|\boldsymbol{x},\boldsymbol{\phi})\). The estimator of the variational lower bound provided by Eq. (8) is called **Stochastic Gradient Variational Bayes** (SGVB) estimator.

## VAE Implementation

The main focus of the article is showing the implementation of the aforementioned architecture in which both **the encoder and the decoder consist in neural networks**, more precisely Multi-Layer Perceptron (MLP) blocks, i.e. fully-connected blocks with only one hidden layer between input and output. This allows optimizing the lower bound (8) by exploiting the most comon gradient ascent algorithms. The implementation also involves:
1. setting the prior as a multivariate normal with zero mean and an identity as covariance matrix,
$$
p(\boldsymbol{z}|\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{0}, \boldsymbol{I}) \tag{9}
$$
2. assuming that the approximate posterior is Gaussian with diagional identity matrix
$$
q(\boldsymbol{z}|\boldsymbol{x}^{(i)},\boldsymbol{\phi}) \equiv \mathcal{N}(\boldsymbol{z}|\boldsymbol{\mu}_\phi(\boldsymbol{x}^{(i)}), diag(\boldsymbol{\sigma}^2_\phi(\boldsymbol{x}^{(i)}))) \tag{10}
$$
3. assuming that the conditional likelihood is also Gaussian with diagonal identity matrix
$$
p(\boldsymbol{x}^{(i)}|\boldsymbol{z}^{(i,l)}, \boldsymbol{\theta}) \equiv \mathcal{N}(\boldsymbol{x}^{(i)}|\boldsymbol{\mu}_\theta(\boldsymbol{z}^{(i,l)}), diag(\boldsymbol{\sigma}^2_\theta(\boldsymbol{z}^{(i,l)}))) \tag{11}
$$

In this way, the output of the encoder and of the decoder will be \(\boldsymbol{\mu}_\phi(\boldsymbol{x}), \boldsymbol{\sigma}^2_\phi(\boldsymbol{x})\) and \(\boldsymbol{\mu}_\theta(\boldsymbol{z_l}), \boldsymbol{\sigma}^2_\theta(\boldsymbol{z}_l)\) respectively. Moreover, thanks to these choices for the distributions, we have that the Kullback-Leibler divergence term assumes an analytical form, allowing us to obtain a **convenient form of the variational lower** bound that we can directly exploit inside the code of a **training loop**:

$$
\tilde{\mathcal{L}}^B(\boldsymbol{\theta}, \boldsymbol{\phi}; \boldsymbol{x}^{(i)}) = \frac{1}{2} \sum_{j=1}^J \left(1 + \log((\boldsymbol{\sigma}^{(i)}_\phi)^2_j) - (\boldsymbol{\mu}^{(i)}_\phi)_j^2 - (\boldsymbol{\sigma}^{(i)}_\phi)_j^2\right) + \frac{1}{L} \sum_{l=1}^L \log \mathcal{N}(\boldsymbol{x

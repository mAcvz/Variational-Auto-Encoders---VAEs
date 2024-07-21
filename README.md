## Introduction and theoretical background

**Variational Auto-Encoders (VAEs)** are an inference paradigm invented by D. P. Kingma and M. Welling with the goal of performing efficient approximate inference and learning with models characterized by continuous latent variables and/or parameters that have intractable posterior distributions. In order to address VAEs, it is worth starting from the basic idea of Variational Bayes, i.e. approximating the target posterior distribution with a distribution

$$
q(\bm{z}|\bm{x},\bm{\phi}), \tag{1}
$$

belonging to a certain family of functions $Q$, where $\bm{z}$ and $\bm{\phi}$ are the the latent variables and the function parameters respectively. If 

$$
p(\bm{x}, \bm{z}|\bm{\theta}) \tag{2}
$$

represents the joint probability distribution of our visible $\bm{x}$ and hidden $\bm{z}$ units, given the model parameters $\bm{\theta}$, the marginal probability distribution of $\bm{x}$ satisfies

$$
\log p(\bm{x}|\bm{\theta}) = D_{KL}(q(\bm{z}|\bm{x},\bm{\phi})||p(\bm{z}|\bm{x}, \bm{\theta})) + \mathcal{L}(\bm{\theta}, \bm{\phi}; \bm{x}) \tag{3}
$$

with 

$$
\mathcal{L}(\bm{\theta}, \bm{\phi}; \bm{x}) = \mathbb{E}_{q_{\phi}} \left[\log p(\bm{x}, \bm{z}|\bm{\theta}) - \log q(\bm{z}|\bm{x},\bm{\phi})\right] \tag{4}
$$

being the **variational lower bound** of the marginal likelihood. Our goal is finding $q$ such that the Kullback-Leibler divergence term is minimum and this minimization problem is equivalent to the maximization of the variational lower bound $\mathcal{L}$. The standard approach to this problem is the Mean Field (MF) ansatz, which consists in assuming a factorized form of $q$ over a subset of latent variables:

$$
q(\bm{z}|\bm{x},\bm{\phi}) = \prod_i q(\bm{z_i}|\bm{x},\bm{\phi}). \tag{5}
$$

The problem is that MF approach involves the calculations of integrals depending on $q_i$ that in many cases are **intractable**, thus limiting the range if its applicability. Starting from these observations, the article focuses on finding a solution to the problems that arise consequently:
- efficient approximate estimation for the parameters $\bm{\theta}$
- efficient approximate estimation of $p(\bm{z}|\bm{x}, \bm{\theta})$
- efficient approximate esitmation of $p(\bm{x}|\bm{\theta})$.

Then, assuming that the MF approach is not doable, in our new scenario:
- $q(\bm{z}|\bm{x},\bm{\phi})$ is not assumed to be factorized anymore and acts as an **encoder** since starting from a datapoint $\bm{x}$ it provides a distribution over the the latent space variables
- $p(\bm{x}|\bm{z},\bm{\theta})$ acts as a **decoder** since starting from an encoded datapoint $\bm{z}$ living in the latent space it provides a distribution over the possible values of $\bm{x}$ that may correspond to it.

Of course, we re left with the problem of learning the parameters that define our probabilistic encoder and decoder. To deal with it, it is useful to rewrite the variational lower bound $(4)$ as 

$$
\mathcal{L}(\bm{\theta}, \bm{\phi}; \bm{x}^{(i)}) = -D_{KL}(q(\bm{z}|\bm{x}^{(i)},\bm{\phi})||p(\bm{z}|\bm{\theta})) + \mathbb{E}_{q_{\phi}} \left[\log p(\bm{x}^{(i)}|\bm{z}, \bm{\theta})\right] \tag{6}
$$

and thus our problem becomes the following **optimization**:

$$
\hat{\bm{\phi}}, \hat{\bm{\theta}} = \argmax_{\bm{\phi}, \bm{\theta}} \mathcal{L}(\bm{\theta}, \bm{\phi}; \bm{x}). \tag{7}
$$

In order to compute the expectation value, the article introduces the **reparametrization trick** to avoid estimates having a large variance that would undermine the optimization. This trick consists in setting

$$
\tilde{\mathcal{L}}^B(\bm{\theta}, \bm{\phi}; \bm{x}^{(i)}) = -D_{KL}(q(\bm{z}|\bm{x}^{(i)},\bm{\phi})||p(\bm{z}|\bm{\theta})) + \frac{1}{L} \sum_{l=1}^L \log p(\bm{x}^{(i)}|\bm{z_l}^{(i,l)}, \bm{\theta}) \tag{8}
$$

where $\bm{z}_l = g_{\bm{\phi}}(\bm{\epsilon_l}, \bm{x})$ is a deterministic differentiable transformation and $\bm{\epsilon}_l \sim p(\bm{\epsilon})$ is a random noise vector chosen such that it ensures $\bm{z}_l \sim q(\bm{z}|\bm{x},\bm{\phi})$. The estimator of the variational lower bound provided by Eq. $(8)$ is called **Stochastic Gradient Variational Bayes** (SGVB) estimator.

## VAE Implementation

The main focus of the article is showing the implementation of the aforementioned architecture in which both **the encoder and the decoder consist in neural networks**, more precisely Multi-Layer Perceptron (MLP) blocks, i.e. fully-connected blocks with only one hidden layer between input and output. This allows optimizing the lower bound $(8)$ by exploiting the most comon gradient ascent algorithms. The implementation also involves:
1. setting the prior as a multivariate normal with zero mean and an identity as covariance matrix,
$$
p(\bm{z}|\bm{\theta}) = \mathcal{N}(\bm{0}, \bm{I}) \tag{9}
$$
2. assuming that the approximate posterior is Gaussian with diagional identity matrix
$$
q(\bm{z}|\bm{x}^{(i)},\bm{\phi}) \equiv \mathcal{N}(\bm{z}|\bm{\mu}_\phi(\bm{x}^{(i)}), diag(\bm{\sigma}^2_\phi(\bm{x}^{(i)}))) \tag{10}
$$
3. assuming that the conditional likelihood is also Gaussian with diagonal identity matrix
$$
p(\bm{x}^{(i)}|\bm{z}^{(i,l)}, \bm{\theta}) \equiv \mathcal{N}(\bm{x}^{(i)}|\bm{\mu}_\theta(\bm{z}^{(i,l)}), diag(\bm{\sigma}^2_\theta(\bm{z}^{(i,l)}))) \tag{11}
$$

In this way, the output of the encoder and of the decoder will be $\bm{\mu}_\phi(\bm{x}), \bm{\sigma}^2_\phi(\bm{x})$ and $\bm{\mu}_\theta(\bm{z_l}), \bm{\sigma}^2_\theta(\bm{z}_l)$ respectively. Moreover, thanks to these choices for the distributions, we have that the Kullback-Leibler divergence term assumes an analytical form, allowing us to obtain a **convenient form of the variational lower** bound that we can directly exploit inside the code of a **training loop**:

$$
\tilde{\mathcal{L}}^B(\bm{\theta}, \bm{\phi}; \bm{x}^{(i)}) = \frac{1}{2} \sum_{j=1}^J \left(1 + \log((\bm{\sigma}^{(i)}_\phi)^2_j) - (\bm{\mu}^{(i)}_\phi)_j^2 - (\bm{\sigma}^{(i)}_\phi)_j^2\right) + \frac{1}{L} \sum_{l=1}^L \log \mathcal{N}(\bm{x}^{(i)}|\bm{\mu}_\theta(\bm{z}^{(i,l)}), diag(\bm{\sigma}^2_\theta(\bm{z}^{(i,l)}))) \tag{12}
$$

with

$$
\bm{z}^{(i,l)} = \bm{\mu}^{(i)}_\phi + \bm{\sigma}^{(i)}_\phi \odot \bm{\epsilon^{(l)}}, \qquad \epsilon^{(l)} \sim \mathcal{N}(\bm{0}, \bm{I})
$$

and 

$$
\begin{align}
\bm{\mu} &=& \bm{W}_4 \bm{h} + \bm{b}_4 \nonumber \\
\log \bm{\sigma}^2 &=& \bm{W}_5 \bm{h} + \bm{b}_5 \nonumber \\
h &=& \text{ReLU}(\bm{W}_3 \bm{x} + \bm{b}_3) \nonumber \tag{13}
\end{align}
$$

In particular, the set of equations $(13)$ holds both for the encoder and decoder parameters, provided that $\bm{x}$ is swapped with $\bm{z}$ in the equation for $\bm{h}$.

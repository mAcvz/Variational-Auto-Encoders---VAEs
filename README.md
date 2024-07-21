## Introduction and theoretical background

**Variational Auto-Encoders (VAEs)** are an inference paradigm invented by D. P. Kingma and M. Welling with the goal of performing efficient approximate inference and learning with models characterized by continuous latent variables and/or parameters that have intractable posterior distributions. In order to address VAEs, it is worth starting from the basic idea of Variational Bayes, i.e. approximating the target posterior distribution with a distribution

$$
q(\mathbf{z}|\mathbf{x},\mathbf{\phi}), \tag{1}
$$

belonging to a certain family of functions $Q$, where $\mathbf{z}$ and $\mathbf{\phi}$ are the latent variables and the function parameters respectively. If 

$$
p(\mathbf{x}, \mathbf{z}|\mathbf{\theta}) \tag{2}
$$

represents the joint probability distribution of our visible $\mathbf{x}$ and hidden $\mathbf{z}$ units, given the model parameters $\mathbf{\theta}$, the marginal probability distribution of $\mathbf{x}$ satisfies

$$
\log p(\mathbf{x}|\mathbf{\theta}) = D_{KL}(q(\mathbf{z}|\mathbf{x},\mathbf{\phi})||p(\mathbf{z}|\mathbf{x}, \mathbf{\theta})) + \mathcal{L}(\mathbf{\theta}, \mathbf{\phi}; \mathbf{x}) \tag{3}
$$

with 

$$
\mathcal{L}(\bm{\theta}, \bm{\phi}; \bm{x}) = \mathbb{E}_{q_{\phi}} \left[\log p(\bm{x}, \bm{z}|\bm{\theta}) - \log q(\bm{z}|\bm{x},\bm{\phi})\right] \tag{4}
$$

being the **variational lower bound** of the marginal likelihood. Our goal is finding $q$ such that the Kullback-Leibler divergence term is minimum and this minimization problem is equivalent to the maximization of the variational lower bound $\mathcal{L}$. The standard approach to this problem is the Mean Field (MF) ansatz, which consists in assuming a factorized form of $q$ over a subset of latent variables:

$$
q(\mathbf{z}|\mathbf{x},\mathbf{\phi}) = \prod_i q(\mathbf{z_i}|\mathbf{x},\mathbf{\phi}). \tag{5}
$$

The problem is that MF approach involves the calculations of integrals depending on $q_i$ that in many cases are **intractable**, thus limiting the range if its applicability. Starting from these observations, the article focuses on finding a solution to the problems that arise consequently:
- efficient approximate estimation for the parameters $\mathbf{\theta}$
- efficient approximate estimation of $p(\mathbf{z}|\mathbf{x}, \mathbf{\theta})$
- efficient approximate esitmation of $p(\mathbf{x}|\mathbf{\theta})$.

Then, assuming that the MF approach is not doable, in our new scenario:
- $q(\mathbf{z}|\mathbf{x},\mathbf{\phi})$ is not assumed to be factorized anymore and acts as an **encoder** since starting from a datapoint $\mathbf{x}$ it provides a distribution over the the latent space variables
- $p(\mathbf{x}|\mathbf{z},\mathbf{\theta})$ acts as a **decoder** since starting from an encoded datapoint $\mathbf{z}$ living in the latent space it provides a distribution over the possible values of $\mathbf{x}$ that may correspond to it.

Of course, we re left with the problem of learning the parameters that define our probabilistic encoder and decoder. To deal with it, it is useful to rewrite the variational lower bound $(4)$ as 

$$
\mathcal{L}(\mathbf{\theta}, \mathbf{\phi}; \mathbf{x}^{(i)}) = -D_{KL}(q(\mathbf{z}|\mathbf{x}^{(i)},\mathbf{\phi})||p(\mathbf{z}|\mathbf{\theta})) + \mathbb{E}_{q_{\phi}} \left[\log p(\mathbf{x}^{(i)}|\mathbf{z}, \mathbf{\theta})\right] \tag{6}
$$

and thus our problem becomes the following **optimization**:

$$
\hat{\mathbf{\phi}}, \hat{\mathbf{\theta}} = \argmax_{\mathbf{\phi}, \mathbf{\theta}} \mathcal{L}(\mathbf{\theta}, \mathbf{\phi}; \mathbf{x}). \tag{7}
$$

In order to compute the expectation value, the article introduces the **reparametrization trick** to avoid estimates having a large variance that would undermine the optimization. This trick consists in setting

$$
\tilde{\mathcal{L}}^B(\mathbf{\theta}, \mathbf{\phi}; \mathbf{x}^{(i)}) = -D_{KL}(q(\mathbf{z}|\mathbf{x}^{(i)},\mathbf{\phi})||p(\mathbf{z}|\mathbf{\theta})) + \frac{1}{L} \sum_{l=1}^L \log p(\mathbf{x}^{(i)}|\mathbf{z_l}^{(i,l)}, \mathbf{\theta}) \tag{8}
$$

where $\mathbf{z}_l = g_{\mathbf{\phi}}(\mathbf{\epsilon_l}, \mathbf{x})$ is a deterministic differentiable transformation and $\mathbf{\epsilon}_l \sim p(\mathbf{\epsilon})$ is a random noise vector chosen such that it ensures $\mathbf{z}_l \sim q(\mathbf{z}|\mathbf{x},\mathbf{\phi})$. The estimator of the variational lower bound provided by Eq. $(8)$ is called **Stochastic Gradient Variational Bayes** (SGVB) estimator.

## VAE Implementation

The main focus of the article is showing the implementation of the aforementioned architecture in which both **the encoder and the decoder consist in neural networks**, more precisely Multi-Layer Perceptron (MLP) blocks, i.e. fully-connected blocks with only one hidden layer between input and output. This allows optimizing the lower bound $(8)$ by exploiting the most comon gradient ascent algorithms. The implementation also involves:
1. setting the prior as a multivariate normal with zero mean and an identity as covariance matrix,
$$
p(\mathbf{z}|\mathbf{\theta}) = \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{9}
$$
2. assuming that the approximate posterior is Gaussian with diagional identity matrix
$$
q(\mathbf{z}|\mathbf{x}^{(i)},\mathbf{\phi}) \equiv \mathcal{N}(\mathbf{z}|\mathbf{\mu}_\phi(\mathbf{x}^{(i)}), diag(\mathbf{\sigma}^2_\phi(\mathbf{x}^{(i)}))) \tag{10}
$$
3. assuming that the conditional likelihood is also Gaussian with diagonal identity matrix
$$
p(\mathbf{x}^{(i)}|\mathbf{z}^{(i,l)}, \mathbf{\theta}) \equiv \mathcal{N}(\mathbf{x}^{(i)}|\mathbf{\mu}_\theta(\mathbf{z}^{(i,l)}), diag(\mathbf{\sigma}^2_\theta(\mathbf{z}^{(i,l)}))) \tag{11}
$$

In this way, the output of the encoder and of the decoder will be $\mathbf{\mu}_\phi(\mathbf{x}), \mathbf{\sigma}^2_\phi(\mathbf{x})$ and $\mathbf{\mu}_\theta(\mathbf{z_l}), \mathbf{\sigma}^2_\theta(\mathbf{z}_l)$ respectively. Moreover, thanks to these choices for the distributions, we have that the Kullback-Leibler divergence term assumes an analytical form, allowing us to obtain a **convenient form of the variational lower** bound that we can directly exploit inside the code of a **training loop**:

$$
\tilde{\mathcal{L}}^B(\mathbf{\theta}, \mathbf{\phi}; \mathbf{x}^{(i)}) = \frac{1}{2} \sum_{j=1}^J \left(1 + \log((\mathbf{\sigma}^{(i)}_\phi)^2_j) - (\mathbf{\mu}^{(i)}_\phi)_j^2 - (\mathbf{\sigma}^{(i)}_\phi)_j^2\right) + \frac{1}{L} \sum_{l=1}^L \log \mathcal{N}(\mathbf{x}^{(i)}|\mathbf{\mu}_\theta(\mathbf{z}^{(i,l)}), diag(\mathbf{\sigma}^2_\theta(\mathbf{z}^{(i,l)}))) \tag{12}
$$

with

$$
\mathbf{z}^{(i,l)} = \mathbf{\mu}^{(i)}_\phi + \mathbf{\sigma}^{(i)}_\phi \odot \mathbf{\epsilon^{(l)}}, \qquad \epsilon^{(l)} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

and 

$$
\begin{align}
\mathbf{\mu} &=& \mathbf{W}_4 \mathbf{h} + \mathbf{b}_4 \nonumber \\
\log \mathbf{\sigma}^2 &=& \mathbf{W}_5 \mathbf{h} + \mathbf{b}_5 \nonumber \\
h &=& \text{ReLU}(\mathbf{W}_3 \mathbf{x} + \mathbf{b}_3) \nonumber \tag{13}
\end{align}
$$

In particular, the set of equations $(13)$ holds both for the encoder and decoder parameters, provided that $\mathbf{x}$ is swapped with $\mathbf{z}$ in the equation for $\mathbf{h}$.

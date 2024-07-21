
# Variational Auto-Encoders (VAEs)
Auto-Encoding Variational Bayes
Diederik P Kingma, Max Welling

https://doi.org/10.48550/arXiv.1312.6114


## Introduction and Theoretical Background

**Variational Auto-Encoders (VAEs)** are an inference paradigm invented by D. P. Kingma and M. Welling with the goal of performing efficient approximate inference and learning with models characterized by continuous latent variables and/or parameters that have intractable posterior distributions. 

In order to address VAEs, it is worth starting from the basic idea of Variational Bayes, i.e. approximating the target posterior distribution with a distribution

$$
q({z}|{x},{\phi}), \tag{1}
$$

belonging to a certain family of functions  $Q$, where $z$ and ${\phi}$ are the latent variables and the function parameters respectively. If 

$$
p({x}, {z}|{\theta}) \tag{2}
$$

represents the joint probability distribution of our visible $x$  and hidden $z$ units, given the model parameters $\theta $, the marginal probability distribution of x satisfies

$$
\log p({x}|{\theta}) = D_{KL}(q({z}|{x},{\phi})||p({z}|{x}, {\theta})) + \mathcal{L}({\theta}, {\phi}; {x}) \tag{3}
$$

with 

$$
\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q} \left[\log p(x, z|\theta) - \log q(z|x,\phi)\right] \tag{4}
$$

being the **variational lower bound** of the marginal likelihood. Our goal is finding $q$ such that the Kullback-Leibler divergence term is minimized, and this minimization problem is equivalent to the maximization of the variational lower bound $\mathcal{L} $. The standard approach to this problem is the Mean Field (MF) ansatz, which consists in assuming a factorized form of  $q$ over a subset of latent variables:

$$
q({z}|{x},{\phi}) = \prod_i q({z_i}|{x},{\phi}). \tag{5}
$$

The problem is that the MF approach involves the calculation of integrals depending on $q_i$ that in many cases are **intractable**, thus limiting the range of its applicability. Starting from these observations, the article focuses on finding a solution to the problems that arise consequently:
- Efficient approximate estimation for the parameters $\theta$
- Efficient approximate estimation of $p(z|x,\theta)$
- Efficient approximate estimation of $p(x| \theta)$.

Then, assuming that the MF approach is not doable, in our new scenario:

- $q(z|x,\phi)$  is not assumed to be factorized anymore and acts as an encoder since starting from a datapoint  $x$  it provides a distribution over the latent space variables.
- $p(x|z,\theta)$  acts as a decoder since starting from an encoded datapoint  $z$  living in the latent space it provides a distribution over the possible values of  $x$  that may correspond to it.

Of course, we are left with the problem of learning the parameters that define our probabilistic encoder and decoder. To deal with it, it is useful to rewrite the variational lower bound (4) as 

$$
\mathcal{L}(\theta, \phi; x^{(i)}) = -D_{KL}(q(z|x^{(i)},\phi)||p(z|\theta)) + \mathbb{E}{q_{\phi}} \left[\log p(x^{(i)}|z, \theta)\right] \tag{6}
$$

and thus our problem becomes the following **optimization**:

$$
\hat{{\phi}}, \hat{{\theta}} = \arg\max_{{\phi}, {\theta}} \mathcal{L}({\theta}, {\phi}; {x}). \tag{7}
$$

In order to compute the expectation value, the article introduces the **reparametrization trick** to avoid estimates having a large variance that would undermine the optimization. This trick consists in setting

$$
\tilde{\mathcal{L}}^B({\theta}, {\phi}; {x}^{(i)}) = -D_{KL}(q({z}|{x}^{(i)},{\phi})||p({z}|{\theta})) + \frac{1}{L} \sum_{l=1}^L \log p({x}^{(i)}|{z_l}^{(i,l)}, {\theta}) \tag{8}
$$

where $z_l = g_{\phi}(\epsilon_l, x)$ is a deterministic differentiable transformation and  $\epsilon_l \sim p(\epsilon)$  is a random noise vector chosen such that it ensures  $z_l \sim q(z|x,\phi)$ . The estimator of the variational lower bound provided by Eq. (8) is called Stochastic Gradient Variational Bayes (SGVB) estimator.


## VAE Implementation
The main focus of the article is showing the implementation of the aforementioned architecture in which both **the encoder and the decoder consist in neural networks**, more precisely Multi-Layer Perceptron (MLP) blocks, i.e. fully-connected blocks with only one hidden layer between input and output. This allows optimizing the lower bound $(8)$ by exploiting the most comon gradient ascent algorithms. The implementation also involves:

Setting the prior as a multivariate normal with zero mean and an identity as covariance matrix:

$$
  p(z | \theta) = \mathcal{N} (0, I \tag{9} ) 
$$
  
Assuming that the approximate posterior is Gaussian with diagonal identity matrix:

$$
   q(z|x^{(i)},\phi) \equiv \mathcal{N}(z|\mu_{\phi(x^{(i)}}), \text{diag}(\sigma^2_\phi(x^{(i)}))) \tag{10}
$$

Assuming that the conditional likelihood is also Gaussian with diagonal identity matrix:

$$
   p(x^{(i)}|z^{(i,l)}, \theta) \equiv \mathcal{N}(x^{(i)}|\mu_{\theta(z^{(i,l)}}), \text{diag}(\sigma^2_\theta(z^{(i,l)}))) \tag{11}
$$




In this way, the output of the encoder and of the decoder will be $\mu_\phi(x), \sigma^2_\phi(x)$  and $\mu_{\theta}(z_l), \sigma^2_{\theta}(z_l)$ respectively. Moreover, thanks to these choices for the distributions, we have that the Kullback-Leibler divergence term assumes an analytical form, allowing us to obtain a convenient form of the variational lower bound that we can directly exploit inside the code of a training loop:

$$
\tilde{\mathcal{L}}^B(\theta, \phi; x^{(i)}) = \frac{1}{2} \sum_{j=1}^J \left(1 + \log((\sigma^{(i)}\phi)^2_j) - (\mu^{(i)}\phi)j^2 - (\sigma^{(i)}\phi)j^2\right) + \frac{1}{L} \sum{l=1}^L \log \mathcal{N}(x^{(i)}|\mu_\theta(z^{(i,l)}), \text{diag}(\sigma^2_\theta(z^{(i,l)}))) \tag{12}
$$

with

$$
z^{(i,l)} = \mu^{(i)}\phi + \sigma^{(i)}\phi \odot \epsilon^{(l)}, \qquad \epsilon^{(l)} \sim \mathcal{N}(0, I)
$$

and



$$
\mu = W_4 h + b_4
$$
	
$$
\log \sigma^2 = W_5 h + b_5
$$


$$
h = \text{ReLU}(W_3 x + b_3)
$$


In particular, the last set of equations holds both for the encoder and decoder parameters, provided that x is swapped with z in the equation for h.


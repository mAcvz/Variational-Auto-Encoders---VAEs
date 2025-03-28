# Variational Autoencoders (VAEs) — Simplified Summary

This repository provides a simplified overview of the key ideas in:

**Auto-Encoding Variational Bayes**  
Diederik P. Kingma & Max Welling  
DOI: [10.48550/arXiv.1312.6114](https://doi.org/10.48550/arXiv.1312.6114)

---

## 🧠 What is a VAE?
A **Variational Autoencoder (VAE)** is a type of neural network that learns to generate new data points by encoding input data into a latent space and then decoding it back to the original space.

It combines:
- **Probabilistic inference** (Bayesian approach)
- **Neural networks** (encoder and decoder)

---

## 🔧 Key Components

### Encoder
The encoder maps input \( x \) to a distribution over latent variables \( z \):

```
q(z | x, φ) ≈ N(μ(x), σ²(x))
```

### Decoder
The decoder reconstructs \( x \) from a sample of \( z \):

```
p(x | z, θ) ≈ N(μ(z), σ²(z))
```

### Prior
We assume a standard normal prior:

```
p(z) = N(0, I)
```

---

## 🧮 Loss Function (What We Optimize)
We want to maximize the Evidence Lower Bound (ELBO), which consists of two terms:

1. **Reconstruction Loss** (how well we reconstruct the input)
2. **KL Divergence** (how close the learned distribution is to the prior)

The loss function becomes:

```
Loss = KL[q(z | x) || p(z)] - E_q[log p(x | z)]
```

With Gaussians, the KL term has a closed-form solution and the second term is estimated using samples from \( z \).

---

## 🔁 Reparametrization Trick
We sample \( z \) using:

```
z = μ + σ ⊙ ε,  where  ε ~ N(0, I)
```

This trick makes the sampling step differentiable so we can train the network with backpropagation.

---

## 🧩 Architecture
Both encoder and decoder use simple feedforward neural networks (MLPs):

```
h = ReLU(W₁ x + b₁)
μ = W₂ h + b₂
log σ² = W₃ h + b₃
```

---

## ✅ Summary
- VAEs learn to encode data into a latent space and reconstruct it.
- Training is done by maximizing a lower bound on the data likelihood.
- It’s a powerful method for generative modeling and unsupervised learning.

---

## 📚 Reference
- [Auto-Encoding Variational Bayes (Kingma & Welling, 2013)](https://arxiv.org/abs/1312.6114)

---

Feel free to clone this repo and play with the implementation!

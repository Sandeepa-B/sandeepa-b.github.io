---
title: 'Variational Inference'
date: 2021-9-15
showexcerpt: True
tags:
  - Learning
---

# Variational Inference

The goal of variational inference is to approximate a conditional intensity of latent variables (I prefer the word *hidden variable* instead) given observed variable. Instead of directly estimate the density, we would like to find its **best approximation with smallest KL divergence** from a family of candidate densities $\mathscr{L}$.

## Problem of approximate inference
Let $\textbf{x}=\{x_i\}^{n}_{i=1}$ be a set of observed variables and $\textbf{z}=\{x_i\}^{n}_{i=1}$ be a set of hidden variables, with joint probability of $p(\textbf{x}, \textbf{z})$. 

- **Inference problem** is to compute the conditional density of the hidden variables given the observations, aka. $p(\textbf{z}|\textbf{x})$  

We can write the conditional density as 

$$
p(\textbf{z}|\textbf{x}) = \frac{p(\textbf{z},\textbf{x})}{p(\textbf{x})} = \frac{p(\textbf{z},\textbf{x})}{\int p(\textbf{z},\textbf{x}) \text{d}\textbf{x}}
$$

The denominator contains the marginal density $p(\textbf{x})$ of the observations, which is computed by integral over the hidden variable from the joint density. We also call $p(\textbf{x})$ the *evidence*. In general, computing integral is hard. Now we introduce one practical example of mixture gaussian model

### Bayesian mixture of Gaussians
Consider a mixture of $K$ unit-variance (variance equals 1) univariate (single variable) Gaussians. The means of $i$'s Gaussian distribution is $\mu_i$, $\mathbf{\mu}=\{\mu_1, \dots, \mu_K\}$. Each mean parameter is sampled from a common prior distribution $p(\mu)$, which we assume $p(\mu)=\mathcal{N}(0,\sigma^2)$. To generate an observation $x_i$ from the model, we first choose a cluster assignment $\mathbf{c}_i=[0,\dots,1,\dots,0]$ (1 at the $c_i$'s position) from a Categorical (uniform) distribution, which means that $x_i$ comes from mixture $c_i$. We then draw $x_i$ from mixture $c_i$, $x_i \sim \mathcal{N}(\mathbf{c}_i^\top \mathbf{\mu}, 1)$

The full model is 
$$
\mu_i \sim \mathcal{N}(0,\sigma^2), i=1,\dots,K\\
c_i \sim \text{Categorical}(\frac{1}{K}, \dots, \frac{1}{K}), i=1,\dots,n\\
p(x_i|c_i, \mathbf{\mu}) = \mathcal{N}(\mathbf{c}^\top_i\mathbf{\mu},1)
$$

The joint density of latent variable is 
$$
p(\mathbf{\mu}, \mathbf{c}, \mathbf{x}) = \prod_{i=1}^n p(x_i,c_i, \mathbf{\mu}) \\
= \prod_{i=1}^n p(x_i|c_i, \mathbf{\mu}) p(c_i, \mathbf{\mu})\\
= \prod_{i=1}^n p(x_i|c_i, \mathbf{\mu}) p(c_i) p(\mathbf{\mu})\\
= p(\mathbf{\mu}) \prod_{i=1}^n p(x_i|c_i, \mathbf{\mu}) p(c_i) \\
$$

Given the observed $\mathbf{x}$, our hidden variables are $\mathbf{z} = \{\mathbf{\mu}, \mathbf{c}\}$. Hence the evidence integral is 

$$
p(\mathbf{x}) = \sum_{c_i}\int p(\mathbf{\mu}) \prod_{i=1}^n p(x_i|c_i, \mathbf{\mu})  p(c_i) \text{d}\mathbf{\mu} \\
= \sum_{c_i}p(c_i)\int p(\mathbf{\mu}) \prod_{i=1}^n p(x_i|c_i, \mathbf{\mu})  \text{d}\mathbf{\mu}
$$

## The evidence lower bound (ELBO)

In variational inference, we specify a family $\mathscr{L}$ of density over the hidden variables. Our goal is to find the best $q(\mathbf{z}) \in \mathscr{L}$ with the smallest KL divergence to the posterior density. Inference becomes a problem of optimization

$$
q^*(\mathbf{z})=\arg\min_{q(\mathbf{z}) \in \mathscr{L}} KL(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x}))
$$

$q^*(\mathbf{z})$ is the best approximation of $p(\mathbf{z}|\mathbf{x})$. Based on the definition of KL divergence

$$
KL(q(\mathbf{z})||p(\mathbf{z}|\mathbf{x})) = \mathbb{E}[\log q(\mathbf{z})] - \mathbb{E}[\log p(\mathbf{z}|\mathbf{x})]\\
= \mathbb{E}[\log q(\mathbf{z})] - \mathbb{E}[\log \frac{p(\mathbf{z},\mathbf{x})}{p(\mathbf{x})}]\\
= \mathbb{E}[\log q(\mathbf{z})] - \mathbb{E}[\log  p(\mathbf{z},\mathbf{x})] + \mathbb{E}[\log p(\mathbf{x})]
$$

Because all expectation are taken with respect to $\mathbf{z}$, $\mathbb{E}[\log p(\mathbf{x})] = \log p(\mathbf{x})$. So this KL divergence requires the computation of $p(\mathbf{x})$ again, which is not trackable. 

Instead of compute KL directly, we optimize an alternative objective that is equivalent to KL adding a constant

$$
\text{ELBO}(q) = \mathbb{E}[\log  p(\mathbf{z},\mathbf{x})] - \mathbb{E}[\log q(\mathbf{z})] 
$$

This function is called evidence lower bound (ELBO). It is clear that
$\text{ELBO} = - \text{KL} + \log p(\mathbf{x})$. Since $\log p(\mathbf{x})$ is a constant with respect to $q(\mathbf{z})$, maximizing ELBO is equivalent to minimizing KL.

We rewrite the formula of ELBO as a sum of log likelihood of data and KL divergence between $q(\mathbf{z})$ and $p(\mathbf{z})$

$$
\text{ELBO}(q) = \mathbb{E}[\log  p(\mathbf{z},\mathbf{x})] - \mathbb{E}[\log q(\mathbf{z})]\\
=\mathbb{E}[\log  p(\mathbf{x}|\mathbf{z})p(\mathbf{z})] - \mathbb{E}[\log q(\mathbf{z})]\\
=\mathbb{E}[\log  p(\mathbf{x}|\mathbf{z})] + \mathbb{E}[\log p(\mathbf{z})] - \mathbb{E}[\log q(\mathbf{z})]\\
=\mathbb{E}[\log  p(\mathbf{x}|\mathbf{z})] - \text{KL}[q(\mathbf{z})||p(\mathbf{z})]
$$

### What does ELBO mean?
1. $\mathbb{E}[\log  p(\mathbf{x}|\mathbf{z})]$ is the expected log likelihood. It encourages the densities of hidden variables to explain the observed data.
2. $- \text{KL}[q(\mathbf{z})||p(\mathbf{z})]$ is the the negative divergence between the variational density and the prior. It encourages the density close to the prior. 

Thus the ELBO mirrors the usual **balance between likelihood and prior**.
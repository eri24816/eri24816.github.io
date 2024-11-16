---
title: 
date: 2024-11-16
authors:
  - eri24816
image: https://i.imgur.com/5kg2YwL.png
draft: false
tags:
  - InformationTheory
categories: math
series: 
summary:
---
# The Problem
  
I was wondering how to quantitatively represent the capacity of an VAE’s latent variable (or a reconstruction step of a diffusion model) in bits, making it comparable to other distributions, such as the data distribution?

# Book on a Rod

Then, a paradox I saw like 8 years ago came to my mind:

> “Theoretically, you can store all information of a book on a rod. To do that, encode the book’s content into a decimal string, put it behind `0.` , and mark at the position of that number fraction of the rod’s length.”

What prohibits one to do that in reality is the noise occurs when inscribing and measuring the mark. So, in reality, it would be more like “You can store at most 50*x bits on a rod of x meters, with one mark on it.”

# Back to the Latent Variable

That paradox is a down-to-earth version of the following:

> “The Shannon entropy of a non-singular continuous distribution (such as Gaussian) is infinity. Therefore, the latent varible have infinity capacity.

That means, with Shannon entropy as the measure, the capacity of a VAE’s latent variable is larger than all categorical data distribution, even the latent space is only one dimensional.

Yes, we have differential entropy, which provides a convergent measure of entropy of continuous distributions. But it is not robust to scaling, and we can’t really say “the distribution has 10 bits of differential entropy”.

After days of thinking, I believe that what’s missing is some noise when we access the latent variable. The noise prevents an encoded code (the mark) to occupy infinitely small distinct subset of the latent space (the rod). With an additive Gaussian noise, we can yield a nice answer to the original problem:

---

Given a $d$-dimensional latent variable $Z\sim\mathcal{N}(0,\,s^{2}I)\,$ and a additive Gaussian noise when accessing the latent variable $N\sim\mathcal{N}(0,t^{2}I)$, the capacity of the latent variable becomes $I(Z,Z+N)=\frac{1}{2}d\log({1+\frac{s^2}{t^2}})$.

---

This measure is robust to scaling as $s$ and $t$ scale together. It can be measured in bits.

The modified VAE now looks like a noisy channel, and the equation looks like the equation in [Shannon–Hartley theorem](https://en.wikipedia.org/wiki/Shannon%E2%80%93Hartley_theorem).
---
title: "PyTorch RNN Tutorial"
date: 2022-03-09T12:54:18+08:00
draft: false
image: https://i.imgur.com/X6kty5v.jpg
tags: ["Python","torch"]
mathjax: true
---

This tutorial will demonstrate how to build and train a simple RNN model with PyTorch.

## Concepts

### What does an RNN do?
 
Given an input sequence $x=[x_1,x_2,\cdots,x_{n}]$, an RNN can generate a corresponding output sequence $\hat y=[\hat y_1,\hat y_2,\cdots,\hat y_{n}]$ 
successively. The strength of RNN is that it can "remember" its previosly seen input elements. When calculating $\hat y_i$, the model can access not only $x_i$ but also the information from $x_0$ to $x_{i-1}$, via its hidden state, $h_{i-1}$.


![Image](https://i.imgur.com/lw62OZL.png#center)

### Autoregressive model

Given the characteristics of an RNN, we can make it do something cool -- predicting a sequence. 

Imagine now we have a initial sequence [1,2,3,4,5] and feed it into an RNN. If the model is good at predicting linear sequences, it will predict some number near 6 as the next number, let's say 6.05. 

![Image](https://i.imgur.com/cCr0pSK.png)

Then we put the 6.05 back to the sequence, make it [1,2,3,4,5,6.05], and feed it into the model again. This time, the model says 7.02, so the sequence becomes [1,2,3,4,5,6.05,7.02]. As we repeat this process, the model can eventually generate a long sequence by itself.

![Image](https://i.imgur.com/UhcKdwA.png)

To be precise, when training the model, we want it predict $x_{i+1}$ after seeing the input $[x_1, x_2, \cdots,x_{i}]$. That is, the difference between its output, $\hat y_i$, and the target output, $x_{i+1}$, should be minimized.

![Image](https://i.imgur.com/jChTnLU.png#center)

We can define sequence $y =[ y_1, y_2,\cdots, y_{n}]$ as the target output, which $y_i = x_{i+1}$.

Thus, the loss function is the mean square error between $\hat y$ and $y$:

$$ \frac{1}{n}\sum_{i= 1}^{n}{(\hat y_i-x_{i+1})^2} = \frac{1}{n}\sum_{i= 1}^{n}{(\hat y_i-y_i)^2} = MSE(\hat y,y)$$.

## Implementation

### 1. Import
Import the modules we will need.
```python
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
```

### 2. Prepare the training data
First, we need the training data. 
Here we generate a sine wave of $\sin(t)$, from $t=0$ to $t=80$, and sample peirod $=0.2$. Then a little noise is added to simulate sampling error.

```python
data = np.sin(np.arange(0,80,0.2))
data += (np.random.random(data.shape)-0.5)*0.2
plt.plot(data)
```
![Image](https://i.imgur.com/07PP9iu.jpg#centers)

By default, an RNN module require its input tensor to have the shape $[ Time, Batch, Feature ]$. To keep it simple, we can just set the 
batch size and feature num to 1. So we have to expand the original data of size $[400]$ to $[ 400, 1,1 ]$ by doing unsqueeze(-1) twice.
```python
print(data.shape) #(400,)

# convert data into a torch tensor and expand the dimension of its shape
data = torch.tensor(data, dtype = torch.float).unsqueeze(-1).unsqueeze(-1)

print(data.shape) # torch.Size([400, 1, 1])
```

Make x and y that satisfy $y_i = x_{i+1}$.

```python
#input sequence
x = data[:-1]

# target output sequence
y = data[1:]
```

```python
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(1,12)
        self.linear = nn.Linear(12,1)
    def forward(self, x, h_0 = None):
        rnn_out, h_n = self.rnn(x, h_0)
        return self.linear(rnn_out), h_n
```
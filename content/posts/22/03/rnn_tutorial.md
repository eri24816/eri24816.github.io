---
title: "PyTorch RNN Tutorial"
date: 2022-03-09T12:54:18+08:00
draft: false
image: https://i.imgur.com/X6kty5v.jpg
tags: ["Python","torch"]
mathjax: true
---

This tutorial will demonstrate how to build and train a simple autoregressive RNN model with PyTorch.

## Concepts

### What does an RNN do?

Given an input sequence $[x_0,x_1,\cdots,x_{n-1}]$, an RNN can generate a corresponding output sequence $[y_0,y_1,\cdots,y_{n-1}]$ 
successively. The strength of RNN is that it can "remember" its previosly seen input elements. When calculating $y_i$, the model can use not only $x_i$, but also $h_i$ to get the information from $x_0$ to $x_{i-1}$.


![Image](https://i.imgur.com/UIwIkg8.png#center)

### Autoregressive model

Given the characteristics of an RNN, we can make it do something cool -- predicting a sequence. 

Imagine now we have a sequence, [1,2,3,4,5], and feed it into an RNN. If the model is good at predicting linear sequences, it may predict 6 as the next number (or something like 5.9 or 6.01) by 

![Image](https://i.imgur.com/1hiTAKz.jpg)

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

By default, an RNN module require its input tensor to have the shape $[ Time, Batch, Feature ]$. In our case, the training data has 
batch size feature num both set to 1. So we have to expand the original data of size $[400]$ to $[ 400, 1,1 ]$ by doing unsqueeze(-1) twice.
```python
print(data.shape) #(400,)

# convert data into a torch tensor and expand the dimension of its shape
data = torch.tensor(data, dtype = torch.float).unsqueeze(-1).unsqueeze(-1)

print(data.shape) # torch.Size([400, 1, 1])
```

To 

```python
x = data[:-1]
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